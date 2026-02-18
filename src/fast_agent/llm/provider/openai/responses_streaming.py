from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai.types.responses import (
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
)

from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.provider.openai._stream_capture import (
    save_stream_chunk as _save_stream_chunk,
)
from fast_agent.llm.provider.openai.streaming_utils import finalize_stream_response
from fast_agent.llm.provider.openai.tool_notifications import OpenAIToolNotificationMixin
from fast_agent.llm.stream_types import StreamChunk

_logger = get_logger(__name__)

_TOOL_START_EVENT_TYPES = {
    "response.web_search_call.in_progress",
    "response.web_search_call.searching",
}
_TOOL_STOP_EVENT_TYPES = {
    "response.web_search_call.completed",
    "response.web_search_call.failed",
}
_WEB_SEARCH_PROGRESS_LABEL = "Searching the web"


def _web_search_status_chunk(status: str) -> str:
    if status == "in_progress":
        return "starting search..."
    if status == "searching":
        return "searching..."
    if status == "completed":
        return "search complete"
    if status == "failed":
        return "search failed"
    return status


class ResponsesStreamingMixin(OpenAIToolNotificationMixin):
    if TYPE_CHECKING:
        from pathlib import Path

        from fast_agent.core.logging.logger import Logger

        logger: Logger
        name: str | None

        def _notify_stream_listeners(self, chunk: StreamChunk) -> None: ...

        def _notify_tool_stream_listeners(
            self, event_type: str, payload: dict[str, Any] | None = None
        ) -> None: ...

        def _update_streaming_progress(
            self, content: str, model: str, estimated_tokens: int
        ) -> int: ...

        def chat_turn(self) -> int: ...

    async def _process_stream(
        self, stream: Any, model: str, capture_filename: Path | None
    ) -> tuple[Any, list[str]]:
        estimated_tokens = 0
        reasoning_chars = 0
        reasoning_segments: list[str] = []
        tool_streams: dict[int, dict[str, Any]] = {}
        tool_streams_by_id: dict[str, dict[str, Any]] = {}
        notified_tool_indices: set[int] = set()
        closed_tool_ids: set[str] = set()
        final_response: Any | None = None

        def _item_is_tool(item: Any) -> bool:
            return getattr(item, "type", None) in {"function_call", "web_search_call"}

        def _tool_name(item: Any) -> str:
            item_type = getattr(item, "type", None)
            if item_type == "web_search_call":
                return "web_search"
            return getattr(item, "name", None) or "tool"

        def _tool_use_id(item: Any, index: int | None, item_id: str | None = None) -> str:
            tool_use = (
                getattr(item, "call_id", None)
                or getattr(item, "id", None)
                or item_id
            )
            if isinstance(tool_use, str) and tool_use:
                return tool_use
            suffix = str(index) if index is not None else "unknown"
            item_type = getattr(item, "type", None) or "tool"
            return f"{item_type}-{suffix}"

        def _register_tool(
            index: int,
            tool_name: str,
            tool_use_id: str,
            *,
            item_type: str,
            notified: bool = False,
        ) -> dict[str, Any]:
            tool_info = {
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "item_type": item_type,
                "notified": notified,
                "index": index,
            }
            tool_streams[index] = tool_info
            tool_streams_by_id[tool_use_id] = tool_info
            return tool_info

        def _resolve_tool_for_event(
            *,
            event_index: int | None,
            event_item_id: str | None,
        ) -> tuple[int, dict[str, Any]]:
            if event_index is not None and event_index in tool_streams:
                return event_index, tool_streams[event_index]

            if event_item_id and event_item_id in tool_streams_by_id:
                tool_info = tool_streams_by_id[event_item_id]
                info_index = tool_info.get("index")
                if isinstance(info_index, int):
                    return info_index, tool_info

            fallback_index = event_index if event_index is not None else -1
            fallback_id = event_item_id or f"web_search-{fallback_index}"
            tool_info = _register_tool(
                fallback_index,
                "web_search",
                fallback_id,
                item_type="web_search_call",
                notified=False,
            )
            return fallback_index, tool_info

        def _close_tool(index: int, tool_use_id: str | None) -> None:
            if index in tool_streams:
                tool_streams.pop(index, None)
            if tool_use_id:
                tool_streams_by_id.pop(tool_use_id, None)
                closed_tool_ids.add(tool_use_id)

        async for event in stream:
            _save_stream_chunk(capture_filename, event)
            event_type = getattr(event, "type", None)

            if isinstance(event, ResponseReasoningSummaryTextDeltaEvent) or event_type in {
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary.delta",
            }:
                delta = getattr(event, "delta", None)
                if delta:
                    reasoning_segments.append(delta)
                    self._notify_stream_listeners(
                        StreamChunk(text=delta, is_reasoning=True)
                    )
                    reasoning_chars += len(delta)
                    await self._emit_streaming_progress(
                        model=f"{model} (summary)",
                        new_total=reasoning_chars,
                        type=ProgressAction.THINKING,
                    )
                continue

            if isinstance(event, ResponseTextDeltaEvent) or event_type in {
                "response.output_text.delta",
                "response.text.delta",
            }:
                delta = getattr(event, "delta", None)
                if delta:
                    self._notify_stream_listeners(
                        StreamChunk(text=delta, is_reasoning=False)
                    )
                    estimated_tokens = self._update_streaming_progress(
                        delta, model, estimated_tokens
                    )
                    self._notify_tool_stream_listeners(
                        "text",
                        {
                            "chunk": delta,
                        },
                    )
                continue

            if event_type in {"response.completed", "response.incomplete", "response.done"}:
                final_response = getattr(event, "response", None) or final_response
                continue
            if event_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if _item_is_tool(item):
                    index = getattr(event, "output_index", None)
                    if index is None:
                        continue
                    tool_name = _tool_name(item)
                    item_type = getattr(item, "type", None) or "tool"
                    item_id = getattr(event, "item_id", None)
                    tool_use = _tool_use_id(item, index, item_id)
                    tool_info = _register_tool(
                        index,
                        tool_name,
                        tool_use,
                        item_type=item_type,
                    )
                    payload = {
                        "tool_name": tool_info["tool_name"],
                        "tool_use_id": tool_info["tool_use_id"],
                        "index": index,
                    }
                    if item_type == "web_search_call":
                        payload["tool_display_name"] = _WEB_SEARCH_PROGRESS_LABEL
                        payload["chunk"] = _web_search_status_chunk("in_progress")
                    self._notify_tool_stream_listeners(
                        "start",
                        payload,
                    )
                    self.logger.info(
                        "Model started streaming tool call",
                        data={
                            "progress_action": ProgressAction.CALLING_TOOL,
                            "agent_name": self.name,
                            "model": model,
                            "tool_name": tool_info["tool_name"],
                            "tool_use_id": tool_info["tool_use_id"],
                            "tool_event": "start",
                        },
                    )
                    tool_info["notified"] = True
                    notified_tool_indices.add(index)
                continue

            if event_type == "response.function_call_arguments.delta":
                index = getattr(event, "output_index", None)
                if index is None:
                    continue
                tool_info = tool_streams.get(index, {})
                self._notify_tool_stream_listeners(
                    "delta",
                    {
                        "tool_name": tool_info.get("tool_name"),
                        "tool_use_id": tool_info.get("tool_use_id"),
                        "index": index,
                        "chunk": getattr(event, "delta", None),
                    },
                )
                continue

            if event_type in (_TOOL_START_EVENT_TYPES | _TOOL_STOP_EVENT_TYPES):
                event_index = getattr(event, "output_index", None)
                event_item_id = getattr(event, "item_id", None)
                index, tool_info = _resolve_tool_for_event(
                    event_index=event_index,
                    event_item_id=event_item_id,
                )
                tool_use_id = tool_info.get("tool_use_id")
                if isinstance(tool_use_id, str) and tool_use_id in closed_tool_ids:
                    continue

                payload = {
                    "tool_name": tool_info.get("tool_name") or "web_search",
                    "tool_use_id": tool_use_id,
                    "index": index,
                    "status": event_type.rsplit(".", 1)[-1],
                }
                status = payload["status"]
                if tool_info.get("item_type") == "web_search_call":
                    payload["tool_display_name"] = _WEB_SEARCH_PROGRESS_LABEL
                    payload["chunk"] = _web_search_status_chunk(str(status))
                self._notify_tool_stream_listeners("status", payload)

                if event_type in _TOOL_START_EVENT_TYPES and not tool_info.get("notified"):
                    self._notify_tool_stream_listeners("start", payload)
                    tool_info["notified"] = True
                    if index >= 0:
                        notified_tool_indices.add(index)

                if event_type in _TOOL_STOP_EVENT_TYPES:
                    self._notify_tool_stream_listeners("stop", payload)
                    _close_tool(index, tool_use_id if isinstance(tool_use_id, str) else None)
                continue

            if event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if not _item_is_tool(item):
                    continue
                index = getattr(event, "output_index", None)
                tool_info = tool_streams.pop(index, {}) if index is not None else {}
                tool_name = _tool_name(item)
                tool_use_id = (
                    getattr(item, "call_id", None)
                    or getattr(item, "id", None)
                    or tool_info.get("tool_use_id")
                )
                if isinstance(tool_use_id, str) and tool_use_id in closed_tool_ids:
                    continue
                if index is None:
                    index = -1
                self._notify_tool_stream_listeners(
                    "stop",
                    {
                        "tool_name": tool_name,
                        "tool_use_id": tool_use_id,
                        "index": index,
                        "tool_display_name": (
                            _WEB_SEARCH_PROGRESS_LABEL
                            if getattr(item, "type", None) == "web_search_call"
                            else None
                        ),
                    },
                )
                self.logger.info(
                    "Model finished streaming tool call",
                    data={
                        "progress_action": ProgressAction.CALLING_TOOL,
                        "agent_name": self.name,
                        "model": model,
                        "tool_name": tool_name,
                        "tool_use_id": tool_use_id,
                        "tool_event": "stop",
                    },
                )
                if index >= 0:
                    notified_tool_indices.add(index)
                _close_tool(index, tool_use_id if isinstance(tool_use_id, str) else None)
                continue

        if tool_streams:
            incomplete_tools = [
                f"{info.get('tool_name', 'unknown')}:{info.get('tool_use_id', 'unknown')}"
                for info in tool_streams.values()
            ]
            self.logger.error(
                "Tool call streaming incomplete - started but never finished",
                data={
                    "incomplete_tools": incomplete_tools,
                    "tool_count": len(tool_streams),
                },
            )
            raise RuntimeError(
                "Streaming completed but tool call(s) never finished: "
                f"{', '.join(incomplete_tools)}"
            )

        if final_response is None:
            try:
                final_response = await stream.get_final_response()
            except Exception as exc:
                self.logger.warning("Failed to fetch final Responses payload", exc_info=exc)
                raise

        finalize_stream_response(
            final_response=final_response,
            model=model,
            agent_name=self.name,
            chat_turn=self.chat_turn,
            logger=self.logger,
            notified_tool_indices=notified_tool_indices,
            emit_tool_fallback=self._emit_tool_notification_fallback,
        )

        return final_response, reasoning_segments

    def _emit_tool_notification_fallback(
        self,
        output_items: list[Any],
        notified_indices: set[int],
        *,
        model: str,
    ) -> None:
        """Emit start/stop notifications when streaming metadata was missing."""
        if not output_items:
            return

        for index, item in enumerate(output_items):
            if index in notified_indices:
                continue
            if getattr(item, "type", None) != "function_call":
                if getattr(item, "type", None) != "web_search_call":
                    continue

            if getattr(item, "type", None) == "web_search_call":
                tool_name = "web_search"
                tool_use_id = getattr(item, "id", None) or f"tool-{index}"
            else:
                tool_name = getattr(item, "name", None) or "tool"
                tool_use_id = (
                    getattr(item, "call_id", None)
                    or getattr(item, "id", None)
                    or f"tool-{index}"
                )

            self._emit_fallback_tool_notification_event(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                index=index,
                model=model,
            )

    async def _emit_streaming_progress(
        self,
        model: str,
        new_total: int,
        type: ProgressAction = ProgressAction.STREAMING,
    ) -> None:
        """Emit a streaming progress event.

        Args:
            model: The model being used.
            new_total: The new total token count.
        """
        token_str = str(new_total).rjust(5)

        data = {
            "progress_action": type,
            "model": model,
            "agent_name": self.name,
            "chat_turn": self.chat_turn(),
            "details": token_str.strip(),
        }
        self.logger.info("Streaming progress", data=data)
