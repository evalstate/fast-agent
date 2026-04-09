from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai.types.responses import (
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
)

from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.history.tool_activities import tool_activity_display_title
from fast_agent.llm.provider.openai._stream_capture import (
    save_stream_chunk as _save_stream_chunk,
)
from fast_agent.llm.provider.openai.streaming_utils import fetch_and_finalize_stream_response
from fast_agent.llm.provider.openai.tool_notifications import OpenAIToolNotificationMixin
from fast_agent.llm.provider.openai.tool_stream_state import OpenAIToolStreamState
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.utils.reasoning_chunk_join import normalize_reasoning_delta

_logger = get_logger(__name__)

_TOOL_START_EVENT_TYPES = {
    "response.web_search_call.in_progress",
    "response.web_search_call.searching",
    "response.mcp_list_tools.in_progress",
    "response.mcp_call.in_progress",
}
_TOOL_STOP_EVENT_TYPES = {
    "response.web_search_call.completed",
    "response.web_search_call.failed",
    "response.mcp_list_tools.completed",
    "response.mcp_list_tools.failed",
    "response.mcp_call.completed",
    "response.mcp_call.failed",
}
_WEB_SEARCH_PROGRESS_LABEL = "Searching the web"
_MCP_LIST_TOOLS_PROGRESS_LABEL = "Loading MCP tools"


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


def _mcp_status_chunk(item_type: str, status: str) -> str:
    if item_type == "mcp_list_tools":
        if status == "in_progress":
            return "loading remote tool definitions..."
        if status == "completed":
            return "remote tool definitions loaded"
        if status == "failed":
            return "failed to load remote tool definitions"
        return status
    if item_type == "mcp_call":
        if status == "in_progress":
            return "calling remote MCP tool..."
        if status == "completed":
            return "remote MCP tool call complete"
        if status == "failed":
            return "remote MCP tool call failed"
        return status
    return status


def _mcp_call_display_title(tool_name: str, *, status: str | None = None) -> str:
    display_name = tool_name.split("/", 1)[-1]
    kind = "result" if status in {"completed", "failed", "cancelled", "incomplete"} else "call"
    return tool_activity_display_title(kind=kind, tool_name=display_name, is_remote=True)


def _mcp_call_output_chunk(output: Any) -> str | None:
    if output is None:
        return None
    if isinstance(output, str):
        preview = output.strip()
    else:
        try:
            import json

            preview = json.dumps(output)
        except Exception:
            preview = str(output).strip()
    if not preview:
        return None
    if len(preview) > 120:
        return f"{preview[:117]}..."
    return preview


def _item_is_responses_tool(item: Any) -> bool:
    return getattr(item, "type", None) in {
        "function_call",
        "custom_tool_call",
        "web_search_call",
        "mcp_list_tools",
        "mcp_call",
    }


def _responses_tool_name(item: Any) -> str:
    item_type = getattr(item, "type", None)
    if item_type == "web_search_call":
        return "web_search"
    if item_type == "mcp_list_tools":
        server_label = getattr(item, "server_label", None)
        if isinstance(server_label, str) and server_label:
            return f"{server_label}/mcp_list_tools"
        return "mcp_list_tools"
    if item_type == "mcp_call":
        tool_name = getattr(item, "name", None) or getattr(item, "tool_name", None)
        server_label = getattr(item, "server_label", None)
        if isinstance(server_label, str) and server_label and isinstance(tool_name, str) and tool_name:
            return f"{server_label}/{tool_name}"
        return tool_name or "mcp_call"
    return getattr(item, "name", None) or "tool"


def _responses_tool_use_id(item: Any, index: int | None, item_id: str | None = None) -> str:
    tool_use = getattr(item, "call_id", None) or getattr(item, "id", None) or item_id
    if isinstance(tool_use, str) and tool_use:
        return tool_use
    suffix = str(index) if index is not None else "unknown"
    item_type = getattr(item, "type", None) or "tool"
    return f"{item_type}-{suffix}"


def _tool_progress_display(item: Any) -> tuple[str | None, str | None]:
    item_type = getattr(item, "type", None)
    if item_type == "web_search_call":
        return _WEB_SEARCH_PROGRESS_LABEL, _web_search_status_chunk("in_progress")
    if item_type == "mcp_list_tools":
        return _MCP_LIST_TOOLS_PROGRESS_LABEL, "loading remote tool definitions..."
    if item_type == "mcp_call":
        arguments = getattr(item, "arguments", None)
        return _mcp_call_display_title(_responses_tool_name(item)), (
            arguments if isinstance(arguments, str) and arguments else None
        )
    return None, None


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

        def _emit_stream_text_delta(
            self,
            *,
            text: str,
            model: str,
            estimated_tokens: int,
        ) -> int: ...

        def chat_turn(self) -> int: ...

    async def _process_stream(
        self, stream: Any, model: str, capture_filename: Path | None
    ) -> tuple[Any, list[str]]:
        estimated_tokens = 0
        reasoning_chars = 0
        reasoning_segments: list[str] = []
        tool_state = OpenAIToolStreamState()
        notified_tool_indices: set[int] = set()
        final_response: Any | None = None

        async for event in stream:
            _save_stream_chunk(capture_filename, event)
            event_type = getattr(event, "type", None)

            if isinstance(event, ResponseReasoningSummaryTextDeltaEvent) or event_type in {
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary.delta",
            }:
                delta = getattr(event, "delta", None)
                if delta:
                    last_char = (
                        reasoning_segments[-1][-1]
                        if reasoning_segments and reasoning_segments[-1]
                        else None
                    )
                    normalized_delta = normalize_reasoning_delta(last_char, delta)
                    if not normalized_delta:
                        continue
                    reasoning_segments.append(normalized_delta)
                    self._notify_stream_listeners(
                        StreamChunk(text=normalized_delta, is_reasoning=True)
                    )
                    reasoning_chars += len(normalized_delta)
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
                    estimated_tokens = self._emit_stream_text_delta(
                        text=delta,
                        model=model,
                        estimated_tokens=estimated_tokens,
                    )
                continue

            if event_type in {"response.completed", "response.incomplete", "response.done"}:
                final_response = getattr(event, "response", None) or final_response
                continue
            if event_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if _item_is_responses_tool(item):
                    index = getattr(event, "output_index", None)
                    if index is None:
                        continue
                    item_type = getattr(item, "type", None) or "tool"
                    tool_info = tool_state.register(
                        tool_use_id=_responses_tool_use_id(
                            item,
                            index,
                            getattr(event, "item_id", None),
                        ),
                        name=_responses_tool_name(item),
                        index=index,
                        item_id=getattr(event, "item_id", None),
                        item_type=item_type,
                        kind="web_search" if item_type == "web_search_call" else "tool",
                    )
                    tool_info.argument_snapshot_present = (
                        item_type == "mcp_call"
                        and isinstance(getattr(item, "arguments", None), str)
                        and bool(getattr(item, "arguments", None))
                    )
                    payload = {
                        "tool_name": tool_info.tool_name,
                        "tool_use_id": tool_info.tool_use_id,
                        "index": index,
                    }
                    display_name, display_chunk = _tool_progress_display(item)
                    if display_name is not None:
                        payload["tool_display_name"] = display_name
                    if display_chunk is not None:
                        payload["chunk"] = display_chunk
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
                            "tool_name": tool_info.tool_name,
                            "tool_use_id": tool_info.tool_use_id,
                            "tool_event": "start",
                        },
                    )
                    tool_info.start_notified = True
                    notified_tool_indices.add(index)
                continue

            if event_type in {
                "response.function_call_arguments.delta",
                "response.custom_tool_call_input.delta",
                "response.mcp_call_arguments.delta",
            }:
                index = getattr(event, "output_index", None)
                if index is None:
                    continue
                tool_info = tool_state.resolve_open(index=index)
                if tool_info is not None and tool_info.item_type == "mcp_call":
                    event_name = (
                        "replace"
                        if (
                            not tool_info.argument_delta_received
                            and not tool_info.argument_snapshot_present
                        )
                        else "delta"
                    )
                    tool_info.argument_delta_received = True
                else:
                    event_name = "delta"
                self._notify_tool_stream_listeners(
                    event_name,
                    {
                        "tool_name": tool_info.tool_name if tool_info else None,
                        "tool_use_id": tool_info.tool_use_id if tool_info else None,
                        "index": index,
                        "chunk": getattr(event, "delta", None),
                    },
                )
                continue

            if event_type in (_TOOL_START_EVENT_TYPES | _TOOL_STOP_EVENT_TYPES):
                event_index = getattr(event, "output_index", None)
                event_item_id = getattr(event, "item_id", None)
                tool_info = tool_state.resolve_open(
                    index=event_index,
                    item_id=event_item_id,
                )
                if tool_info is None:
                    if tool_state.is_completed(index=event_index, item_id=event_item_id):
                        continue
                    fallback_index = event_index if event_index is not None else -1
                    fallback_prefix = (
                        "web_search"
                        if "web_search_call" in event_type
                        else "mcp_list_tools"
                        if "mcp_list_tools" in event_type
                        else "mcp_call"
                    )
                    tool_info = tool_state.register(
                        tool_use_id=event_item_id or f"{fallback_prefix}-{fallback_index}",
                        name=(
                            "web_search"
                            if "web_search_call" in event_type
                            else "mcp_list_tools"
                            if "mcp_list_tools" in event_type
                            else "mcp_call"
                        ),
                        index=fallback_index,
                        item_id=event_item_id,
                        item_type=(
                            "web_search_call"
                            if "web_search_call" in event_type
                            else "mcp_list_tools"
                            if "mcp_list_tools" in event_type
                            else "mcp_call"
                        ),
                        kind="web_search" if "web_search_call" in event_type else "tool",
                    )

                index = tool_info.index if tool_info.index is not None else -1
                tool_use_id = tool_info.tool_use_id
                payload = {
                    "tool_name": tool_info.tool_name or "web_search",
                    "tool_use_id": tool_use_id,
                    "index": index,
                    "status": event_type.rsplit(".", 1)[-1],
                }
                status = payload["status"]
                if tool_info.item_type == "web_search_call":
                    payload["tool_display_name"] = _WEB_SEARCH_PROGRESS_LABEL
                    payload["chunk"] = _web_search_status_chunk(str(status))
                elif tool_info.item_type == "mcp_list_tools":
                    payload["tool_display_name"] = _MCP_LIST_TOOLS_PROGRESS_LABEL
                    payload["chunk"] = _mcp_status_chunk("mcp_list_tools", str(status))
                elif tool_info.item_type == "mcp_call":
                    payload["tool_display_name"] = _mcp_call_display_title(
                        tool_info.tool_name,
                        status=str(status),
                    )
                    if status == "failed":
                        payload["chunk"] = _mcp_status_chunk("mcp_call", str(status))
                self._notify_tool_stream_listeners("status", payload)

                if event_type in _TOOL_START_EVENT_TYPES and not tool_info.start_notified:
                    self._notify_tool_stream_listeners("start", payload)
                    tool_info.start_notified = True
                    if index >= 0:
                        notified_tool_indices.add(index)

                if event_type in _TOOL_STOP_EVENT_TYPES:
                    if tool_info.item_type == "mcp_call":
                        tool_info.awaiting_output_item_done = True
                        tool_state.close(
                            index=index,
                            tool_use_id=tool_use_id,
                            item_id=event_item_id,
                        )
                        continue
                    self._notify_tool_stream_listeners("stop", payload)
                    self.logger.info(
                        "Model finished streaming tool call",
                        data={
                            "progress_action": ProgressAction.CALLING_TOOL,
                            "agent_name": self.name,
                            "model": model,
                            "tool_name": payload.get("tool_name"),
                            "tool_use_id": payload.get("tool_use_id"),
                            "tool_event": "stop",
                        },
                    )
                    tool_info.stop_notified = True
                    tool_state.close(index=index, tool_use_id=tool_use_id, item_id=event_item_id)
                continue

            if event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if not _item_is_responses_tool(item):
                    continue
                index = getattr(event, "output_index", None)
                item_id = getattr(event, "item_id", None) or getattr(item, "id", None)
                tool_use_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                tool_info = tool_state.resolve(
                    index=index,
                    tool_use_id=tool_use_id,
                    item_id=item_id,
                )
                was_already_completed = tool_info is not None and tool_state.is_completed(
                    index=index,
                    tool_use_id=tool_use_id,
                    item_id=item_id,
                )
                tool_info = tool_state.close(
                    index=index,
                    tool_use_id=tool_use_id,
                    item_id=item_id,
                ) or tool_info
                if tool_info is None and tool_state.is_completed(
                    index=index,
                    tool_use_id=tool_use_id,
                    item_id=item_id,
                ):
                    continue
                tool_name = _responses_tool_name(item)
                tool_use_id = (
                    tool_use_id
                    or (tool_info.tool_use_id if tool_info is not None else None)
                )
                if index is None:
                    index = tool_info.index if tool_info and tool_info.index is not None else -1
                item_type = tool_info.item_type if tool_info else getattr(item, "type", None)
                stop_payload = {
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "index": index,
                    "tool_display_name": None,
                }
                if item_type == "web_search_call":
                    stop_payload["tool_display_name"] = _WEB_SEARCH_PROGRESS_LABEL
                elif item_type == "mcp_call":
                    if tool_info is not None:
                        tool_info.awaiting_output_item_done = False
                    stop_payload["tool_display_name"] = _mcp_call_display_title(
                        tool_name,
                        status=getattr(item, "status", "completed"),
                    )
                    result_chunk = _mcp_call_output_chunk(getattr(item, "output", None))
                    if result_chunk is not None:
                        self._notify_tool_stream_listeners(
                            "replace",
                            {
                                **stop_payload,
                                "chunk": result_chunk,
                            },
                        )
                if not (was_already_completed and tool_info and tool_info.stop_notified):
                    self._notify_tool_stream_listeners(
                        "stop",
                        stop_payload,
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
                    if tool_info is not None:
                        tool_info.stop_notified = True
                if index >= 0:
                    notified_tool_indices.add(index)
                continue

        final_response = await fetch_and_finalize_stream_response(
            stream=stream,
            final_response=final_response,
            fetch_failure_message="Failed to fetch final Responses payload",
            use_exc_info_on_fetch_failure=True,
            incomplete_entries=tool_state.incomplete(),
            model=model,
            agent_name=self.name,
            chat_turn=self.chat_turn,
            logger=self.logger,
            notified_tool_indices=notified_tool_indices,
            emit_tool_fallback=self._emit_tool_notification_fallback,
        )
        self._emit_deferred_mcp_result_notifications(
            final_response=final_response,
            tool_state=tool_state,
            model=model,
        )
        return final_response, reasoning_segments

    def _emit_deferred_mcp_result_notifications(
        self,
        *,
        final_response: Any,
        tool_state: OpenAIToolStreamState,
        model: str,
    ) -> None:
        for index, item in enumerate(getattr(final_response, "output", []) or []):
            if getattr(item, "type", None) != "mcp_call":
                continue
            tool_use_id = getattr(item, "call_id", None) or getattr(item, "id", None)
            item_id = getattr(item, "id", None)
            tool_info = tool_state.resolve(
                index=index,
                tool_use_id=tool_use_id,
                item_id=item_id,
            )
            if tool_info is None or not tool_info.awaiting_output_item_done or tool_info.stop_notified:
                continue
            tool_name = _responses_tool_name(item)
            stop_payload = {
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "index": index,
                "tool_display_name": _mcp_call_display_title(
                    tool_name,
                    status=getattr(item, "status", "completed"),
                ),
            }
            result_chunk = _mcp_call_output_chunk(getattr(item, "output", None))
            if result_chunk is not None:
                self._notify_tool_stream_listeners(
                    "replace",
                    {
                        **stop_payload,
                        "chunk": result_chunk,
                    },
                )
            self._notify_tool_stream_listeners("stop", stop_payload)
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
            tool_info.awaiting_output_item_done = False
            tool_info.stop_notified = True

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
            if getattr(item, "type", None) not in {
                "function_call",
                "custom_tool_call",
                "web_search_call",
                "mcp_list_tools",
                "mcp_call",
            }:
                continue

            if getattr(item, "type", None) == "web_search_call":
                tool_name = "web_search"
                tool_use_id = getattr(item, "id", None) or f"tool-{index}"
            elif getattr(item, "type", None) == "mcp_list_tools":
                tool_name = "mcp_list_tools"
                tool_use_id = getattr(item, "id", None) or f"tool-{index}"
            elif getattr(item, "type", None) == "mcp_call":
                tool_name = getattr(item, "name", None) or "mcp_call"
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
