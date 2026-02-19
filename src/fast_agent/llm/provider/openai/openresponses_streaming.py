from __future__ import annotations

import json
import re
import warnings
from typing import TYPE_CHECKING, Any

from openai.types.responses import ResponseReasoningSummaryTextDeltaEvent, ResponseTextDeltaEvent

from fast_agent.event_progress import ProgressAction
from fast_agent.llm.provider.openai.streaming_utils import finalize_stream_response
from fast_agent.llm.provider.openai.tool_notifications import OpenAIToolNotificationMixin
from fast_agent.llm.stream_types import StreamChunk

if TYPE_CHECKING:
    from openai.types.responses import (
        ResponseReasoningDeltaStreamingEvent,  # ty: ignore[unresolved-import]
    )
else:
    try:  # OpenAI SDK versions may not include reasoning delta events yet.
        from openai.types.responses import ResponseReasoningDeltaStreamingEvent
    except Exception:  # pragma: no cover - fallback for older SDKs
        class ResponseReasoningDeltaStreamingEvent:  # type: ignore[no-redef]
            pass



_TOOL_START_STATUSES = {"in_progress", "queued", "started"}
_TOOL_STOP_STATUSES = {"completed", "failed", "cancelled", "incomplete"}
_TOOL_STATUS_PATTERN = re.compile(r"^response\.(?P<tool>[^.]+)\.(?P<status>[^.]+)$")


STREAM_CAPTURE_DISABLED_MESSAGE = "Stream capture disabled"


def _safe_event_payload(event: Any) -> Any:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            return event.model_dump(warnings="none")
    except Exception:
        try:
            return event.model_dump_json()
        except Exception:
            return str(event)


def _save_stream_chunk(filename_base: Any, chunk: Any) -> None:
    if not filename_base:
        return
    try:
        chunk_file = filename_base.with_name(f"{filename_base.name}.jsonl")
        payload = _safe_event_payload(chunk)
        with chunk_file.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
    except Exception:
        return


class OpenResponsesStreamingMixin(OpenAIToolNotificationMixin):
    if TYPE_CHECKING:
        from fast_agent.core.logging.logger import Logger

        logger: Logger
        name: str | None

        def _notify_stream_listeners(self, chunk: StreamChunk) -> None: ...

        def _notify_tool_stream_listeners(
            self, event_type: str, payload: dict[str, Any] | None = None
        ) -> None: ...

        def _update_streaming_progress(
            self, chunk: str, model: str, current_total: int
        ) -> int: ...

        def chat_turn(self) -> int: ...

        async def _emit_streaming_progress(
            self,
            model: str,
            new_total: int,
            type: ProgressAction = ProgressAction.STREAMING,
        ) -> None: ...

        def _emit_tool_notification_fallback(
            self, output_items: list[Any], notified_indices: set[int], *, model: str
        ) -> None: ...

    def _is_tool_item(self, item: Any) -> bool:
        item_type = getattr(item, "type", None)
        if not item_type:
            return False
        return item_type in {"function_call", "tool_call"} or item_type.endswith("_call")

    def _tool_name_from_item(self, item: Any) -> str:
        return (
            getattr(item, "name", None)
            or getattr(item, "tool_name", None)
            or getattr(item, "type", None)
            or "tool"
        )

    def _tool_use_id_from_item(self, item: Any) -> str | None:
        return getattr(item, "call_id", None) or getattr(item, "id", None)

    def _tool_name_from_event_type(self, event_type: str | None) -> str | None:
        if not event_type:
            return None
        if not event_type.startswith("response."):
            return None
        suffix = event_type[len("response.") :]
        tool_slug = suffix.split(".", 1)[0]
        return tool_slug or None

    def _build_tool_info(
        self,
        *,
        item: Any | None = None,
        index: int | None = None,
        event_type: str | None = None,
        item_id: str | None = None,
    ) -> dict[str, Any]:
        tool_type = getattr(item, "type", None) if item is not None else None
        tool_name = self._tool_name_from_item(item) if item is not None else None
        tool_use_id = self._tool_use_id_from_item(item) if item is not None else None
        if not tool_name:
            tool_name = self._tool_name_from_event_type(event_type) or "tool"
        if not tool_use_id:
            tool_use_id = item_id
        if not tool_use_id and index is not None:
            tool_use_id = f"tool-{index}"
        return {
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "item_id": item_id or getattr(item, "id", None),
            "index": index,
            "type": tool_type,
            "notified": False,
        }

    def _tool_payload(self, info: dict[str, Any], *, status: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool_name": info.get("tool_name"),
            "tool_use_id": info.get("tool_use_id"),
            "index": info.get("index"),
        }
        if info.get("item_id"):
            payload["item_id"] = info.get("item_id")
        if info.get("type"):
            payload["tool_type"] = info.get("type")
        if status:
            payload["status"] = status
        return payload

    def _lookup_tool_info(
        self,
        *,
        tool_streams: dict[int, dict[str, Any]],
        tool_streams_by_id: dict[str, dict[str, Any]],
        index: int | None,
        item_id: str | None,
        event_type: str | None,
    ) -> dict[str, Any]:
        if index is not None and index in tool_streams:
            return tool_streams[index]
        if item_id and item_id in tool_streams_by_id:
            return tool_streams_by_id[item_id]
        return self._build_tool_info(index=index, event_type=event_type, item_id=item_id)

    async def _process_stream(
        self, stream: Any, model: str, capture_filename: Any
    ) -> tuple[Any, list[str]]:
        estimated_tokens = 0
        reasoning_chars = 0
        reasoning_segments: list[str] = []
        tool_streams: dict[int, dict[str, Any]] = {}
        tool_streams_by_id: dict[str, dict[str, Any]] = {}
        notified_tool_indices: set[int] = set()
        final_response: Any | None = None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Pydantic serializer warnings",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*PydanticSerializationUnexpectedValue.*",
                category=UserWarning,
            )
            async for event in stream:
                _save_stream_chunk(capture_filename, event)
                event_type = getattr(event, "type", None)
                delta = getattr(event, "delta", None)

                if event_type == "response.content_part.added":
                    part = getattr(event, "part", None)
                    part_type = getattr(part, "type", None)
                    part_text = getattr(part, "text", None)
                    if part_type in {"reasoning", "reasoning_text"} and part_text:
                        reasoning_segments.append(part_text)
                        self._notify_stream_listeners(
                            StreamChunk(text=part_text, is_reasoning=True)
                        )
                        reasoning_chars += len(part_text)
                        await self._emit_streaming_progress(
                            model=f"{model} (reasoning)",
                            new_total=reasoning_chars,
                            type=ProgressAction.THINKING,
                        )
                        continue

                if isinstance(event, ResponseReasoningSummaryTextDeltaEvent) or event_type in {
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_summary.delta",
                }:
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

                if isinstance(event, ResponseReasoningDeltaStreamingEvent) or event_type in {
                    "response.reasoning.delta",
                    "response.reasoning_text.delta",
                }:
                    if delta:
                        reasoning_segments.append(delta)
                        self._notify_stream_listeners(
                            StreamChunk(text=delta, is_reasoning=True)
                        )
                        reasoning_chars += len(delta)
                        await self._emit_streaming_progress(
                            model=f"{model} (reasoning)",
                            new_total=reasoning_chars,
                            type=ProgressAction.THINKING,
                        )
                    continue

                if isinstance(event, ResponseTextDeltaEvent) or event_type in {
                    "response.output_text.delta",
                    "response.text.delta",
                }:
                    if delta:
                        self._notify_stream_listeners(
                            StreamChunk(text=delta, is_reasoning=False)
                        )
                        estimated_tokens = self._update_streaming_progress(
                            delta, model, estimated_tokens
                        )
                        self._notify_tool_stream_listeners("text", {"chunk": delta})
                    continue

                if event_type in {"response.completed", "response.incomplete"}:
                    final_response = getattr(event, "response", None) or final_response
                    continue

                if event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if not self._is_tool_item(item):
                        continue
                    index = getattr(event, "output_index", None)
                    if index is None:
                        continue
                    tool_info = self._build_tool_info(
                        item=item, index=index, item_id=getattr(event, "item_id", None)
                    )
                    tool_streams[index] = tool_info
                    if tool_info.get("item_id"):
                        tool_streams_by_id[tool_info["item_id"]] = tool_info
                    if tool_info["tool_name"] and tool_info["tool_use_id"]:
                        payload = self._tool_payload(tool_info)
                        self._notify_tool_stream_listeners("start", payload)
                        self.logger.info(
                            "Model started streaming tool call",
                            data={
                                "progress_action": ProgressAction.CALLING_TOOL,
                                "agent_name": self.name,
                                "model": model,
                                "tool_name": payload["tool_name"],
                                "tool_use_id": payload["tool_use_id"],
                                "tool_event": "start",
                            },
                        )
                        tool_info["notified"] = True
                        notified_tool_indices.add(index)
                    continue

                if event_type and event_type.endswith(".delta") and (
                    "function_call_arguments" in event_type
                    or "custom_tool_call_input" in event_type
                    or "_call" in event_type
                ):
                    index = getattr(event, "output_index", None)
                    item_id = getattr(event, "item_id", None)
                    tool_info = self._lookup_tool_info(
                        tool_streams=tool_streams,
                        tool_streams_by_id=tool_streams_by_id,
                        index=index,
                        item_id=item_id,
                        event_type=event_type,
                    )
                    if delta:
                        payload = self._tool_payload(tool_info)
                        payload["chunk"] = delta
                        self._notify_tool_stream_listeners("delta", payload)
                    continue

                if event_type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if not self._is_tool_item(item):
                        continue
                    index = getattr(event, "output_index", None)
                    tool_info = tool_streams.pop(index, {}) if index is not None else {}
                    if tool_info.get("item_id") in tool_streams_by_id:
                        tool_streams_by_id.pop(tool_info["item_id"], None)
                    if index is None:
                        index = tool_info.get("index") or -1
                    tool_name = self._tool_name_from_item(item)
                    tool_use_id = (
                        self._tool_use_id_from_item(item) or tool_info.get("tool_use_id")
                    )
                    payload = {
                        "tool_name": tool_name,
                        "tool_use_id": tool_use_id,
                        "index": index,
                    }
                    if not tool_info.get("stopped_notified"):
                        self._notify_tool_stream_listeners("stop", payload)
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
                    continue

                if event_type:
                    match = _TOOL_STATUS_PATTERN.match(event_type)
                    if match and "_call" in match.group("tool"):
                        status = match.group("status")
                        index = getattr(event, "output_index", None)
                        item_id = getattr(event, "item_id", None)
                        tool_info = self._lookup_tool_info(
                            tool_streams=tool_streams,
                            tool_streams_by_id=tool_streams_by_id,
                            index=index,
                            item_id=item_id,
                            event_type=event_type,
                        )
                        payload = self._tool_payload(tool_info, status=status)
                        self._notify_tool_stream_listeners("status", payload)
                        if status in _TOOL_START_STATUSES and not tool_info.get("notified"):
                            self._notify_tool_stream_listeners("start", payload)
                            tool_info["notified"] = True
                            if index is not None:
                                notified_tool_indices.add(index)
                        if status in _TOOL_STOP_STATUSES:
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
                            tool_info["stopped_notified"] = True
                        continue

        if final_response is None:
            try:
                final_response = await stream.get_final_response()
            except Exception as exc:
                self.logger.warning(
                    "Failed to fetch final Open Responses payload",
                    data={"error": str(exc)},
                )
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
        if not output_items:
            return

        for index, item in enumerate(output_items):
            if index in notified_indices:
                continue
            if not self._is_tool_item(item):
                continue

            tool_name = self._tool_name_from_item(item)
            tool_use_id = self._tool_use_id_from_item(item) or f"tool-{index}"

            self._emit_fallback_tool_notification_event(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                index=index,
                model=model,
            )
