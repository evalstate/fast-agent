from __future__ import annotations

import json
import re
import warnings
from typing import TYPE_CHECKING, Any

from openai.types.responses import ResponseReasoningSummaryTextDeltaEvent, ResponseTextDeltaEvent

from fast_agent.event_progress import ProgressAction
from fast_agent.llm.provider.openai.streaming_utils import fetch_and_finalize_stream_response
from fast_agent.llm.provider.openai.tool_event_helpers import (
    fallback_tool_spec,
    tool_event_payload,
    tool_family_for_item_type,
)
from fast_agent.llm.provider.openai.tool_notifications import OpenAIToolNotificationMixin
from fast_agent.llm.provider.openai.tool_stream_state import (
    OpenAIToolStreamEntry,
    OpenAIToolStreamState,
)
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.utils.reasoning_chunk_join import (
    ReasoningTextAccumulator,
    normalize_reasoning_delta,
)

if TYPE_CHECKING:
    from openai.types.responses import (
        ResponseReasoningDeltaStreamingEvent,  # ty: ignore[unresolved-import]
    )
else:
    try:  # OpenAI SDK versions may not include reasoning delta events yet.
        from openai.types.responses import ResponseReasoningDeltaStreamingEvent
    except Exception:  # pragma: no cover - fallback for older SDKs
        class ResponseReasoningDeltaStreamingEvent:
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

        def _emit_stream_text_delta(
            self,
            *,
            text: str,
            model: str,
            estimated_tokens: int,
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

    def _tool_payload(
        self,
        info: OpenAIToolStreamEntry,
        *,
        status: str | None = None,
        phase: str = "call",
    ) -> dict[str, Any]:
        payload = tool_event_payload(
            tool_name=info.tool_name,
            tool_use_id=info.tool_use_id,
            index=info.index if info.index is not None else -1,
            family=tool_family_for_item_type(info.item_type),
            phase="result" if phase == "result" else "call",
            status=status,
        )
        if info.index is None:
            payload["index"] = None
        if info.item_id:
            payload["item_id"] = info.item_id
        if info.item_type:
            payload["tool_type"] = info.item_type
        return payload

    def _log_tool_stream_event(
        self,
        *,
        model: str,
        tool_name: str | None,
        tool_use_id: str | None,
        event_type: str,
        fallback: bool = False,
    ) -> None:
        message = (
            "Model emitted fallback tool notification"
            if fallback
            else "Model started streaming tool call"
            if event_type == "start"
            else "Model finished streaming tool call"
        )
        data: dict[str, Any] = {
            "progress_action": ProgressAction.CALLING_TOOL,
            "agent_name": self.name,
            "model": model,
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "tool_event": event_type,
        }
        if fallback:
            data["fallback"] = True
        self.logger.info(message, data=data)

    async def _process_stream(
        self, stream: Any, model: str, capture_filename: Any
    ) -> tuple[Any, list[str]]:
        estimated_tokens = 0
        reasoning_chars = 0
        reasoning_segments = ReasoningTextAccumulator(normalizer=normalize_reasoning_delta)
        tool_state = OpenAIToolStreamState()
        notified_tool_indices: set[int] = set()
        notified_tool_use_ids: set[str] = set()
        final_response: Any | None = None
        anonymous_tool_counter = 0

        def mark_tool_notified(
            *,
            tool_use_id: str | None,
            index: int | None,
        ) -> None:
            if tool_use_id:
                notified_tool_use_ids.add(tool_use_id)
            if index is not None:
                notified_tool_indices.add(index)

        def tool_use_id_for_event(*, event: Any, item: Any, index: int | None) -> str:
            nonlocal anonymous_tool_counter

            tool_use_id = self._tool_use_id_from_item(item)
            if isinstance(tool_use_id, str) and tool_use_id:
                return tool_use_id

            item_id = getattr(event, "item_id", None) or getattr(item, "id", None)
            if isinstance(item_id, str) and item_id:
                return item_id

            if index is not None:
                return f"tool-{index}"

            sequence_number = getattr(event, "sequence_number", None)
            if isinstance(sequence_number, int):
                return f"tool-seq-{sequence_number}"

            anonymous_tool_counter += 1
            return f"tool-unknown-{anonymous_tool_counter}"

        async def emit_reasoning_delta(delta_text: str, *, progress_model: str) -> None:
            nonlocal reasoning_chars

            normalized_delta = reasoning_segments.append(delta_text)
            if not normalized_delta:
                return
            self._notify_stream_listeners(StreamChunk(text=normalized_delta, is_reasoning=True))
            reasoning_chars += len(normalized_delta)
            await self._emit_streaming_progress(
                model=progress_model,
                new_total=reasoning_chars,
                type=ProgressAction.THINKING,
            )

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
                        await emit_reasoning_delta(
                            part_text,
                            progress_model=f"{model} (reasoning)",
                        )
                        continue

                if isinstance(event, ResponseReasoningSummaryTextDeltaEvent) or event_type in {
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_summary.delta",
                }:
                    if delta:
                        await emit_reasoning_delta(
                            delta,
                            progress_model=f"{model} (summary)",
                        )
                    continue

                if isinstance(event, ResponseReasoningDeltaStreamingEvent) or event_type in {
                    "response.reasoning.delta",
                    "response.reasoning_text.delta",
                }:
                    if delta:
                        await emit_reasoning_delta(
                            delta,
                            progress_model=f"{model} (reasoning)",
                        )
                    continue

                if isinstance(event, ResponseTextDeltaEvent) or event_type in {
                    "response.output_text.delta",
                    "response.text.delta",
                }:
                    if delta:
                        estimated_tokens = self._emit_stream_text_delta(
                            text=delta,
                            model=model,
                            estimated_tokens=estimated_tokens,
                        )
                    continue

                if event_type in {"response.completed", "response.incomplete"}:
                    final_response = getattr(event, "response", None) or final_response
                    continue

                if event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if not self._is_tool_item(item):
                        continue
                    index = getattr(event, "output_index", None)
                    item_id = getattr(event, "item_id", None) or getattr(item, "id", None)
                    tool_info = tool_state.register(
                        tool_use_id=tool_use_id_for_event(event=event, item=item, index=index),
                        name=self._tool_name_from_item(item),
                        index=index,
                        item_id=item_id,
                        item_type=getattr(item, "type", None),
                    )
                    if tool_info.tool_name and tool_info.tool_use_id and not tool_info.start_notified:
                        payload = self._tool_payload(tool_info)
                        self._notify_tool_stream_listeners("start", payload)
                        self._log_tool_stream_event(
                            model=model,
                            tool_name=payload["tool_name"],
                            tool_use_id=payload["tool_use_id"],
                            event_type="start",
                        )
                        tool_info.start_notified = True
                    if tool_info.start_notified:
                        mark_tool_notified(
                            tool_use_id=tool_info.tool_use_id,
                            index=index,
                        )
                    continue

                if event_type and event_type.endswith(".delta") and (
                    "function_call_arguments" in event_type
                    or "custom_tool_call_input" in event_type
                    or "_call" in event_type
                ):
                    index = getattr(event, "output_index", None)
                    item_id = getattr(event, "item_id", None)
                    tool_info = tool_state.resolve_open(
                        index=index,
                        item_id=item_id,
                    )
                    if delta:
                        if tool_info is None:
                            continue
                        payload = self._tool_payload(tool_info)
                        payload["chunk"] = delta
                        self._notify_tool_stream_listeners("delta", payload)
                    continue

                if event_type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if not self._is_tool_item(item):
                        continue
                    index = getattr(event, "output_index", None)
                    item_id = getattr(event, "item_id", None) or getattr(item, "id", None)
                    tool_use_id = self._tool_use_id_from_item(item)
                    tool_info = tool_state.close(index=index, tool_use_id=tool_use_id, item_id=item_id)
                    if tool_info is None and tool_state.is_completed(
                        index=index,
                        tool_use_id=tool_use_id,
                        item_id=item_id,
                    ):
                        continue
                    if tool_info is None:
                        continue
                    if index is None:
                        index = tool_info.index if tool_info and tool_info.index is not None else -1
                    tool_name = self._tool_name_from_item(item)
                    tool_use_id = (
                        tool_use_id or (tool_info.tool_use_id if tool_info is not None else None)
                    )
                    payload = {
                        **tool_event_payload(
                            tool_name=tool_name,
                            tool_use_id=tool_use_id,
                            index=index,
                            family=tool_family_for_item_type(getattr(item, "type", None)),
                            phase="result",
                        )
                    }
                    if not (tool_info.stop_notified if tool_info is not None else False):
                        self._notify_tool_stream_listeners("stop", payload)
                        self._log_tool_stream_event(
                            model=model,
                            tool_name=tool_name,
                            tool_use_id=tool_use_id,
                            event_type="stop",
                        )
                        mark_tool_notified(
                            tool_use_id=tool_use_id,
                            index=index,
                        )
                        if tool_info is not None:
                            tool_info.stop_notified = True
                    continue

                if event_type:
                    match = _TOOL_STATUS_PATTERN.match(event_type)
                    if match and "_call" in match.group("tool"):
                        status = match.group("status")
                        index = getattr(event, "output_index", None)
                        item_id = getattr(event, "item_id", None)
                        tool_info = tool_state.resolve_open(
                            index=index,
                            item_id=item_id,
                        )
                        if tool_info is None:
                            if tool_state.is_completed(index=index, item_id=item_id):
                                continue
                            continue
                        payload = self._tool_payload(tool_info, status=status)
                        self._notify_tool_stream_listeners("status", payload)
                        if status in _TOOL_START_STATUSES and not tool_info.start_notified:
                            self._notify_tool_stream_listeners("start", payload)
                            tool_info.start_notified = True
                            self._log_tool_stream_event(
                                model=model,
                                tool_name=payload.get("tool_name"),
                                tool_use_id=payload.get("tool_use_id"),
                                event_type="start",
                            )
                            mark_tool_notified(
                                tool_use_id=tool_info.tool_use_id,
                                index=tool_info.index,
                            )
                        if status in _TOOL_STOP_STATUSES:
                            self._notify_tool_stream_listeners("stop", payload)
                            self._log_tool_stream_event(
                                model=model,
                                tool_name=payload.get("tool_name"),
                                tool_use_id=payload.get("tool_use_id"),
                                event_type="stop",
                            )
                            mark_tool_notified(
                                tool_use_id=tool_info.tool_use_id,
                                index=tool_info.index,
                            )
                            tool_info.stop_notified = True
                            tool_state.close(
                                index=tool_info.index,
                                tool_use_id=tool_info.tool_use_id,
                                item_id=item_id,
                            )
                        continue

        def emit_tool_fallback(
            output_items: list[Any],
            notified_indices: set[int],
            *,
            model: str,
        ) -> None:
            self._emit_tool_notification_fallback(
                output_items,
                notified_indices,
                model=model,
                notified_tool_use_ids=notified_tool_use_ids,
            )

        final_response = await fetch_and_finalize_stream_response(
            stream=stream,
            final_response=final_response,
            fetch_failure_message="Failed to fetch final Open Responses payload",
            use_exc_info_on_fetch_failure=False,
            incomplete_entries=tool_state.incomplete(),
            model=model,
            agent_name=self.name,
            chat_turn=self.chat_turn,
            logger=self.logger,
            notified_tool_indices=notified_tool_indices,
            emit_tool_fallback=emit_tool_fallback,
        )
        return final_response, reasoning_segments.parts()

    def _emit_tool_notification_fallback(
        self,
        output_items: list[Any],
        notified_indices: set[int],
        *,
        model: str,
        notified_tool_use_ids: set[str] | None = None,
    ) -> None:
        if not output_items:
            return

        deduped_tool_use_ids = notified_tool_use_ids or set()
        for index, item in enumerate(output_items):
            if not self._is_tool_item(item):
                continue

            tool_name, tool_use_id, family = fallback_tool_spec(item, index)
            if index in notified_indices or tool_use_id in deduped_tool_use_ids:
                continue

            payload = tool_event_payload(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                index=index,
                family=family,
                phase="call",
            )
            self._emit_fallback_tool_notification_event(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                index=index,
                model=model,
                payload=payload,
            )
