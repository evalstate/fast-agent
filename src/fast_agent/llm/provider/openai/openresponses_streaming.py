from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openai.types.responses import ResponseReasoningSummaryTextDeltaEvent, ResponseTextDeltaEvent

from fast_agent.event_progress import ProgressAction
from fast_agent.llm.provider.openai._stream_capture import (
    save_stream_chunk as _save_stream_chunk,
)
from fast_agent.llm.provider.openai.responses_events import (
    is_responses_reasoning_delta_event,
    is_responses_terminal_event,
    is_responses_text_delta_event,
)
from fast_agent.llm.provider.openai.streaming_utils import fetch_and_finalize_stream_response
from fast_agent.llm.provider.openai.tool_event_helpers import (
    ToolStreamLifecycleEvent,
    fallback_tool_spec,
    item_is_responses_tool,
    responses_event_item_id,
    responses_item_tool_use_id,
    responses_item_type,
    responses_lifecycle_event_info,
    responses_tool_name,
    tool_event_payload,
    tool_family_for_item_type,
    tool_stream_log_record,
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



_DONE_RESULT_TOOL_TYPES = {"mcp_call"}


STREAM_CAPTURE_DISABLED_MESSAGE = "Stream capture disabled"


@dataclass(slots=True)
class _OpenResponsesStreamState:
    estimated_tokens: int = 0
    reasoning_chars: int = 0
    reasoning_segments: ReasoningTextAccumulator = field(
        default_factory=lambda: ReasoningTextAccumulator(
            normalizer=normalize_reasoning_delta
        )
    )
    tool_state: OpenAIToolStreamState = field(default_factory=OpenAIToolStreamState)
    notified_tool_indices: set[int] = field(default_factory=set)
    notified_tool_use_ids: set[str] = field(default_factory=set)
    final_response: Any | None = None
    anonymous_tool_counter: int = 0


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
        return item_is_responses_tool(item)

    def _tool_name_from_item(self, item: Any) -> str:
        return responses_tool_name(item)

    def _tool_use_id_from_item(self, item: Any) -> str | None:
        return responses_item_tool_use_id(item)

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
        event_type: ToolStreamLifecycleEvent,
        fallback: bool = False,
    ) -> None:
        message, data = tool_stream_log_record(
            agent_name=self.name,
            model=model,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            event_type=event_type,
            fallback=fallback,
        )
        self.logger.info(message, data=data)

    @staticmethod
    def _mark_openresponses_tool_notified(
        state: _OpenResponsesStreamState,
        *,
        tool_use_id: str | None,
        index: int | None,
    ) -> None:
        if tool_use_id:
            state.notified_tool_use_ids.add(tool_use_id)
        if index is not None:
            state.notified_tool_indices.add(index)

    def _tool_use_id_for_openresponses_event(
        self,
        *,
        event: Any,
        item: Any,
        index: int | None,
        state: _OpenResponsesStreamState,
    ) -> str:
        tool_use_id = self._tool_use_id_from_item(item)
        if isinstance(tool_use_id, str) and tool_use_id:
            return tool_use_id

        item_id = responses_event_item_id(event, item)
        if item_id is not None:
            return item_id

        if index is not None:
            return f"tool-{index}"

        sequence_number = getattr(event, "sequence_number", None)
        if isinstance(sequence_number, int):
            return f"tool-seq-{sequence_number}"

        state.anonymous_tool_counter += 1
        return f"tool-unknown-{state.anonymous_tool_counter}"

    async def _emit_openresponses_reasoning_delta(
        self,
        delta_text: str,
        *,
        progress_model: str,
        state: _OpenResponsesStreamState,
    ) -> None:
        normalized_delta = state.reasoning_segments.append(delta_text)
        if not normalized_delta:
            return
        self._notify_stream_listeners(
            StreamChunk(text=normalized_delta, is_reasoning=True)
        )
        state.reasoning_chars += len(normalized_delta)
        await self._emit_streaming_progress(
            model=progress_model,
            new_total=state.reasoning_chars,
            type=ProgressAction.THINKING,
        )

    async def _handle_openresponses_content_part_added(
        self,
        event: Any,
        model: str,
        state: _OpenResponsesStreamState,
    ) -> bool:
        if getattr(event, "type", None) != "response.content_part.added":
            return False
        part = getattr(event, "part", None)
        part_type = getattr(part, "type", None)
        part_text = getattr(part, "text", None)
        if part_type in {"reasoning", "reasoning_text"} and part_text:
            await self._emit_openresponses_reasoning_delta(
                part_text,
                progress_model=f"{model} (reasoning)",
                state=state,
            )
        return True

    async def _handle_openresponses_reasoning_event(
        self,
        event: Any,
        event_type: Any,
        delta: Any,
        model: str,
        state: _OpenResponsesStreamState,
    ) -> bool:
        summary_event = isinstance(
            event,
            ResponseReasoningSummaryTextDeltaEvent,
        ) or is_responses_reasoning_delta_event(event_type)
        reasoning_event = isinstance(event, ResponseReasoningDeltaStreamingEvent) or event_type in {
            "response.reasoning.delta",
            "response.reasoning_text.delta",
        }
        if not summary_event and not reasoning_event:
            return False
        if delta:
            label = "summary" if summary_event else "reasoning"
            await self._emit_openresponses_reasoning_delta(
                delta,
                progress_model=f"{model} ({label})",
                state=state,
            )
        return True

    def _handle_openresponses_text_delta(
        self,
        event: Any,
        event_type: Any,
        delta: Any,
        model: str,
        state: _OpenResponsesStreamState,
    ) -> bool:
        if not (
            isinstance(event, ResponseTextDeltaEvent)
            or is_responses_text_delta_event(event_type)
        ):
            return False
        if delta:
            state.estimated_tokens = self._emit_stream_text_delta(
                text=delta,
                model=model,
                estimated_tokens=state.estimated_tokens,
            )
        return True

    @staticmethod
    def _handle_openresponses_terminal_event(
        event: Any,
        event_type: Any,
        state: _OpenResponsesStreamState,
    ) -> bool:
        if not is_responses_terminal_event(event_type):
            return False
        state.final_response = getattr(event, "response", None) or state.final_response
        return True

    def _handle_openresponses_output_item_added(
        self,
        event: Any,
        event_type: Any,
        model: str,
        state: _OpenResponsesStreamState,
    ) -> bool:
        if event_type != "response.output_item.added":
            return False
        item = getattr(event, "item", None)
        if not self._is_tool_item(item):
            return True

        index = getattr(event, "output_index", None)
        item_id = responses_event_item_id(event, item)
        tool_info = state.tool_state.register(
            tool_use_id=self._tool_use_id_for_openresponses_event(
                event=event, item=item, index=index, state=state
            ),
            name=self._tool_name_from_item(item),
            index=index,
            item_id=item_id,
            item_type=responses_item_type(item),
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
            self._mark_openresponses_tool_notified(
                state,
                tool_use_id=tool_info.tool_use_id,
                index=index,
            )
        return True

    def _handle_openresponses_tool_delta(
        self,
        event: Any,
        event_type: Any,
        delta: Any,
        state: _OpenResponsesStreamState,
    ) -> bool:
        if not (
            event_type
            and event_type.endswith(".delta")
            and (
                "function_call_arguments" in event_type
                or "custom_tool_call_input" in event_type
                or "_call" in event_type
            )
        ):
            return False
        if not delta:
            return True
        tool_info = state.tool_state.resolve_open(
            index=getattr(event, "output_index", None),
            item_id=responses_event_item_id(event),
        )
        if tool_info is not None:
            payload = self._tool_payload(tool_info)
            payload["chunk"] = delta
            self._notify_tool_stream_listeners("delta", payload)
        return True

    def _handle_openresponses_output_item_done(
        self,
        event: Any,
        event_type: Any,
        model: str,
        state: _OpenResponsesStreamState,
    ) -> bool:
        if event_type != "response.output_item.done":
            return False
        item = getattr(event, "item", None)
        if not self._is_tool_item(item):
            return True

        index = getattr(event, "output_index", None)
        item_id = responses_event_item_id(event, item)
        tool_use_id = self._tool_use_id_from_item(item)
        tool_info = state.tool_state.close(
            index=index, tool_use_id=tool_use_id, item_id=item_id
        )
        if tool_info is None:
            return True

        resolved_index = index if index is not None else tool_info.index
        if resolved_index is None:
            resolved_index = -1
        resolved_tool_use_id = tool_use_id or tool_info.tool_use_id
        payload = tool_event_payload(
            tool_name=self._tool_name_from_item(item),
            tool_use_id=resolved_tool_use_id,
            index=resolved_index,
            family=tool_family_for_item_type(responses_item_type(item)),
            phase="result",
        )
        if not tool_info.stop_notified:
            self._notify_tool_stream_listeners("stop", payload)
            self._log_tool_stream_event(
                model=model,
                tool_name=payload.get("tool_name"),
                tool_use_id=payload.get("tool_use_id"),
                event_type="stop",
            )
            self._mark_openresponses_tool_notified(
                state,
                tool_use_id=resolved_tool_use_id,
                index=resolved_index,
            )
            tool_info.stop_notified = True
        return True

    def _handle_openresponses_tool_status(
        self,
        event: Any,
        event_type: Any,
        model: str,
        state: _OpenResponsesStreamState,
    ) -> bool:
        if not event_type:
            return False
        event_info = responses_lifecycle_event_info(
            event_type,
            include_function_calls=True,
        )
        if event_info is None:
            return False

        status = event_info.status
        index = getattr(event, "output_index", None)
        item_id = responses_event_item_id(event)
        tool_info = state.tool_state.resolve_open(index=index, item_id=item_id)
        if tool_info is None:
            return True

        payload = self._tool_payload(tool_info, status=status)
        self._notify_tool_stream_listeners("status", payload)
        if event_info.lifecycle == "start" and not tool_info.start_notified:
            self._notify_tool_stream_listeners("start", payload)
            tool_info.start_notified = True
            self._log_tool_stream_event(
                model=model,
                tool_name=payload.get("tool_name"),
                tool_use_id=payload.get("tool_use_id"),
                event_type="start",
            )
            self._mark_openresponses_tool_notified(
                state,
                tool_use_id=tool_info.tool_use_id,
                index=tool_info.index,
            )
        if event_info.lifecycle == "stop" and tool_info.item_type in _DONE_RESULT_TOOL_TYPES:
            tool_info.awaiting_output_item_done = True
            return True
        if event_info.lifecycle == "stop":
            self._notify_tool_stream_listeners("stop", payload)
            self._log_tool_stream_event(
                model=model,
                tool_name=payload.get("tool_name"),
                tool_use_id=payload.get("tool_use_id"),
                event_type="stop",
            )
            self._mark_openresponses_tool_notified(
                state,
                tool_use_id=tool_info.tool_use_id,
                index=tool_info.index,
            )
            tool_info.stop_notified = True
            state.tool_state.close(
                index=tool_info.index,
                tool_use_id=tool_info.tool_use_id,
                item_id=item_id,
            )
        return True

    def _close_deferred_openresponses_tools(
        self,
        state: _OpenResponsesStreamState,
        *,
        model: str,
    ) -> None:
        for tool_info in state.tool_state.incomplete():
            if not tool_info.awaiting_output_item_done:
                continue
            if not tool_info.stop_notified:
                payload = self._tool_payload(tool_info, phase="result")
                self._notify_tool_stream_listeners("stop", payload)
                self._log_tool_stream_event(
                    model=model,
                    tool_name=payload.get("tool_name"),
                    tool_use_id=payload.get("tool_use_id"),
                    event_type="stop",
                )
                self._mark_openresponses_tool_notified(
                    state,
                    tool_use_id=tool_info.tool_use_id,
                    index=tool_info.index,
                )
                tool_info.stop_notified = True
            state.tool_state.close(
                index=tool_info.index,
                tool_use_id=tool_info.tool_use_id,
                item_id=tool_info.item_id,
            )

    async def _handle_openresponses_stream_event(
        self,
        event: Any,
        model: str,
        state: _OpenResponsesStreamState,
    ) -> None:
        event_type = getattr(event, "type", None)
        delta = getattr(event, "delta", None)
        handled = await self._handle_openresponses_content_part_added(event, model, state)
        if not handled:
            handled = await self._handle_openresponses_reasoning_event(
                event, event_type, delta, model, state
            )
        if not handled:
            handled = self._handle_openresponses_text_delta(
                event, event_type, delta, model, state
            )
        if not handled:
            handled = self._handle_openresponses_terminal_event(event, event_type, state)
        if not handled:
            handled = self._handle_openresponses_output_item_added(
                event, event_type, model, state
            )
        if not handled:
            handled = self._handle_openresponses_tool_delta(
                event, event_type, delta, state
            )
        if not handled:
            handled = self._handle_openresponses_output_item_done(
                event, event_type, model, state
            )
        if not handled:
            self._handle_openresponses_tool_status(event, event_type, model, state)

    async def _process_stream(
        self, stream: Any, model: str, capture_filename: Any
    ) -> tuple[Any, list[str]]:
        state = _OpenResponsesStreamState()
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
                await self._handle_openresponses_stream_event(event, model, state)

        self._close_deferred_openresponses_tools(state, model=model)

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
                notified_tool_use_ids=state.notified_tool_use_ids,
            )

        final_response = await fetch_and_finalize_stream_response(
            stream=stream,
            final_response=state.final_response,
            fetch_failure_message="Failed to fetch final Open Responses payload",
            use_exc_info_on_fetch_failure=False,
            incomplete_entries=state.tool_state.incomplete(),
            model=model,
            agent_name=self.name,
            chat_turn=self.chat_turn,
            logger=self.logger,
            notified_tool_indices=state.notified_tool_indices,
            emit_tool_fallback=emit_tool_fallback,
        )
        return final_response, state.reasoning_segments.parts()

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
