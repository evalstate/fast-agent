from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from openai.types.responses import (
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
)

from fast_agent.core.logging.json_serializer import snapshot_json_value
from fast_agent.core.logging.logger import get_logger
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
    ResponsesLifecycleEventInfo,
    ToolStreamLifecycleEvent,
    fallback_tool_spec,
    item_is_responses_tool,
    item_type_is_responses_function_tool_call,
    responses_event_item_id,
    responses_item_tool_use_id,
    responses_item_type,
    responses_lifecycle_event_info,
    responses_tool_name,
    responses_tool_use_id,
    tool_event_payload,
    tool_family_for_item_type,
    tool_stream_log_record,
)
from fast_agent.llm.provider.openai.tool_notifications import OpenAIToolNotificationMixin
from fast_agent.llm.provider.openai.tool_stream_state import OpenAIToolStreamState
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.tool_activity_presentation import (
    ToolActivityFamily,
    build_tool_activity_presentation,
    tool_activity_status_text,
)
from fast_agent.utils.reasoning_chunk_join import (
    ReasoningTextAccumulator,
    join_reasoning_summary_parts,
    normalize_reasoning_delta,
)

_logger = get_logger(__name__)

_ARGUMENT_DELTA_EVENT_TYPES = {
    "response.function_call_arguments.delta",
    "response.custom_tool_call_input.delta",
    "response.mcp_call_arguments.delta",
}
type _ResponsesToolKind = Literal["tool", "server_tool", "web_search"]
_DEFAULT_TOOL_KIND: _ResponsesToolKind = "tool"
_TOOL_KIND_BY_ACTIVITY_FAMILY: dict[ToolActivityFamily, _ResponsesToolKind] = {
    "web_search": "web_search",
    "remote_tool": "server_tool",
    "remote_tool_listing": "server_tool",
    "remote_tool_search": "server_tool",
}


def _preview_json_like(value: Any) -> str | None:
    normalized = snapshot_json_value(value)
    if normalized is None:
        return None
    if normalized in ({}, []):
        return None
    preview = normalized.strip() if isinstance(normalized, str) else json.dumps(normalized)
    if not preview:
        return None
    if len(preview) > 120:
        return f"{preview[:117]}..."
    return preview


def _mcp_call_output_chunk(output: Any) -> str | None:
    return _preview_json_like(output)


def _tool_search_arguments_chunk(arguments: Any) -> str | None:
    return _preview_json_like(arguments)


def _tool_progress_chunk(item: Any, *, family: ToolActivityFamily) -> str | None:
    item_type = responses_item_type(item)
    if item_type == "tool_search_call":
        return _tool_search_arguments_chunk(getattr(item, "arguments", None)) or (
            tool_activity_status_text(family=family, status="in_progress")
        )
    if item_type in {"web_search_call", "mcp_list_tools"}:
        return tool_activity_status_text(family=family, status="in_progress")
    if item_type == "mcp_call":
        arguments = getattr(item, "arguments", None)
        return arguments if isinstance(arguments, str) and arguments else None
    return None


class ResponsesStreamingMixin(OpenAIToolNotificationMixin):
    if TYPE_CHECKING:
        from pathlib import Path

        from fast_agent.core.logging.logger import Logger
        from fast_agent.llm.tool_tracking import ToolKind

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

    def _is_provider_managed_function_call(self, name: str) -> bool:
        del name
        return False

    def _tool_family_for_responses_item(
        self,
        *,
        item_type: str | None,
        tool_name: str,
    ) -> ToolActivityFamily:
        del tool_name
        return tool_family_for_item_type(item_type)

    def _tool_kind_for_responses_item(
        self,
        *,
        item_type: str | None,
        tool_name: str,
    ) -> "ToolKind":
        family = self._tool_family_for_responses_item(
            item_type=item_type,
            tool_name=tool_name,
        )
        return _TOOL_KIND_BY_ACTIVITY_FAMILY.get(family, _DEFAULT_TOOL_KIND)

    def _log_tool_stream_event(
        self,
        *,
        model: str,
        tool_name: str | None,
        tool_use_id: str | None,
        event_type: ToolStreamLifecycleEvent,
    ) -> None:
        message, data = tool_stream_log_record(
            agent_name=self.name,
            model=model,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            event_type=event_type,
        )
        self.logger.info(message, data=data)

    @staticmethod
    def _mark_responses_tool_notified(
        *,
        notified_tool_indices: set[int],
        notified_tool_use_ids: set[str] | None,
        index: int | None,
        tool_use_id: str | None,
    ) -> None:
        if index is not None and index >= 0:
            notified_tool_indices.add(index)
        if notified_tool_use_ids is not None and tool_use_id:
            notified_tool_use_ids.add(tool_use_id)

    def _handle_responses_output_item_added(
        self,
        *,
        event: Any,
        tool_state: OpenAIToolStreamState,
        notified_tool_indices: set[int],
        model: str,
        notified_tool_use_ids: set[str] | None = None,
    ) -> bool:
        item = getattr(event, "item", None)
        if not item_is_responses_tool(item):
            return False
        item_type = responses_item_type(item) or "tool"
        tool_name = responses_tool_name(item)

        index = getattr(event, "output_index", None)
        item_id = responses_event_item_id(event, item)

        tool_info = tool_state.register(
            tool_use_id=responses_tool_use_id(
                item,
                index,
                item_id,
            ),
            name=tool_name,
            index=index,
            item_id=item_id,
            item_type=item_type,
            kind=self._tool_kind_for_responses_item(
                item_type=item_type,
                tool_name=tool_name,
            ),
        )
        tool_info.argument_snapshot_present = (
            item_type == "mcp_call"
            and isinstance(getattr(item, "arguments", None), str)
            and bool(getattr(item, "arguments", None))
        )
        family = self._tool_family_for_responses_item(
            item_type=item_type,
            tool_name=tool_name,
        )
        display_chunk = _tool_progress_chunk(item, family=family)
        display_name = build_tool_activity_presentation(
            tool_name=tool_name,
            family=family,
            phase="call",
        ).display_name
        payload = tool_event_payload(
            tool_name=tool_info.tool_name,
            tool_use_id=tool_info.tool_use_id,
            index=index if index is not None else -1,
            family=family,
            phase="call",
            chunk=display_chunk,
        )
        if index is None:
            payload["index"] = None
        payload["tool_display_name"] = display_name
        self._notify_tool_stream_listeners("start", payload)
        self._log_tool_stream_event(
            model=model,
            tool_name=tool_info.tool_name,
            tool_use_id=tool_info.tool_use_id,
            event_type="start",
        )
        tool_info.start_notified = True
        self._mark_responses_tool_notified(
            notified_tool_indices=notified_tool_indices,
            notified_tool_use_ids=notified_tool_use_ids,
            index=index,
            tool_use_id=tool_info.tool_use_id,
        )
        return True

    def _handle_responses_argument_delta(
        self,
        *,
        event: Any,
        tool_state: OpenAIToolStreamState,
    ) -> bool:
        index = getattr(event, "output_index", None)
        item_id = responses_event_item_id(event)
        tool_info = tool_state.resolve_open(index=index, item_id=item_id)
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

        if tool_info is None:
            payload = {
                "tool_name": None,
                "tool_use_id": None,
                "index": index if index is not None else -1,
                "chunk": getattr(event, "delta", None),
            }
            if index is None:
                payload["index"] = None
        else:
            payload = tool_event_payload(
                tool_name=tool_info.tool_name,
                tool_use_id=tool_info.tool_use_id,
                index=index if index is not None else -1,
                family=self._tool_family_for_responses_item(
                    item_type=tool_info.item_type,
                    tool_name=tool_info.tool_name,
                ),
                phase="call",
                chunk=getattr(event, "delta", None),
            )
            if index is None:
                payload["index"] = None
        self._notify_tool_stream_listeners(event_name, payload)
        return True

    def _resolve_lifecycle_tool_info(
        self,
        *,
        event_info: ResponsesLifecycleEventInfo,
        event_index: int | None,
        event_item_id: str | None,
        tool_state: OpenAIToolStreamState,
    ) -> Any:
        tool_info = tool_state.resolve_open(index=event_index, item_id=event_item_id)
        if tool_info is not None:
            return tool_info
        if tool_state.is_completed(index=event_index, item_id=event_item_id):
            return None

        fallback_index = event_index if event_index is not None else -1
        return tool_state.register(
            tool_use_id=event_item_id or f"{event_info.tool_name}-{fallback_index}",
            name=event_info.tool_name,
            index=fallback_index,
            item_id=event_item_id,
            item_type=event_info.item_type,
            kind=self._tool_kind_for_responses_item(
                item_type=event_info.item_type,
                tool_name=event_info.tool_name,
            ),
        )

    def _handle_responses_tool_lifecycle_event(
        self,
        *,
        event: Any,
        event_info: ResponsesLifecycleEventInfo,
        tool_state: OpenAIToolStreamState,
        notified_tool_indices: set[int],
        model: str,
        notified_tool_use_ids: set[str] | None = None,
    ) -> bool:
        event_index = getattr(event, "output_index", None)
        event_item_id = responses_event_item_id(event)
        tool_info = self._resolve_lifecycle_tool_info(
            event_info=event_info,
            event_index=event_index,
            event_item_id=event_item_id,
            tool_state=tool_state,
        )
        if tool_info is None:
            return True

        index = tool_info.index if tool_info.index is not None else -1
        tool_use_id = tool_info.tool_use_id
        tool_name = tool_info.tool_name or event_info.tool_name
        status = event_info.status
        family = self._tool_family_for_responses_item(
            item_type=tool_info.item_type,
            tool_name=tool_name,
        )
        payload = tool_event_payload(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            index=index,
            family=family,
            phase="call",
            status=status,
            chunk=tool_activity_status_text(family=family, status=status) or None,
        )
        self._notify_tool_stream_listeners("status", payload)

        if event_info.lifecycle == "start" and not tool_info.start_notified:
            self._notify_tool_stream_listeners("start", payload)
            self._log_tool_stream_event(
                model=model,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                event_type="start",
            )
            tool_info.start_notified = True
            self._mark_responses_tool_notified(
                notified_tool_indices=notified_tool_indices,
                notified_tool_use_ids=notified_tool_use_ids,
                index=index,
                tool_use_id=tool_use_id,
            )

        if event_info.lifecycle != "stop":
            return True

        if tool_info.item_type == "mcp_call":
            tool_info.awaiting_output_item_done = True
            tool_state.close(
                index=index,
                tool_use_id=tool_use_id,
                item_id=event_item_id,
            )
            return True

        self._notify_tool_stream_listeners("stop", payload)
        self._log_tool_stream_event(
            model=model,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            event_type="stop",
        )
        tool_info.stop_notified = True
        tool_state.close(index=index, tool_use_id=tool_use_id, item_id=event_item_id)
        self._mark_responses_tool_notified(
            notified_tool_indices=notified_tool_indices,
            notified_tool_use_ids=notified_tool_use_ids,
            index=index,
            tool_use_id=tool_use_id,
        )
        return True

    def _handle_responses_output_item_done(
        self,
        *,
        event: Any,
        tool_state: OpenAIToolStreamState,
        notified_tool_indices: set[int],
        model: str,
        notified_tool_use_ids: set[str] | None = None,
    ) -> bool:
        item = getattr(event, "item", None)
        if not item_is_responses_tool(item):
            return False

        index = getattr(event, "output_index", None)
        item_id = responses_event_item_id(event, item)
        tool_use_id = responses_item_tool_use_id(item)
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
        tool_info = (
            tool_state.close(
                index=index,
                tool_use_id=tool_use_id,
                item_id=item_id,
            )
            or tool_info
        )
        if tool_info is None and tool_state.is_completed(
            index=index,
            tool_use_id=tool_use_id,
            item_id=item_id,
        ):
            return True
        if tool_info is None:
            return True

        tool_name = responses_tool_name(item)
        tool_use_id = tool_use_id or tool_info.tool_use_id
        if index is None:
            index = tool_info.index if tool_info.index is not None else -1
        item_type = tool_info.item_type if tool_info else responses_item_type(item)
        family = self._tool_family_for_responses_item(
            item_type=item_type if isinstance(item_type, str) else None,
            tool_name=tool_name,
        )
        stop_payload = tool_event_payload(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            index=index,
            family=family,
            phase="result" if family == "remote_tool" else "call",
        )
        if item_type == "tool_search_call":
            self._notify_tool_stream_listeners(
                "replace",
                {
                    **stop_payload,
                    "chunk": tool_activity_status_text(
                        family=family,
                        status=str(getattr(item, "status", None) or "completed"),
                    ),
                },
            )
        elif item_type == "mcp_call":
            tool_info.awaiting_output_item_done = False
            result_chunk = _mcp_call_output_chunk(getattr(item, "output", None))
            if result_chunk is not None:
                self._notify_tool_stream_listeners(
                    "replace",
                    {
                        **stop_payload,
                        "chunk": result_chunk,
                    },
                )

        if was_already_completed and tool_info.stop_notified:
            return True

        self._notify_tool_stream_listeners("stop", stop_payload)
        self._log_tool_stream_event(
            model=model,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            event_type="stop",
        )
        tool_info.stop_notified = True
        self._mark_responses_tool_notified(
            notified_tool_indices=notified_tool_indices,
            notified_tool_use_ids=notified_tool_use_ids,
            index=index,
            tool_use_id=tool_use_id,
        )
        return True

    async def _handle_responses_reasoning_delta(
        self,
        *,
        event: Any,
        event_type: str | None,
        reasoning_segments: ReasoningTextAccumulator,
        reasoning_summary_parts: dict[tuple[str, int], ReasoningTextAccumulator],
        reasoning_chars: int,
        model: str,
    ) -> tuple[bool, int]:
        if not isinstance(
            event, ResponseReasoningSummaryTextDeltaEvent
        ) and not is_responses_reasoning_delta_event(event_type):
            return False, reasoning_chars

        delta = getattr(event, "delta", None)
        if not delta:
            return True, reasoning_chars

        item_id = getattr(event, "item_id", None)
        summary_index = getattr(event, "summary_index", None)
        if isinstance(item_id, str) and isinstance(summary_index, int):
            part_key = (item_id, summary_index)
            part = reasoning_summary_parts.get(part_key)
            if part is None:
                part = ReasoningTextAccumulator()
                reasoning_summary_parts[part_key] = part
            part.append(delta)

        normalized_delta = reasoning_segments.append(delta)
        if not normalized_delta:
            return True, reasoning_chars

        self._notify_stream_listeners(StreamChunk(text=normalized_delta, is_reasoning=True))
        reasoning_chars += len(normalized_delta)
        await self._emit_streaming_progress(
            model=f"{model} (summary)",
            new_total=reasoning_chars,
            type=ProgressAction.THINKING,
        )
        return True, reasoning_chars

    def _handle_responses_tool_stream_event(
        self,
        *,
        event: Any,
        event_type: str | None,
        tool_state: OpenAIToolStreamState,
        notified_tool_indices: set[int],
        notified_tool_use_ids: set[str],
        model: str,
    ) -> bool:
        if event_type == "response.output_item.added":
            self._handle_responses_output_item_added(
                event=event,
                tool_state=tool_state,
                notified_tool_indices=notified_tool_indices,
                notified_tool_use_ids=notified_tool_use_ids,
                model=model,
            )
            return True

        if event_type in _ARGUMENT_DELTA_EVENT_TYPES:
            self._handle_responses_argument_delta(
                event=event,
                tool_state=tool_state,
            )
            return True

        lifecycle_event_info = responses_lifecycle_event_info(event_type)
        if lifecycle_event_info is not None:
            self._handle_responses_tool_lifecycle_event(
                event=event,
                event_info=lifecycle_event_info,
                tool_state=tool_state,
                notified_tool_indices=notified_tool_indices,
                notified_tool_use_ids=notified_tool_use_ids,
                model=model,
            )
            return True

        if event_type == "response.output_item.done":
            self._handle_responses_output_item_done(
                event=event,
                tool_state=tool_state,
                notified_tool_indices=notified_tool_indices,
                notified_tool_use_ids=notified_tool_use_ids,
                model=model,
            )
            return True

        return False

    async def _process_stream(
        self, stream: Any, model: str, capture_filename: Path | None
    ) -> tuple[Any, list[str]]:
        estimated_tokens = 0
        reasoning_chars = 0
        reasoning_segments = ReasoningTextAccumulator(normalizer=normalize_reasoning_delta)
        reasoning_summary_parts: dict[tuple[str, int], ReasoningTextAccumulator] = {}
        tool_state = OpenAIToolStreamState()
        notified_tool_indices: set[int] = set()
        notified_tool_use_ids: set[str] = set()
        final_response: Any | None = None

        async for event in stream:
            _save_stream_chunk(capture_filename, event)
            event_type = getattr(event, "type", None)

            handled, reasoning_chars = await self._handle_responses_reasoning_delta(
                event=event,
                event_type=event_type,
                reasoning_segments=reasoning_segments,
                reasoning_summary_parts=reasoning_summary_parts,
                reasoning_chars=reasoning_chars,
                model=model,
            )
            if handled:
                continue

            if isinstance(event, ResponseTextDeltaEvent) or is_responses_text_delta_event(
                event_type
            ):
                delta = getattr(event, "delta", None)
                if delta:
                    estimated_tokens = self._emit_stream_text_delta(
                        text=delta,
                        model=model,
                        estimated_tokens=estimated_tokens,
                    )
                continue

            if is_responses_terminal_event(event_type):
                final_response = getattr(event, "response", None) or final_response
                continue
            if self._handle_responses_tool_stream_event(
                event=event,
                event_type=event_type,
                tool_state=tool_state,
                notified_tool_indices=notified_tool_indices,
                notified_tool_use_ids=notified_tool_use_ids,
                model=model,
            ):
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
            fetch_failure_message="Failed to fetch final Responses payload",
            use_exc_info_on_fetch_failure=True,
            incomplete_entries=tool_state.incomplete(),
            model=model,
            agent_name=self.name,
            chat_turn=self.chat_turn,
            logger=self.logger,
            notified_tool_indices=notified_tool_indices,
            emit_tool_fallback=emit_tool_fallback,
        )
        self._emit_deferred_mcp_result_notifications(
            final_response=final_response,
            tool_state=tool_state,
            model=model,
        )
        reasoning_parts = [
            part.text()
            for part in reasoning_summary_parts.values()
            if part.text()
        ]
        if reasoning_parts:
            summary_text = join_reasoning_summary_parts(reasoning_parts)
            return final_response, [summary_text] if summary_text else []
        return final_response, reasoning_segments.parts()

    def _emit_deferred_mcp_result_notifications(
        self,
        *,
        final_response: Any,
        tool_state: OpenAIToolStreamState,
        model: str,
    ) -> None:
        for index, item in enumerate(getattr(final_response, "output", []) or []):
            if responses_item_type(item) != "mcp_call":
                continue
            tool_use_id = responses_item_tool_use_id(item)
            item_id = responses_event_item_id(None, item)
            tool_info = tool_state.resolve(
                index=index,
                tool_use_id=tool_use_id,
                item_id=item_id,
            )
            if (
                tool_info is None
                or not tool_info.awaiting_output_item_done
                or tool_info.stop_notified
            ):
                continue
            tool_name = responses_tool_name(item)
            stop_payload = tool_event_payload(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                index=index,
                family="remote_tool",
                phase="result",
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
            self._notify_tool_stream_listeners("stop", stop_payload)
            self._log_tool_stream_event(
                model=model,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                event_type="stop",
            )
            tool_info.awaiting_output_item_done = False
            tool_info.stop_notified = True

    def _emit_tool_notification_fallback(
        self,
        output_items: list[Any],
        notified_indices: set[int],
        *,
        model: str,
        notified_tool_use_ids: set[str] | None = None,
    ) -> None:
        """Emit start/stop notifications when streaming metadata was missing."""
        if not output_items:
            return

        for index, item in enumerate(output_items):
            if index in notified_indices:
                continue
            if not item_is_responses_tool(item):
                continue

            tool_name, tool_use_id, family = fallback_tool_spec(item, index)
            if notified_tool_use_ids is not None and tool_use_id in notified_tool_use_ids:
                continue
            item_type = responses_item_type(item)
            if item_type_is_responses_function_tool_call(
                item_type
            ) and self._is_provider_managed_function_call(tool_name):
                family = self._tool_family_for_responses_item(
                    item_type=item_type,
                    tool_name=tool_name,
                )
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
