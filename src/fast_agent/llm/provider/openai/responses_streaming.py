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
from fast_agent.llm.provider.openai.streaming_utils import fetch_and_finalize_stream_response
from fast_agent.llm.provider.openai.tool_event_helpers import (
    fallback_tool_spec,
    item_is_responses_tool,
    responses_tool_name,
    responses_tool_use_id,
    tool_event_payload,
    tool_family_for_item_type,
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
    normalize_reasoning_delta,
)

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


def _preview_json_like(value: Any) -> str | None:
    normalized = snapshot_json_value(value)
    if normalized is None:
        return None
    if normalized == {} or normalized == []:
        return None
    if isinstance(normalized, str):
        preview = normalized.strip()
    else:
        preview = json.dumps(normalized)
    if not preview:
        return None
    if len(preview) > 120:
        return f"{preview[:117]}..."
    return preview


def _mcp_call_output_chunk(output: Any) -> str | None:
    return _preview_json_like(output)


def _tool_search_arguments_chunk(arguments: Any) -> str | None:
    return _preview_json_like(arguments)

def _responses_tool_family(item: Any) -> ToolActivityFamily:
    return tool_family_for_item_type(getattr(item, "type", None))


def _tool_progress_display(item: Any) -> tuple[ToolActivityFamily, str, str | None]:
    tool_name = responses_tool_name(item)
    family = _responses_tool_family(item)
    display_name = build_tool_activity_presentation(
        tool_name=tool_name,
        family=family,
        phase="call",
    ).display_name

    item_type = getattr(item, "type", None)
    if item_type == "tool_search_call":
        return family, display_name, (
            _tool_search_arguments_chunk(getattr(item, "arguments", None))
            or tool_activity_status_text(family=family, status="in_progress")
        )
    if item_type in {"web_search_call", "mcp_list_tools"}:
        return family, display_name, tool_activity_status_text(family=family, status="in_progress")
    if item_type == "mcp_call":
        arguments = getattr(item, "arguments", None)
        return family, display_name, (
            arguments if isinstance(arguments, str) and arguments else None
        )
    return family, display_name, None
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

    def _log_tool_stream_event(
        self,
        *,
        model: str,
        tool_name: str | None,
        tool_use_id: str | None,
        event_type: Literal["start", "stop"],
    ) -> None:
        message = (
            "Model started streaming tool call"
            if event_type == "start"
            else "Model finished streaming tool call"
        )
        self.logger.info(
            message,
            data={
                "progress_action": ProgressAction.CALLING_TOOL,
                "agent_name": self.name,
                "model": model,
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "tool_event": event_type,
            },
        )

    def _handle_responses_output_item_added(
        self,
        *,
        event: Any,
        tool_state: OpenAIToolStreamState,
        notified_tool_indices: set[int],
        model: str,
    ) -> bool:
        item = getattr(event, "item", None)
        if not item_is_responses_tool(item):
            return False

        index = getattr(event, "output_index", None)
        if index is None:
            return True

        item_type = getattr(item, "type", None) or "tool"
        tool_info = tool_state.register(
            tool_use_id=responses_tool_use_id(
                item,
                index,
                getattr(event, "item_id", None),
            ),
            name=responses_tool_name(item),
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
        family, display_name, display_chunk = _tool_progress_display(item)
        payload = tool_event_payload(
            tool_name=tool_info.tool_name,
            tool_use_id=tool_info.tool_use_id,
            index=index,
            family=family,
            phase="call",
            chunk=display_chunk,
        )
        payload["tool_display_name"] = display_name
        self._notify_tool_stream_listeners("start", payload)
        self._log_tool_stream_event(
            model=model,
            tool_name=tool_info.tool_name,
            tool_use_id=tool_info.tool_use_id,
            event_type="start",
        )
        tool_info.start_notified = True
        notified_tool_indices.add(index)
        return True

    def _handle_responses_argument_delta(
        self,
        *,
        event: Any,
        tool_state: OpenAIToolStreamState,
    ) -> bool:
        index = getattr(event, "output_index", None)
        if index is None:
            return True

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

        if tool_info is None:
            payload = {
                "tool_name": None,
                "tool_use_id": None,
                "index": index,
                "chunk": getattr(event, "delta", None),
            }
        else:
            payload = tool_event_payload(
                tool_name=tool_info.tool_name,
                tool_use_id=tool_info.tool_use_id,
                index=index,
                family=tool_family_for_item_type(tool_info.item_type),
                phase="call",
                chunk=getattr(event, "delta", None),
            )
        self._notify_tool_stream_listeners(event_name, payload)
        return True

    def _resolve_lifecycle_tool_info(
        self,
        *,
        event_type: str,
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
        if "web_search_call" in event_type:
            fallback_name = "web_search"
            fallback_item_type = "web_search_call"
        elif "mcp_list_tools" in event_type:
            fallback_name = "mcp_list_tools"
            fallback_item_type = "mcp_list_tools"
        else:
            fallback_name = "mcp_call"
            fallback_item_type = "mcp_call"

        return tool_state.register(
            tool_use_id=event_item_id or f"{fallback_name}-{fallback_index}",
            name=fallback_name,
            index=fallback_index,
            item_id=event_item_id,
            item_type=fallback_item_type,
            kind="web_search" if fallback_item_type == "web_search_call" else "tool",
        )

    def _handle_responses_tool_lifecycle_event(
        self,
        *,
        event: Any,
        event_type: str,
        tool_state: OpenAIToolStreamState,
        notified_tool_indices: set[int],
        model: str,
    ) -> bool:
        event_index = getattr(event, "output_index", None)
        event_item_id = getattr(event, "item_id", None)
        tool_info = self._resolve_lifecycle_tool_info(
            event_type=event_type,
            event_index=event_index,
            event_item_id=event_item_id,
            tool_state=tool_state,
        )
        if tool_info is None:
            return True

        index = tool_info.index if tool_info.index is not None else -1
        tool_use_id = tool_info.tool_use_id
        tool_name = tool_info.tool_name or "web_search"
        status = event_type.rsplit(".", 1)[-1]
        family = tool_family_for_item_type(tool_info.item_type)
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

        if event_type in _TOOL_START_EVENT_TYPES and not tool_info.start_notified:
            self._notify_tool_stream_listeners("start", payload)
            self._log_tool_stream_event(
                model=model,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                event_type="start",
            )
            tool_info.start_notified = True
            if index >= 0:
                notified_tool_indices.add(index)

        if event_type not in _TOOL_STOP_EVENT_TYPES:
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
        return True

    def _handle_responses_output_item_done(
        self,
        *,
        event: Any,
        tool_state: OpenAIToolStreamState,
        notified_tool_indices: set[int],
        model: str,
    ) -> bool:
        item = getattr(event, "item", None)
        if not item_is_responses_tool(item):
            return False

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
            return True
        if tool_info is None:
            return True

        tool_name = responses_tool_name(item)
        tool_use_id = tool_use_id or tool_info.tool_use_id
        if index is None:
            index = tool_info.index if tool_info.index is not None else -1
        item_type = tool_info.item_type if tool_info else getattr(item, "type", None)
        family = tool_family_for_item_type(item_type if isinstance(item_type, str) else None)
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
        if index >= 0:
            notified_tool_indices.add(index)
        return True

    async def _process_stream(
        self, stream: Any, model: str, capture_filename: Path | None
    ) -> tuple[Any, list[str]]:
        estimated_tokens = 0
        reasoning_chars = 0
        reasoning_segments = ReasoningTextAccumulator(normalizer=normalize_reasoning_delta)
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
                    normalized_delta = reasoning_segments.append(delta)
                    if not normalized_delta:
                        continue
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
                self._handle_responses_output_item_added(
                    event=event,
                    tool_state=tool_state,
                    notified_tool_indices=notified_tool_indices,
                    model=model,
                )
                continue

            if event_type in {
                "response.function_call_arguments.delta",
                "response.custom_tool_call_input.delta",
                "response.mcp_call_arguments.delta",
            }:
                self._handle_responses_argument_delta(
                    event=event,
                    tool_state=tool_state,
                )
                continue

            if event_type in (_TOOL_START_EVENT_TYPES | _TOOL_STOP_EVENT_TYPES):
                self._handle_responses_tool_lifecycle_event(
                    event=event,
                    event_type=event_type,
                    tool_state=tool_state,
                    notified_tool_indices=notified_tool_indices,
                    model=model,
                )
                continue

            if event_type == "response.output_item.done":
                self._handle_responses_output_item_done(
                    event=event,
                    tool_state=tool_state,
                    notified_tool_indices=notified_tool_indices,
                    model=model,
                )
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
        return final_response, reasoning_segments.parts()

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
                "tool_search_call",
                "web_search_call",
                "mcp_list_tools",
                "mcp_call",
            }:
                continue

            tool_name, tool_use_id, family = fallback_tool_spec(item, index)
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
