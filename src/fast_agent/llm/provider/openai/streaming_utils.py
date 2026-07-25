from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fast_agent.core.logging.logger import Logger
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.tool_call_errors import format_incomplete_tool_call_error

CompletedOutputItem = tuple[int | None, int, Any]
"""An item observed on ``response.output_item.done``: (output_index, sequence, item)."""


def record_completed_output_item(
    completed_output_items: list[CompletedOutputItem],
    event: Any,
    *,
    fallback_sequence: int,
) -> None:
    """Remember a completed output item so an empty terminal payload can be rebuilt.

    ``output_index`` and ``sequence_number`` are both optional on the wire, so
    ``fallback_sequence`` (a stream-position counter) keeps ordering deterministic
    when the provider omits them.
    """
    item = getattr(event, "item", None)
    if item is None:
        return

    output_index = getattr(event, "output_index", None)
    sequence_number = getattr(event, "sequence_number", None)
    completed_output_items.append(
        (
            output_index if isinstance(output_index, int) else None,
            sequence_number if isinstance(sequence_number, int) else fallback_sequence,
            item,
        )
    )


def _reconstruct_empty_response_output(
    final_response: Any,
    completed_output_items: Sequence[CompletedOutputItem],
) -> tuple[Any, bool]:
    """Rebuild ``output`` from streamed items when the terminal payload arrives empty.

    Returns the (possibly copied) response and whether reconstruction happened.
    """
    if getattr(final_response, "output", None) or not completed_output_items:
        return final_response, False

    reconstructed = copy.copy(final_response)
    reconstructed.output = [
        item
        for _output_index, _sequence, item in sorted(
            completed_output_items,
            key=lambda entry: (entry[0] is None, entry[0] or 0, entry[1]),
        )
    ]
    return reconstructed, True


class ToolFallbackEmitter(Protocol):
    def __call__(
        self,
        output_items: list[Any],
        notified_indices: set[int],
        *,
        model: str,
    ) -> None: ...


class IncompleteToolEntry(Protocol):
    tool_name: str
    tool_use_id: str


def finalize_stream_response(
    *,
    final_response: Any,
    model: str,
    agent_name: str | None,
    chat_turn: Callable[[], int],
    logger: Logger,
    notified_tool_indices: set[int],
    emit_tool_fallback: ToolFallbackEmitter,
) -> None:
    usage = getattr(final_response, "usage", None)
    if usage:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        token_str = str(output_tokens).rjust(5)
        data = {
            "progress_action": ProgressAction.STREAMING,
            "model": model,
            "agent_name": agent_name,
            "chat_turn": chat_turn(),
            "details": token_str.strip(),
        }
        logger.info("Streaming progress", data=data)
        logger.info(
            f"Streaming complete - Model: {model}, Input tokens: {input_tokens}, Output tokens: {output_tokens}"
        )

    output_items = list(getattr(final_response, "output", []) or [])
    emit_tool_fallback(output_items, notified_tool_indices, model=model)


def validate_incomplete_tool_entries(
    *,
    incomplete_entries: Sequence[IncompleteToolEntry],
    final_response: Any,
    logger: Logger,
) -> None:
    if not incomplete_entries:
        return

    incomplete_tools = [f"{entry.tool_name}:{entry.tool_use_id}" for entry in incomplete_entries]
    response_status = getattr(final_response, "status", None)
    log_method = logger.warning if response_status == "incomplete" else logger.error
    log_method(
        "Tool call streaming incomplete - started but never finished",
        data={
            "incomplete_tools": incomplete_tools,
            "tool_count": len(incomplete_entries),
            "response_status": response_status,
        },
    )
    if response_status != "incomplete":
        raise RuntimeError(format_incomplete_tool_call_error(incomplete_tools))


async def fetch_and_finalize_stream_response(
    *,
    stream: Any,
    final_response: Any | None,
    fetch_failure_message: str,
    use_exc_info_on_fetch_failure: bool,
    incomplete_entries: Sequence[IncompleteToolEntry],
    model: str,
    agent_name: str | None,
    chat_turn: Callable[[], int],
    logger: Logger,
    notified_tool_indices: set[int],
    emit_tool_fallback: ToolFallbackEmitter,
    completed_output_items: Sequence[CompletedOutputItem] = (),
) -> Any:
    if final_response is None:
        try:
            final_response = await stream.get_final_response()
        except Exception as exc:
            if use_exc_info_on_fetch_failure:
                logger.warning(fetch_failure_message, exc_info=exc)
            else:
                logger.warning(fetch_failure_message, data={"error": str(exc)})
            raise

    final_response, reconstructed_output = _reconstruct_empty_response_output(
        final_response, completed_output_items
    )
    if reconstructed_output:
        logger.warning(
            "Reconstructed empty terminal Responses output from completed stream items",
            data={
                "model": model,
                "terminal_status": getattr(final_response, "status", None),
                "completed_output_item_count": len(completed_output_items),
                "completed_output_item_types": sorted(
                    {
                        str(getattr(item, "type", "unknown"))
                        for _output_index, _sequence, item in completed_output_items
                    }
                ),
            },
        )

    validate_incomplete_tool_entries(
        incomplete_entries=incomplete_entries,
        final_response=final_response,
        logger=logger,
    )
    finalize_stream_response(
        final_response=final_response,
        model=model,
        agent_name=agent_name,
        chat_turn=chat_turn,
        logger=logger,
        notified_tool_indices=notified_tool_indices,
        emit_tool_fallback=emit_tool_fallback,
    )
    return final_response
