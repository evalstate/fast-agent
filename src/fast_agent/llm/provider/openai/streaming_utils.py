from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from fast_agent.core.logging.logger import Logger
from fast_agent.event_progress import ProgressAction


class ToolFallbackEmitter(Protocol):
    def __call__(
        self,
        output_items: list[Any],
        notified_indices: set[int],
        *,
        model: str,
    ) -> None: ...


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
