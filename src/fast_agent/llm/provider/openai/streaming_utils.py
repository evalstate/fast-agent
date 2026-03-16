from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Callable, Protocol, Self, TypeVar

if TYPE_CHECKING:
    from fast_agent.core.logging.logger import Logger
from fast_agent.event_progress import ProgressAction

_StreamEventT = TypeVar("_StreamEventT")


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


class _IdleTimeoutAsyncStream(AsyncIterator[_StreamEventT]):
    """Apply an idle timeout to an async stream while preserving stream helpers.

    The timeout is enforced between stream events, not across the total stream
    lifetime. This allows healthy long-running generations to continue while
    still failing a stream that stops producing events entirely.
    """

    def __init__(
        self,
        stream: AsyncIterable[_StreamEventT],
        *,
        idle_timeout_seconds: float | None,
        timeout_message: str,
    ) -> None:
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._idle_timeout_seconds = idle_timeout_seconds
        self._timeout_message = timeout_message

    def __aiter__(self) -> Self:
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def __anext__(self) -> _StreamEventT:
        next_event = self._iterator.__anext__()
        if self._idle_timeout_seconds is None:
            return await next_event
        try:
            return await asyncio.wait_for(next_event, timeout=self._idle_timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(self._timeout_message) from exc


def with_stream_idle_timeout(
    stream: AsyncIterable[_StreamEventT],
    *,
    idle_timeout_seconds: float | None,
    timeout_message: str,
) -> AsyncIterator[_StreamEventT]:
    """Return a stream iterator that times out only when the stream goes idle."""

    return _IdleTimeoutAsyncStream(
        stream,
        idle_timeout_seconds=idle_timeout_seconds,
        timeout_message=timeout_message,
    )


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

    incomplete_tools = [
        f"{entry.tool_name}:{entry.tool_use_id}" for entry in incomplete_entries
    ]
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
        raise RuntimeError(
            "Streaming completed but tool call(s) never finished: "
            f"{', '.join(incomplete_tools)}"
        )


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
