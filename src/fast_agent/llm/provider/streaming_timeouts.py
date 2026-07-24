from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Self, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Awaitable

T = TypeVar("T")


class StreamIdleTimeoutError(TimeoutError):
    """Raised when an established provider stream stops producing events."""

    def __init__(self, timeout_seconds: float, *, events_received: int) -> None:
        super().__init__(f"No stream events were received for {timeout_seconds} seconds.")
        self.timeout_seconds = timeout_seconds
        self.events_received = events_received


class _IdleTimeoutAsyncStream(AsyncIterator[T]):
    """Apply an idle timeout between provider events while preserving stream helpers."""

    def __init__(
        self,
        stream: "AsyncIterable[T]",
        *,
        idle_timeout_seconds: float | None,
    ) -> None:
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._idle_timeout_seconds = idle_timeout_seconds
        self._events_received = 0

    def __aiter__(self) -> Self:
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def __anext__(self) -> T:
        next_event = self._iterator.__anext__()
        if self._idle_timeout_seconds is None:
            event = await next_event
        else:
            try:
                event = await asyncio.wait_for(
                    next_event,
                    timeout=self._idle_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise StreamIdleTimeoutError(
                    self._idle_timeout_seconds,
                    events_received=self._events_received,
                ) from exc
        self._events_received += 1
        return event


def with_stream_idle_timeout(
    stream: "AsyncIterable[T]",
    *,
    idle_timeout_seconds: float | None,
) -> "AsyncIterator[T]":
    """Return a stream iterator that times out only between provider events."""

    return _IdleTimeoutAsyncStream(
        stream,
        idle_timeout_seconds=idle_timeout_seconds,
    )


async def await_stream_start(
    awaitable: "Awaitable[T]",
    *,
    timeout_seconds: float | None,
    timeout_message: str,
) -> T:
    """Await stream startup with the same timeout semantics as stream idleness."""

    if timeout_seconds is None:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(timeout_message) from exc


@asynccontextmanager
async def enter_stream_with_timeout(
    stream_context: Any,
    *,
    timeout_seconds: float | None,
    timeout_message: str,
) -> "AsyncIterator[Any]":
    """Enter an async stream context with a startup timeout."""

    stream = await await_stream_start(
        stream_context.__aenter__(),
        timeout_seconds=timeout_seconds,
        timeout_message=timeout_message,
    )
    try:
        yield stream
    except BaseException as exc:
        suppress = await stream_context.__aexit__(type(exc), exc, exc.__traceback__)
        if not suppress:
            raise
    else:
        await stream_context.__aexit__(None, None, None)
