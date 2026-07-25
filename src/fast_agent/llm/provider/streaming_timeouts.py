from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Self, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Awaitable, Callable

T = TypeVar("T")

STREAM_GAP_OBSERVATION_THRESHOLD_SECONDS: Final = 10.0


@dataclass(frozen=True, slots=True)
class StreamTiming:
    events_received: int
    first_event_wait_seconds: float | None
    max_inter_event_wait_seconds: float | None
    inter_event_waits_over_threshold: int
    timed_out_wait_seconds: float | None


def stream_timing_payload(
    timing: StreamTiming,
    *,
    timed_out: bool,
) -> dict[str, int | float | bool | None]:
    """Render stream timing for structured logs and diagnostics channels."""

    def milliseconds(seconds: float | None) -> float | None:
        return round(seconds * 1000.0, 2) if seconds is not None else None

    payload: dict[str, int | float | bool | None] = {
        "events_received": timing.events_received,
        "first_event_wait_ms": milliseconds(timing.first_event_wait_seconds),
        "max_inter_event_wait_ms": milliseconds(timing.max_inter_event_wait_seconds),
        "inter_event_waits_over_10s": timing.inter_event_waits_over_threshold,
        "timed_out": timed_out,
    }
    if timed_out:
        payload["timed_out_wait_ms"] = milliseconds(timing.timed_out_wait_seconds)
    return payload


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
        monotonic: "Callable[[], float]" = time.monotonic,
    ) -> None:
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._idle_timeout_seconds = idle_timeout_seconds
        self._monotonic = monotonic
        self._events_received = 0
        self._first_event_wait_seconds: float | None = None
        self._max_inter_event_wait_seconds: float | None = None
        self._inter_event_waits_over_threshold = 0
        self._timed_out_wait_seconds: float | None = None

    def __aiter__(self) -> Self:
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    @property
    def timing(self) -> StreamTiming:
        return StreamTiming(
            events_received=self._events_received,
            first_event_wait_seconds=self._first_event_wait_seconds,
            max_inter_event_wait_seconds=self._max_inter_event_wait_seconds,
            inter_event_waits_over_threshold=self._inter_event_waits_over_threshold,
            timed_out_wait_seconds=self._timed_out_wait_seconds,
        )

    async def __anext__(self) -> T:
        wait_started = self._monotonic()
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
                self._timed_out_wait_seconds = self._monotonic() - wait_started
                raise StreamIdleTimeoutError(
                    self._idle_timeout_seconds,
                    events_received=self._events_received,
                ) from exc

        wait_seconds = self._monotonic() - wait_started
        if self._events_received == 0:
            self._first_event_wait_seconds = wait_seconds
        else:
            self._max_inter_event_wait_seconds = max(
                self._max_inter_event_wait_seconds or 0.0,
                wait_seconds,
            )
            if wait_seconds > STREAM_GAP_OBSERVATION_THRESHOLD_SECONDS:
                self._inter_event_waits_over_threshold += 1
        self._events_received += 1
        return event


def with_stream_idle_timeout(
    stream: "AsyncIterable[T]",
    *,
    idle_timeout_seconds: float | None,
    monotonic: "Callable[[], float]" = time.monotonic,
) -> "_IdleTimeoutAsyncStream[T]":
    """Return a stream iterator that times out only between provider events."""

    return _IdleTimeoutAsyncStream(
        stream,
        idle_timeout_seconds=idle_timeout_seconds,
        monotonic=monotonic,
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
