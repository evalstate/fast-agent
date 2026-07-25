from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.streaming_timeouts import (
    StreamIdleTimeoutError,
    with_stream_idle_timeout,
)


class _ImmediateStream:
    def __init__(self, values: list[str]) -> None:
        self._values = iter(values)

    def __aiter__(self) -> _ImmediateStream:
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._values)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FirstThenIdleStream:
    def __init__(self) -> None:
        self._first = True

    def __aiter__(self) -> _FirstThenIdleStream:
        return self

    async def __anext__(self) -> str:
        if self._first:
            self._first = False
            return "first"
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class _IdleAnthropicStream:
    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self) -> _IdleAnthropicStream:
        return self

    async def __anext__(self) -> Any:
        await asyncio.sleep(1)
        raise StopAsyncIteration


class _AnthropicStreamManager:
    def __init__(self, stream: _IdleAnthropicStream) -> None:
        self.stream = stream

    async def __aenter__(self) -> _IdleAnthropicStream:
        return self.stream

    async def __aexit__(self, *_args: object) -> None:
        self.stream.closed = True


class _AnthropicStreamMethod:
    def __init__(self, manager: _AnthropicStreamManager) -> None:
        self.manager = manager

    def __call__(self, **_kwargs: Any) -> _AnthropicStreamManager:
        return self.manager


@pytest.mark.asyncio
async def test_stream_timing_records_first_wait_and_exceptional_inter_event_gaps() -> None:
    timed_stream = with_stream_idle_timeout(
        _ImmediateStream(["a", "b", "c"]),
        idle_timeout_seconds=None,
        monotonic=_SequenceClock([0.0, 2.0, 5.0, 17.0, 20.0, 31.0]),
    )

    assert await timed_stream.__anext__() == "a"
    assert await timed_stream.__anext__() == "b"
    assert await timed_stream.__anext__() == "c"

    timing = timed_stream.timing
    assert timing.events_received == 3
    assert timing.first_event_wait_seconds == 2.0
    assert timing.max_inter_event_wait_seconds == 12.0
    assert timing.inter_event_waits_over_threshold == 2
    assert timing.timed_out_wait_seconds is None


@pytest.mark.asyncio
async def test_stream_timing_records_timeout_after_events() -> None:
    timed_stream = with_stream_idle_timeout(
        _FirstThenIdleStream(),
        idle_timeout_seconds=0.01,
        monotonic=_SequenceClock([0.0, 0.1, 1.0, 1.01]),
    )

    assert await timed_stream.__anext__() == "first"
    with pytest.raises(StreamIdleTimeoutError) as exc_info:
        await timed_stream.__anext__()

    assert exc_info.value.events_received == 1
    assert timed_stream.timing.events_received == 1
    assert timed_stream.timing.timed_out_wait_seconds == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_anthropic_stream_enforces_between_event_idle_timeout() -> None:
    context = Context()
    context.config = Settings(
        anthropic=AnthropicSettings(api_key="test-key"),
    )
    llm = AnthropicLLM(context=context, model="claude-test")
    stream = _IdleAnthropicStream()
    manager = _AnthropicStreamManager(stream)
    anthropic = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(stream=_AnthropicStreamMethod(manager)),
        )
    )

    with pytest.raises(
        StreamIdleTimeoutError,
        match="No stream events were received for 0.01 seconds",
    ):
        await llm._execute_anthropic_stream(
            anthropic=cast("Any", anthropic),
            arguments={},
            model="claude-test",
            capture_filename=None,
            timeout_seconds=0.01,
        )

    assert stream.closed
