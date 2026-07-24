from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.streaming_timeouts import StreamIdleTimeoutError


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
