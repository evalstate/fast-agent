import json
from types import SimpleNamespace

import httpx2
import pytest
from openai import APIError

from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL, FAST_AGENT_RETRY
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import LlmStopReason, PromptMessageExtended, RequestParams


class FailingOpenAILLM(OpenAILLM):
    """Test double that always raises an APIError."""

    def __init__(self, **kwargs) -> None:
        super().__init__(provider=Provider.OPENAI, **kwargs)
        self.attempts = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools=None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self.attempts += 1
        self._notify_stream_listeners(StreamChunk(text=f"partial {self.attempts}"))
        raise APIError(
            "simulated failure",
            httpx2.Request("GET", "http://example.com"),
            body=None,
        )


@pytest.mark.asyncio
async def test_retry_exhaustion_returns_error_channel():
    ctx = SimpleNamespace(executor=None, config=None)
    llm = FailingOpenAILLM(context=ctx, name="fail-llm")
    llm.retry_count = 0

    response = await llm.generate([Prompt.user("hi")])

    assert llm.attempts == 1  # no retries when FAST_AGENT_RETRIES=0
    assert response.stop_reason == LlmStopReason.ERROR
    assert response.channels is not None
    assert FAST_AGENT_ERROR_CHANNEL in response.channels
    error_block = response.channels[FAST_AGENT_ERROR_CHANNEL][0]
    assert "request failed" in (get_text(error_block) or "")


@pytest.mark.asyncio
async def test_retry_attempts_and_backoff_are_configurable():
    ctx = SimpleNamespace(executor=None, config=None)
    llm = FailingOpenAILLM(context=ctx, name="fail-llm")
    llm.retry_count = 1
    llm.retry_backoff_seconds = 0.01
    stream_events = []
    llm.add_stream_listener(stream_events.append)

    response = await llm.generate([Prompt.user("hi")])

    assert llm.attempts == 2  # initial + 1 retry
    assert [(chunk.event, chunk.text) for chunk in stream_events] == [
        ("delta", "partial 1"),
        ("rollback", ""),
        ("delta", "partial 2"),
        ("rollback", ""),
    ]
    assert response.stop_reason == LlmStopReason.ERROR
    assert response.channels is not None
    retry_payload = json.loads(get_text(response.channels[FAST_AGENT_RETRY][0]) or "")
    assert retry_payload["provider_attempts"] == 2


@pytest.mark.asyncio
async def test_fatal_error_preserves_partial_stream_without_rollback() -> None:
    ctx = SimpleNamespace(executor=None, config=None)
    llm = FailingOpenAILLM(context=ctx, name="fail-llm")
    stream_events: list[StreamChunk] = []
    llm.add_stream_listener(stream_events.append)

    async def fail_after_streaming() -> None:
        llm._notify_stream_listeners(StreamChunk(text="useful partial"))
        raise ProviderKeyError("Missing API key")

    with pytest.raises(ProviderKeyError):
        await llm._execute_with_retry(fail_after_streaming)

    assert [(chunk.event, chunk.text) for chunk in stream_events] == [
        ("delta", "useful partial"),
    ]


@pytest.mark.asyncio
async def test_retry_notices_are_emitted_on_stderr(capsys):
    ctx = SimpleNamespace(executor=None, config=None)
    llm = FailingOpenAILLM(context=ctx, name="fail-llm")
    llm.retry_count = 1
    llm.retry_backoff_seconds = 0.01

    await llm.generate([Prompt.user("hi")])

    captured = capsys.readouterr()
    assert "Provider Error" not in captured.out
    assert "Retrying in" not in captured.out
    assert "Provider Error" in captured.err
    assert "Retrying in" in captured.err
