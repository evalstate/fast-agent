from types import SimpleNamespace

import pytest
from httpx import Request
from openai import APIError

from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.core.prompt import Prompt
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.helpers.content_helpers import get_text
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
        raise APIError("simulated failure", Request("GET", "http://example.com"), body=None)


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

    response = await llm.generate([Prompt.user("hi")])

    assert llm.attempts == 2  # initial + 1 retry
    assert response.stop_reason == LlmStopReason.ERROR
