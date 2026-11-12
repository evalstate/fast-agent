"""OpenAI-compatible chat completions: an end_turn with no content and no tool call
(often a stream cut off mid-reasoning) is asked once more, as OpenAI Responses does.

Regression: on an HF-routed DeepSeek V4.1 Flash run three trials ended this way; the
agent loop took the empty reply as the end of its work.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import TextContent

from fast_agent.constants import (
    FAST_AGENT_PROVIDER_ERROR,
    FAST_AGENT_RETRY,
    OPENAI_COMPACTION_ITEMS,
)
from fast_agent.context import Context
from fast_agent.llm.provider.error_utils import is_empty_completion
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.mcp.prompt import Prompt
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.types import PromptMessageExtended


def _empty(reasoning: str = "Let me try the near-identity pai") -> PromptMessageExtended:
    response = Prompt.assistant("", stop_reason=LlmStopReason.END_TURN)
    response.channels = {"reasoning": [TextContent(type="text", text=reasoning)]}
    return response


def _retries(response: PromptMessageExtended) -> list[dict[str, Any]]:
    block = (response.channels or {})[FAST_AGENT_RETRY][0]
    assert isinstance(block, TextContent)
    return json.loads(block.text)["retries"]


class ScriptedOpenAI(OpenAILLM):
    """Replies from a script; records every request it was sent."""

    def __init__(self, replies: list[PromptMessageExtended]) -> None:
        super().__init__(context=Context(), model="deepseek-ai/DeepSeek-V4.1-Flash")
        self.replies = replies
        self.requests: list[Any] = []

    async def _run_openai_completion_request(self, request: Any) -> Any:
        self.requests.append(request)
        return None

    async def _finalize_openai_completion(self, request: Any, completion: Any) -> Any:
        request.messages.append({"role": "assistant", "content": ""})  # as the real one does
        return self.replies[len(self.requests) - 1]


@pytest.mark.asyncio
async def test_an_empty_end_turn_is_asked_once_more_with_a_fresh_request() -> None:
    llm = ScriptedOpenAI([_empty(), Prompt.assistant("Done.")])

    response = await llm._openai_completion([{"role": "user", "content": "go"}])

    assert response.all_text() == "Done."
    (retry,) = _retries(response)
    assert retry["reason"] == "empty_completion"
    assert retry["error_type"] == "EmptyCompletionError"
    first, second = llm.requests
    assert first.messages is not second.messages
    assert [m["role"] for m in second.messages] == ["user", "assistant"]  # one reply, not two


@pytest.mark.asyncio
async def test_empty_twice_is_an_error_not_a_finished_turn() -> None:
    llm = ScriptedOpenAI([_empty(), _empty()])

    response = await llm._openai_completion([{"role": "user", "content": "go"}])

    assert len(llm.requests) == 2
    assert response.stop_reason == LlmStopReason.ERROR
    assert "no assistant content or tool calls" in response.all_text()
    assert [r["reason"] for r in _retries(response)] == ["empty_completion"]
    block = (response.channels or {})[FAST_AGENT_PROVIDER_ERROR][0]
    assert isinstance(block, TextContent)
    error = json.loads(block.text)
    assert error == {
        "schema": "fast-agent.provider-error/v1",
        "error_type": "EmptyCompletionError",
        "status": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        Prompt.assistant("Done.", stop_reason=LlmStopReason.END_TURN),
        Prompt.assistant("", stop_reason=LlmStopReason.MAX_TOKENS),
        Prompt.assistant("", stop_reason=LlmStopReason.CANCELLED),
    ],
)
async def test_replies_and_other_stops_are_returned_as_they_are(
    reply: PromptMessageExtended,
) -> None:
    llm = ScriptedOpenAI([reply])

    assert await llm._openai_completion([{"role": "user", "content": "go"}]) is reply
    assert len(llm.requests) == 1
    assert FAST_AGENT_RETRY not in (reply.channels or {})


def test_a_compaction_checkpoint_is_not_an_empty_completion() -> None:
    checkpoint = Prompt.assistant("", stop_reason=LlmStopReason.END_TURN)
    checkpoint.channels = {OPENAI_COMPACTION_ITEMS: [TextContent(type="text", text="{}")]}

    assert not is_empty_completion(checkpoint)
    assert is_empty_completion(_empty())
