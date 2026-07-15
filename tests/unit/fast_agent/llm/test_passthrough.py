from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from fast_agent.llm.internal.passthrough import (
    CALL_TOOL_INDICATOR,
    FIXED_RESPONSE_INDICATOR,
    PassthroughLLM,
)
from fast_agent.mcp.prompt import Prompt

if TYPE_CHECKING:
    from fast_agent.mcp.interfaces import FastAgentLLMProtocol
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class FormattedResponse(BaseModel):
    thinking: str
    message: str


sample_json = '{"thinking":"The user wants to have a conversation about guitars, which are a broad...","message":"Sure! I love talking about guitars."}'


@pytest.mark.asyncio
async def test_simple_return():
    llm: FastAgentLLMProtocol = PassthroughLLM()
    response = await llm.generate(messages=[Prompt.user("playback message")])
    assert "assistant" == response.role
    assert "playback message" == response.first_text()


@pytest.mark.asyncio
async def test_set_fixed_return():
    llm: FastAgentLLMProtocol = PassthroughLLM()
    response: PromptMessageExtended = await llm.generate(
        messages=[Prompt.user(f"{FIXED_RESPONSE_INDICATOR} foo")]
    )
    assert "foo" == response.first_text()

    response: PromptMessageExtended = await llm.generate(
        messages=[Prompt.user("other messages respond with foo")]
    )
    assert "foo" == response.first_text()


@pytest.mark.asyncio
async def test_set_fixed_return_ignores_not_set():
    llm: FastAgentLLMProtocol = PassthroughLLM()
    response: PromptMessageExtended = await llm.generate(
        messages=[Prompt.user(f"{FIXED_RESPONSE_INDICATOR}")]
    )
    assert "***FIXED_RESPONSE" == response.first_text()

    response: PromptMessageExtended = await llm.generate(messages=[Prompt.user("ignored message")])
    assert "ignored message" == response.first_text()


@pytest.mark.asyncio
async def test_parse_tool_call_no_args():
    llm: FastAgentLLMProtocol = PassthroughLLM()
    name, args = llm._parse_tool_command(f"{CALL_TOOL_INDICATOR} mcp_tool_name")
    assert "mcp_tool_name" == name
    assert None is args


@pytest.mark.asyncio
async def test_parse_tool_call_with_args():
    llm: FastAgentLLMProtocol = PassthroughLLM()
    name, args = llm._parse_tool_command(
        f'{CALL_TOOL_INDICATOR} mcp_tool_name_args {{"arg": "value"}}'
    )
    assert "mcp_tool_name_args" == name
    assert args is not None
    assert "value" == args["arg"]


@pytest.mark.asyncio
async def test_generates_structured():
    llm: FastAgentLLMProtocol = PassthroughLLM()

    model, response = await llm.structured([Prompt.user(sample_json)], FormattedResponse)
    assert model is not None
    assert (
        model.thinking
        == "The user wants to have a conversation about guitars, which are a broad..."
    )


@pytest.mark.asyncio
async def test_returns_assistant_message_verbatim():
    llm: FastAgentLLMProtocol = PassthroughLLM()
    assistant_msg = Prompt.assistant("already answered")
    result = await llm.generate([assistant_msg])
    assert result.role == "assistant"
    assert result.first_text() == "already answered"


@pytest.mark.asyncio
async def test_passthrough_does_not_report_character_counts_as_tokens():
    llm: FastAgentLLMProtocol = PassthroughLLM()

    await llm.generate(messages=[Prompt.user("test message")])
    await llm.generate(messages=[Prompt.user("second message")])

    assert llm.usage_accumulator.turns == []
    assert llm.usage_accumulator.summary.total is None


@pytest.mark.asyncio
async def test_passthrough_tool_call_does_not_create_token_usage():
    llm: FastAgentLLMProtocol = PassthroughLLM()

    await llm.generate(messages=[Prompt.user(f"{CALL_TOOL_INDICATOR} some_tool {{}}")])

    assert llm.usage_accumulator.turns == []
