from typing import List

import pytest
from mcp import CallToolRequest, Tool
from mcp.types import CallToolRequestParams

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


class ToolGeneratingLlm(PassthroughLLM):
    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        tool_calls = {}
        tool_calls["my_id"] = CallToolRequest(
            method="tools/call", params=CallToolRequestParams(name="tool_function")
        )
        return Prompt.assistant(
            "Another turn",
            stop_reason=LlmStopReason.TOOL_USE,
            tool_calls=tool_calls,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_loop(fast_agent):
    @fast_agent.agent(instruction="You are a helpful AI Agent")
    async def agent_function():
        async with fast_agent.run() as agent:
            tool_llm = ToolGeneratingLlm()
            agent.default._llm = tool_llm
            assert "Another turn" == await agent.default.send(
                "New implementation", RequestParams(max_iterations=0)
            )

    await agent_function()


def tool_function() -> int:
    return 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_loop_construction():
    tool_llm = ToolGeneratingLlm()
    tool_agent = ToolAgent(AgentConfig("tool_calling"), [tool_function])
    tool_agent._llm = tool_llm
    result = await tool_agent.generate("test")
    assert "Another turn" == result.last_text()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_loop_unknown_tool():
    tool_llm = ToolGeneratingLlm()
    tool_agent = ToolAgent(AgentConfig("tool_calling"), [])
    tool_agent._llm = tool_llm

    tool_calls = {
        "my_id": CallToolRequest(
            method="tools/call", params=CallToolRequestParams(name="tool_function")
        )
    }
    assistant_message = Prompt.assistant(
        "Another turn",
        stop_reason=LlmStopReason.TOOL_USE,
        tool_calls=tool_calls,
    )

    tool_response = await tool_agent.run_tools(assistant_message)
    assert tool_response.channels is not None
    assert FAST_AGENT_ERROR_CHANNEL in tool_response.channels
    channel_content = tool_response.channels[FAST_AGENT_ERROR_CHANNEL][0]
    assert getattr(channel_content, "text", None) == "Tool 'tool_function' is not available"

    result = await tool_agent.generate("test")

    assert result.stop_reason == LlmStopReason.ERROR
    assert result.last_text() == "Another turn"
