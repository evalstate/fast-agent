from typing import List

import pytest
from mcp import CallToolRequest, Tool
from mcp.types import CallToolRequestParams

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from mcp_agent.core.prompt import Prompt


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
