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
            tool_agent: ToolAgent = ToolAgent(AgentConfig("tool_calling"), [tool_function])
            tool_agent._llm = tool_llm
            assert "Another turn" == await tool_agent.send(
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

    # make sure that the error content is also visible to the LLM via this "User" message
    assert "user" == tool_response.role
    assert "Tool 'tool_function' is not available" in tool_response.first_text()


class PersistentToolGeneratingLlm(PassthroughLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_count = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self.call_count += 1
        tool_calls = {
            f"persistent_{self.call_count}": CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="tool_function"),
            )
        }
        return Prompt.assistant(
            "Loop again",
            stop_reason=LlmStopReason.TOOL_USE,
            tool_calls=tool_calls,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_loop_respects_llm_default_max_iterations():
    tool_llm = PersistentToolGeneratingLlm(request_params=RequestParams(max_iterations=2))
    tool_agent = ToolAgent(AgentConfig("tool_calling"), [tool_function])
    tool_agent._llm = tool_llm

    await tool_agent.generate("test default")

    expected_calls = tool_llm.default_request_params.max_iterations + 1
    assert tool_llm.call_count == expected_calls


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_loop_respects_request_param_override():
    tool_llm = PersistentToolGeneratingLlm(request_params=RequestParams(max_iterations=5))
    tool_agent = ToolAgent(AgentConfig("tool_calling"), [tool_function])
    tool_agent._llm = tool_llm

    override_params = RequestParams(max_iterations=1)
    await tool_agent.generate("test override", override_params)

    expected_calls = override_params.max_iterations + 1
    assert tool_llm.call_count == expected_calls
