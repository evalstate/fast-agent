import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams, CallToolResult, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core.prompt import Prompt
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import LlmStopReason, PromptMessageExtended, RequestParams


class FakeLLM(FastAgentLLM[PromptMessageExtended, PromptMessageExtended]):
    def __init__(self, **kwargs):
        super().__init__(provider=Provider.FAST_AGENT, name="fake-llm", **kwargs)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools=None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        return Prompt.assistant("ok")

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: list[PromptMessageExtended],
        model,
        request_params: RequestParams | None = None,
    ):
        return None, Prompt.assistant("ok")

    def _convert_extended_messages_to_provider(
        self, messages: list[PromptMessageExtended]
    ) -> list[PromptMessageExtended]:
        return messages


def _seed_pending_tool_call(agent: ToolAgent) -> None:
    pending_tool_call = CallToolRequest(
        params=CallToolRequestParams(name="fake_tool", arguments={})
    )
    agent._message_history = [
        Prompt.assistant(
            "pending",
            stop_reason=LlmStopReason.TOOL_USE,
            tool_calls={"call-1": pending_tool_call},
        )
    ]








@pytest.mark.asyncio
async def test_unanswered_tool_call_raises_on_new_turn():
    agent = ToolAgent(AgentConfig("test-agent"))
    agent._llm = FakeLLM()
    _seed_pending_tool_call(agent)

    with pytest.raises(ValueError, match="Invalid conversation history"):
        await agent.generate_impl([Prompt.user("hello")], RequestParams())


@pytest.mark.asyncio
async def test_unanswered_tool_call_allowed_with_tool_results():
    agent = ToolAgent(AgentConfig("test-agent"))
    agent._llm = FakeLLM()
    _seed_pending_tool_call(agent)

    tool_result = CallToolResult(
        content=[TextContent(type="text", text="ok")],
    )
    tool_result_message = PromptMessageExtended(
        role="user",
        tool_results={"call-1": tool_result},
    )

    result = await agent.generate_impl([tool_result_message], RequestParams())
    assert result.role == "assistant"
