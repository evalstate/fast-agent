import asyncio

import pytest
from mcp import CallToolRequest, Tool
from mcp.types import CallToolRequestParams, CallToolResult

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.config import get_settings, update_global_settings
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.session import get_session_manager, reset_session_manager
from fast_agent.types.llm_stop_reason import LlmStopReason


class ToolGeneratingLlm(PassthroughLLM):
    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
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
        async with fast_agent.run():
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
        multipart_messages: list[PromptMessageExtended],
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


class ExplodingAfterToolResultLlm(PassthroughLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._turn = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self._turn += 1
        if self._turn == 1:
            tool_calls = {
                "side_effect_call": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="side_effect_tool", arguments={}),
                )
            }
            return Prompt.assistant(
                "run tool",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls=tool_calls,
            )

        raise RuntimeError("llm boom")


class ContinuedToolResultLlm(PassthroughLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen_last_message: PromptMessageExtended | None = None

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self.seen_last_message = multipart_messages[-1].model_copy(deep=True)
        tool_result_text = " ".join(
            (get_text(tool_result.content[0]) or "")
            for tool_result in (self.seen_last_message.tool_results or {}).values()
            if tool_result.content
        )
        combined_text = "\n".join(
            text for text in [tool_result_text, self.seen_last_message.all_text()] if text
        )
        return Prompt.assistant(combined_text, stop_reason=LlmStopReason.END_TURN)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_preserves_completed_tool_result_after_followup_llm_failure(tmp_path):
    old_settings = get_settings()
    override = old_settings.model_copy(update={"environment_dir": str(tmp_path / "env")})
    update_global_settings(override)
    reset_session_manager()

    tool_runs = 0

    async def side_effect_tool() -> str:
        nonlocal tool_runs
        tool_runs += 1
        await asyncio.sleep(0)
        return f"ok {tool_runs}"

    try:
        exploding_llm = ExplodingAfterToolResultLlm()
        agent = ToolAgent(AgentConfig("tool-loop-resume"), [side_effect_tool])
        agent._llm = exploding_llm

        with pytest.raises(RuntimeError, match="llm boom"):
            await agent.generate("trigger")

        assert tool_runs == 1

        manager = get_session_manager()
        session = manager.current_session
        assert session is not None

        history_path = session.latest_history_path(agent.name)
        assert history_path is not None
        assert history_path.exists()

        saved_messages = load_prompt(history_path)
        assert saved_messages
        assert saved_messages[-1].role == "user"
        assert saved_messages[-1].tool_results is not None
        assert "side_effect_call" in saved_messages[-1].tool_results
        saved_result = saved_messages[-1].tool_results["side_effect_call"]
        assert isinstance(saved_result, CallToolResult)
        assert len(saved_result.content) == 1
        assert saved_result.content[0].text == "ok 1"

        resumed_llm = ContinuedToolResultLlm()
        resumed_agent = ToolAgent(AgentConfig("tool-loop-resume"), [side_effect_tool])
        resumed_agent._llm = resumed_llm

        resumed = await manager.resume_session_async(resumed_agent)
        assert resumed is not None

        result = await resumed_agent.generate("after resume")

        assert result.stop_reason == LlmStopReason.END_TURN
        assert result.last_text() == "ok 1\nafter resume"
        assert tool_runs == 1
        assert resumed_llm.seen_last_message is not None
        assert resumed_llm.seen_last_message.tool_results is not None
        assert "side_effect_call" in resumed_llm.seen_last_message.tool_results
        assert resumed_llm.seen_last_message.all_text() == "after resume"
        assert resumed_llm.seen_last_message.tool_results["side_effect_call"].content[0].text == "ok 1"
    finally:
        update_global_settings(old_settings)
        reset_session_manager()
