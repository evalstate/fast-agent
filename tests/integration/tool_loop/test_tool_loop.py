import asyncio

import pytest
from mcp import CallToolRequest, Tool
from mcp_types import CallToolRequestParams, CallToolResult, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.config import get_settings, update_global_settings
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.session import SessionManager, reset_session_manager, set_session_manager
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
    result = await tool_agent.generate("test", RequestParams(max_iterations=0))
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
    assert tool_response.channels is None or FAST_AGENT_ERROR_CHANNEL not in tool_response.channels

    assert "user" == tool_response.role
    assert tool_response.tool_results is not None
    unknown_result = tool_response.tool_results["my_id"]
    assert unknown_result.is_error is True
    assert get_text(unknown_result.content[0]) == (
        "Tool 'tool_function' is not available. No tools are currently available."
    )


class CorrectingToolNameLlm(PassthroughLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_count = 0
        self.seen_unknown_tool_error = False

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self.call_count += 1
        if self.call_count == 1:
            return Prompt.assistant(
                "Use the wrong name",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "wrong_name": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="missing_tool"),
                    )
                },
            )
        if self.call_count == 2:
            latest_message = multipart_messages[-1]
            latest_results = latest_message.tool_results or {}
            latest_result = latest_results.get("wrong_name")
            self.seen_unknown_tool_error = bool(
                latest_result
                and latest_result.content
                and "Available tools: tool_function." in (get_text(latest_result.content[0]) or "")
            )
            return Prompt.assistant(
                "Correct the tool name",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "correct_name": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="tool_function"),
                    )
                },
            )
        return Prompt.assistant("Recovered", stop_reason=LlmStopReason.END_TURN)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_loop_allows_unknown_tool_name_correction():
    tool_llm = CorrectingToolNameLlm()
    tool_agent = ToolAgent(AgentConfig("tool_calling"), [tool_function])
    tool_agent._llm = tool_llm

    result = await tool_agent.generate("test")

    assert result.last_text() == "Recovered"
    assert tool_llm.call_count == 3
    assert tool_llm.seen_unknown_tool_error is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unknown_parallel_tool_does_not_block_valid_sibling():
    tool_runs = 0

    def counting_tool() -> int:
        nonlocal tool_runs
        tool_runs += 1
        return tool_runs

    tool_agent = ToolAgent(AgentConfig("tool_calling"), [counting_tool])
    assistant_message = Prompt.assistant(
        "Run both",
        stop_reason=LlmStopReason.TOOL_USE,
        tool_calls={
            "unknown": CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="missing_tool"),
            ),
            "valid": CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="counting_tool"),
            ),
        },
    )

    tool_response = await tool_agent.run_tools(assistant_message)

    assert tool_runs == 1
    assert tool_response.channels is None or FAST_AGENT_ERROR_CHANNEL not in tool_response.channels
    assert tool_response.tool_results is not None
    assert tool_response.tool_results["unknown"].is_error is True
    assert tool_response.tool_results["valid"].is_error is False


class PersistentUnknownToolLlm(PassthroughLLM):
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
        return Prompt.assistant(
            "Use the unavailable name",
            stop_reason=LlmStopReason.TOOL_USE,
            tool_calls={
                f"unknown_{self.call_count}": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="missing_tool"),
                )
            },
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unknown_tool_name_recovery_respects_max_iterations():
    tool_llm = PersistentUnknownToolLlm()
    tool_agent = ToolAgent(AgentConfig("tool_calling"), [tool_function])
    tool_agent._llm = tool_llm

    result = await tool_agent.generate("test", RequestParams(max_iterations=1))

    assert result.stop_reason == LlmStopReason.MAX_ITERATIONS
    assert tool_llm.call_count == 2


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
    override = old_settings.model_copy(update={"home": str(tmp_path / "env")})
    update_global_settings(override)
    reset_session_manager()

    tool_runs = 0

    async def side_effect_tool() -> str:
        nonlocal tool_runs
        tool_runs += 1
        await asyncio.sleep(0)
        return f"ok {tool_runs}"

    try:
        manager = SessionManager(home_override=tmp_path / "env")
        set_session_manager(manager)
        exploding_llm = ExplodingAfterToolResultLlm()
        agent = ToolAgent(AgentConfig("tool-loop-resume"), [side_effect_tool])
        agent._llm = exploding_llm

        with pytest.raises(RuntimeError, match="llm boom"):
            await agent.generate("trigger")

        assert tool_runs == 1

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
        saved_content = saved_result.content[0]
        assert isinstance(saved_content, TextContent)
        assert saved_content.text == "ok 1"

        resumed_llm = ContinuedToolResultLlm()
        resumed_agent = ToolAgent(AgentConfig("tool-loop-resume"), [side_effect_tool])
        resumed_agent._llm = resumed_llm

        resumed = await manager.resume_session_agents_async(
            {resumed_agent.name: resumed_agent},
            fallback_agent_name=resumed_agent.name,
        )
        assert resumed is not None

        result = await resumed_agent.generate("after resume")

        assert result.stop_reason == LlmStopReason.END_TURN
        assert result.last_text() == "ok 1\nafter resume"
        assert tool_runs == 1
        assert resumed_llm.seen_last_message is not None
        assert resumed_llm.seen_last_message.tool_results is not None
        assert "side_effect_call" in resumed_llm.seen_last_message.tool_results
        assert resumed_llm.seen_last_message.all_text() == "after resume"
        resumed_content = resumed_llm.seen_last_message.tool_results["side_effect_call"].content[0]
        assert isinstance(resumed_content, TextContent)
        assert resumed_content.text == "ok 1"
    finally:
        update_global_settings(old_settings)
        reset_session_manager()
