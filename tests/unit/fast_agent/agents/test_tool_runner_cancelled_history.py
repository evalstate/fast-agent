import asyncio

import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


async def cancel_tool() -> None:
    raise asyncio.CancelledError("cancelled by test")


async def ok_tool() -> str:
    return "ok"


class CancelledToolUseLlm(PassthroughLLM):
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
                "cancel_call": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="cancel_tool", arguments={}),
                )
            }
            return Prompt.assistant(
                "use tool",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls=tool_calls,
            )

        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


class CancelledStopReasonLlm(PassthroughLLM):
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
                "cancel_call": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="ok_tool", arguments={}),
                )
            }
            return Prompt.assistant(
                "use tool",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls=tool_calls,
            )

        return Prompt.assistant("", stop_reason=LlmStopReason.CANCELLED)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_tool_loop_resets_history() -> None:
    llm = CancelledToolUseLlm()
    agent = ToolAgent(AgentConfig("cancelled"), [cancel_tool])
    agent._llm = llm

    agent.load_message_history(
        [
            Prompt.user("previous"),
            Prompt.assistant("ok", stop_reason=LlmStopReason.END_TURN),
        ]
    )

    with pytest.raises(asyncio.CancelledError):
        await agent.generate("trigger")

    history = agent.message_history
    assert len(history) == 2
    assert history[-1].last_text() == "ok"
    assert all(
        msg.stop_reason != LlmStopReason.TOOL_USE
        for msg in history
        if msg.role == "assistant"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_stop_reason_rolls_back_history_and_skips_after_turn_hooks() -> None:
    llm = CancelledStopReasonLlm()
    agent = ToolAgent(AgentConfig("cancelled-stop"), [ok_tool])
    agent._llm = llm

    hook_called = False

    async def after_turn_complete(_runner, _message) -> None:
        nonlocal hook_called
        hook_called = True

    agent.tool_runner_hooks = ToolRunnerHooks(after_turn_complete=after_turn_complete)

    agent.load_message_history(
        [
            Prompt.user("previous"),
            Prompt.assistant("ok", stop_reason=LlmStopReason.END_TURN),
        ]
    )

    result = await agent.generate("trigger")

    assert result.stop_reason == LlmStopReason.CANCELLED
    assert hook_called is False

    history = agent.message_history
    assert len(history) == 2
    assert history[-1].last_text() == "ok"

    rollback_state = getattr(agent, "_last_turn_history_state", None)
    assert rollback_state is not None
    assert getattr(rollback_state, "status", None) == "rolled_back_to_assistant"
    assert getattr(rollback_state, "removed_messages", None) == 4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_tool_loop_with_use_history_false_keeps_history_unchanged() -> None:
    llm = CancelledStopReasonLlm()
    agent = ToolAgent(AgentConfig("cancelled-no-history", use_history=False), [ok_tool])
    agent._llm = llm

    baseline_history = [
        Prompt.user("seed"),
        Prompt.assistant("seed-response", stop_reason=LlmStopReason.END_TURN),
    ]
    agent.load_message_history(baseline_history)

    result = await agent.generate("trigger", RequestParams(use_history=False))

    assert result.stop_reason == LlmStopReason.CANCELLED
    assert len(agent.message_history) == len(baseline_history)
    for index, message in enumerate(agent.message_history):
        assert message.role == baseline_history[index].role
        assert message.last_text() == baseline_history[index].last_text()

    rollback_state = getattr(agent, "_last_turn_history_state", None)
    assert rollback_state is not None
    assert getattr(rollback_state, "status", None) == "history_disabled"
    assert getattr(rollback_state, "removed_messages", None) == 0
