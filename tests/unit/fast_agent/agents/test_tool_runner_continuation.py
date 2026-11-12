"""Provider-requested follow-ups share the normal tool-loop lifecycle and budget."""

import asyncio

import pytest
from mcp_types import CallToolRequest, CallToolRequestParams, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


class ContinuingLlm(PassthroughLLM):
    def __init__(self, stops: list[LlmStopReason], *, cancel_at: int | None = None):
        super().__init__()
        self.stops = stops
        self.cancel_at = cancel_at
        self.inputs: list[list[PromptMessageExtended]] = []

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self.inputs.append(list(multipart_messages))
        if len(self.inputs) == self.cancel_at:
            raise asyncio.CancelledError
        stop = self.stops[min(len(self.inputs) - 1, len(self.stops) - 1)]
        return PromptMessageExtended(
            role="assistant",
            content=[text_content(f"step-{len(self.inputs)}")],
            stop_reason=stop,
            tool_calls=(
                {
                    "call_test": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="test_tool", arguments={}),
                    )
                }
                if stop == LlmStopReason.TOOL_USE
                else None
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_history", [True, False])
@pytest.mark.parametrize("budget", [0, 2])
async def test_endless_continuation_is_bounded(use_history: bool, budget: int) -> None:
    llm = ContinuingLlm([LlmStopReason.CONTINUE])
    agent = ToolAgent(AgentConfig("continuing", use_history=use_history), [])
    agent._llm = llm

    result = await agent.generate(
        "go", RequestParams(max_iterations=budget, use_history=use_history)
    )

    assert result.stop_reason == LlmStopReason.MAX_ITERATIONS
    assert len(llm.inputs) == budget + 1
    for index, messages in enumerate(llm.inputs):
        assert [m.role for m in messages] == ["user"] + ["assistant"] * index
        assert not any(m.tool_calls or m.tool_results for m in messages)
    if use_history:
        assert len(agent.message_history) == budget + 2
        assert agent.message_history[-1].stop_reason == LlmStopReason.MAX_ITERATIONS
    else:
        assert not agent.message_history


@pytest.mark.asyncio
@pytest.mark.parametrize("use_history", [True, False])
async def test_continuations_and_tools_share_one_budget(use_history: bool) -> None:
    executed: list[str] = []

    def test_tool() -> str:
        executed.append("tool")
        return "result"

    llm = ContinuingLlm(
        [
            LlmStopReason.CONTINUE,
            LlmStopReason.TOOL_USE,
            LlmStopReason.CONTINUE,
            LlmStopReason.END_TURN,
        ]
    )
    agent = ToolAgent(AgentConfig("mixed", use_history=use_history), [test_tool])
    agent._llm = llm

    result = await agent.generate("go", RequestParams(max_iterations=2, use_history=use_history))

    assert result.stop_reason == LlmStopReason.MAX_ITERATIONS
    assert len(llm.inputs) == 3
    assert executed == ["tool"]
    final_input = llm.inputs[-1]
    assert [m.role for m in final_input] == ["user", "assistant", "assistant", "user"]
    assert final_input[-1].tool_results
    assert list(final_input[-1].tool_results) == ["call_test"]


@pytest.mark.asyncio
async def test_cancellation_during_continuation_preserves_checkpoint() -> None:
    llm = ContinuingLlm([LlmStopReason.CONTINUE], cancel_at=2)
    agent = ToolAgent(AgentConfig("cancel-continuation"), [])
    agent._llm = llm

    with pytest.raises(asyncio.CancelledError):
        await agent.generate("go", RequestParams(max_iterations=2))

    assert len(llm.inputs) == 2
    assert [m.role for m in agent.message_history] == ["user", "assistant"]
    assert agent.message_history[-1].stop_reason == LlmStopReason.CONTINUE
    assert not any(m.tool_calls or m.tool_results for m in agent.message_history)


@pytest.mark.asyncio
async def test_continuation_does_not_fire_tool_hooks_or_finalize_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def before_llm(*_args):
        events.append("llm")

    async def tool_hook(*_args):
        events.append("tool")

    async def complete(*_args):
        events.append("complete")

    llm = ContinuingLlm([LlmStopReason.CONTINUE, LlmStopReason.END_TURN])
    agent = ToolAgent(AgentConfig("hooks"), [])
    agent._llm = llm
    monkeypatch.setattr(
        agent,
        "_tool_runner_hooks",
        lambda: ToolRunnerHooks(
            before_llm_call=before_llm,
            before_tool_call=tool_hook,
            after_tool_call=tool_hook,
            after_turn_complete=complete,
        ),
    )

    result = await agent.generate("go")

    assert result.stop_reason == LlmStopReason.END_TURN
    assert events == ["llm", "llm", "complete"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal",
    [
        LlmStopReason.SAFETY,
        LlmStopReason.MAX_TOKENS,
        LlmStopReason.ERROR,
        LlmStopReason.TIMEOUT,
        LlmStopReason.CANCELLED,
    ],
)
async def test_continuation_does_not_override_later_terminal_outcomes(
    terminal: LlmStopReason,
) -> None:
    llm = ContinuingLlm([LlmStopReason.CONTINUE, terminal])
    agent = ToolAgent(AgentConfig("terminal"), [])
    agent._llm = llm

    result = await agent.generate("go", RequestParams(max_iterations=5))

    assert result.stop_reason == terminal
    assert len(llm.inputs) == 2
