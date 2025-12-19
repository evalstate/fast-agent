import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


def tool_one() -> int:
    return 1


def tool_two() -> int:
    return 2


class TwoStepToolUseLlm(PassthroughLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[list[str]] = []
        self._turn = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self._turn += 1
        self.calls.append(
            [
                get_text(block) or ""
                for msg in multipart_messages
                for block in (msg.content or [])
                if get_text(block)
            ]
        )

        if self._turn == 1:
            tool_calls = {
                "id_one": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="tool_one", arguments={}),
                ),
                "id_two": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="tool_two", arguments={}),
                ),
            }
            return Prompt.assistant(
                "use tools",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls=tool_calls,
            )

        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


class HookedToolAgent(ToolAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[str] = []
        self._injected = False

    def _tool_runner_hooks(self) -> ToolRunnerHooks | None:
        async def before_llm_call(runner, messages):
            self.events.append(f"before_llm_call:{runner.iteration}")
            if not self._injected:
                runner.append_messages("extra from hook")
                self._injected = True

        async def after_llm_call(runner, message):
            self.events.append(f"after_llm_call:{message.stop_reason}")

        async def before_tool_call(runner, message):
            self.events.append(f"before_tool_call:{len(message.tool_calls or {})}")

        async def after_tool_call(runner, message):
            self.events.append(f"after_tool_call:{len(message.tool_results or {})}")

        return ToolRunnerHooks(
            before_llm_call=before_llm_call,
            after_llm_call=after_llm_call,
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_runner_hooks_fire_and_can_inject_messages():
    llm = TwoStepToolUseLlm()
    agent = HookedToolAgent(AgentConfig("hooked"), [tool_one, tool_two])
    agent._llm = llm

    result = await agent.generate("hi")
    assert result.last_text() == "done"

    assert any("extra from hook" in entry for entry in llm.calls[0])

    assert agent.events == [
        "before_llm_call:0",
        f"after_llm_call:{LlmStopReason.TOOL_USE}",
        "before_tool_call:2",
        "after_tool_call:2",
        "before_llm_call:1",
        f"after_llm_call:{LlmStopReason.END_TURN}",
    ]


# Track tool invocations globally for the regression test
_tool_invocations: list[str] = []


def tracked_tool_a() -> str:
    _tool_invocations.append("tool_a")
    return "result_a"


def tracked_tool_b() -> str:
    _tool_invocations.append("tool_b")
    return "result_b"


class TwoRoundToolUseLlm(PassthroughLLM):
    """LLM that returns tool_use twice before completing."""

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
            # First round: call tool_a
            return Prompt.assistant(
                "calling tool_a",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "call_1": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="tracked_tool_a", arguments={}),
                    ),
                },
            )

        if self._turn == 2:
            # Second round: call tool_b
            return Prompt.assistant(
                "calling tool_b",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "call_2": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="tracked_tool_b", arguments={}),
                    ),
                },
            )

        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_two_tool_use_rounds_both_execute():
    """Regression test: ensure second tool-use round executes new tools, not cached response."""
    _tool_invocations.clear()

    llm = TwoRoundToolUseLlm()
    agent = ToolAgent(AgentConfig("test"), [tracked_tool_a, tracked_tool_b])
    agent._llm = llm

    result = await agent.generate("hi")
    assert result.last_text() == "done"

    # Both tools must have been called - if caching bug exists, only tool_a would be called
    assert _tool_invocations == ["tool_a", "tool_b"], (
        f"Expected both tools to execute, got: {_tool_invocations}"
    )

