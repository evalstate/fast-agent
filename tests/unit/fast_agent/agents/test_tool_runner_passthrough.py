import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.constants import FAST_AGENT_SYNTHETIC_FINAL_CHANNEL, FAST_AGENT_USAGE
from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


def passthrough_tool() -> str:
    return "raw-result"


class ToolThenFinalizeLlm(PassthroughLLM):
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

        if self.call_count == 1:
            return PromptMessageExtended(
                role="assistant",
                content=[text_content("use tool")],
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "call_1": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="passthrough_tool", arguments={}),
                    )
                },
                channels={FAST_AGENT_USAGE: [text_content('{"token_count": 12}')]},
            )

        return Prompt.assistant("postprocessed", stop_reason=LlmStopReason.END_TURN)


class HookCapturePassthroughAgent(ToolAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.after_llm_calls: list[tuple[LlmStopReason | None, bool]] = []
        self.after_turn_calls: list[tuple[LlmStopReason | None, bool]] = []

    def _tool_runner_hooks(self) -> ToolRunnerHooks | None:
        async def after_llm_call(_runner, message):
            has_marker = bool((message.channels or {}).get(FAST_AGENT_SYNTHETIC_FINAL_CHANNEL))
            self.after_llm_calls.append((message.stop_reason, has_marker))

        async def after_turn_complete(_runner, message):
            has_marker = bool((message.channels or {}).get(FAST_AGENT_SYNTHETIC_FINAL_CHANNEL))
            self.after_turn_calls.append((message.stop_reason, has_marker))

        return ToolRunnerHooks(
            after_llm_call=after_llm_call,
            after_turn_complete=after_turn_complete,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_passthrough_skips_second_llm_call_and_synthesizes_terminal_assistant() -> None:
    llm = ToolThenFinalizeLlm()
    agent = ToolAgent(AgentConfig("passthrough"), [passthrough_tool])
    agent._llm = llm

    result = await agent.generate("hi", RequestParams(tool_result_passthrough=True))

    assert llm.call_count == 1
    assert result.role == "assistant"
    assert result.stop_reason == LlmStopReason.END_TURN
    assert result.last_text() == "raw-result"
    assert FAST_AGENT_SYNTHETIC_FINAL_CHANNEL in (result.channels or {})
    assert (result.channels or {}).get(FAST_AGENT_USAGE)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_passthrough_use_history_true_appends_tool_result_and_synthesized_assistant() -> None:
    llm = ToolThenFinalizeLlm()
    agent = ToolAgent(AgentConfig("passthrough"), [passthrough_tool])
    agent._llm = llm

    await agent.generate("hi", RequestParams(tool_result_passthrough=True))

    history = agent.message_history
    assert len(history) == 4
    assert history[0].role == "user"
    assert history[1].role == "assistant"
    assert history[1].stop_reason == LlmStopReason.TOOL_USE
    assert history[2].role == "user"
    assert history[2].tool_results is not None
    assert history[3].role == "assistant"
    assert history[3].stop_reason == LlmStopReason.END_TURN
    assert FAST_AGENT_SYNTHETIC_FINAL_CHANNEL in (history[3].channels or {})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_passthrough_use_history_false_does_not_persist_turn_messages() -> None:
    llm = ToolThenFinalizeLlm()
    agent = ToolAgent(AgentConfig("passthrough-no-history", use_history=False), [passthrough_tool])
    agent._llm = llm

    seed_history = [
        Prompt.user("seed"),
        Prompt.assistant("seed-response", stop_reason=LlmStopReason.END_TURN),
    ]
    agent.load_message_history(seed_history)

    result = await agent.generate(
        "hi",
        RequestParams(
            use_history=False,
            tool_result_passthrough=True,
        ),
    )

    assert llm.call_count == 1
    assert result.last_text() == "raw-result"
    assert len(agent.message_history) == len(seed_history)
    assert [message.role for message in agent.message_history] == ["user", "assistant"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_passthrough_fires_after_llm_and_after_turn_hooks_with_synthetic_marker() -> None:
    llm = ToolThenFinalizeLlm()
    agent = HookCapturePassthroughAgent(AgentConfig("hooked-passthrough"), [passthrough_tool])
    agent._llm = llm

    await agent.generate("hi", RequestParams(tool_result_passthrough=True))

    assert agent.after_llm_calls == [
        (LlmStopReason.TOOL_USE, False),
        (LlmStopReason.END_TURN, True),
    ]
    assert agent.after_turn_calls == [(LlmStopReason.END_TURN, True)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_postprocess_mode_remains_default_behavior() -> None:
    llm = ToolThenFinalizeLlm()
    agent = ToolAgent(AgentConfig("postprocess"), [passthrough_tool])
    agent._llm = llm

    result = await agent.generate("hi")

    assert llm.call_count == 2
    assert result.stop_reason == LlmStopReason.END_TURN
    assert result.last_text() == "postprocessed"
