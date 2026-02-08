import asyncio

import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


async def cancel_tool() -> None:
    raise asyncio.CancelledError("cancelled by test")


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
