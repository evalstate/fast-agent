import pytest
from mcp_types import CallToolRequest, CallToolRequestParams, Tool

from fast_agent.acp.server.common import map_llm_stop_reason_to_acp
from fast_agent.agents import tool_runner
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core.direct_factory import _apply_trim_history_hook
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


def looping_tool() -> str:
    return "tool-result"


class AlwaysToolCallingLlm(PassthroughLLM):
    """An LLM that never finishes - it requests the same tool on every turn."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_count = 0
        self.inputs: list[list[PromptMessageExtended]] = []

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self.call_count += 1
        self.inputs.append(list(multipart_messages))
        return PromptMessageExtended(
            role="assistant",
            content=[text_content("calling again")],
            stop_reason=LlmStopReason.TOOL_USE,
            tool_calls={
                f"call_{self.call_count}": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="looping_tool", arguments={}),
                )
            },
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exhausting_max_iterations_reports_a_stop_reason() -> None:
    llm = AlwaysToolCallingLlm()
    agent = ToolAgent(AgentConfig("looping"), [looping_tool])
    agent._llm = llm

    result = await agent.generate("go", RequestParams(max_iterations=3))

    assert result.stop_reason == LlmStopReason.MAX_ITERATIONS
    # The budget is spent before the loop notices it is exhausted.
    assert llm.call_count == 4


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("trim_tool_history", [False, True])
@pytest.mark.parametrize("use_history", [False, True])
async def test_iteration_limit_preserves_completed_tool_results(
    trim_tool_history: bool, use_history: bool
) -> None:
    llm = AlwaysToolCallingLlm()
    config = AgentConfig("looping", trim_tool_history=trim_tool_history, use_history=use_history)
    agent = ToolAgent(config, [looping_tool])
    agent._llm = llm
    _apply_trim_history_hook(agent, config)

    result = await agent.generate("go", RequestParams(max_iterations=2, use_history=use_history))

    assert result.stop_reason == LlmStopReason.MAX_ITERATIONS
    history = list(agent.message_history)
    if use_history:
        assert result.tool_calls
        assert history[-1].tool_results
        assert history[-1].tool_results.keys() == result.tool_calls.keys()
        for request, response in zip(history[1::2], history[2::2], strict=True):
            assert request.tool_calls
            assert response.tool_results
            assert request.tool_calls.keys() == response.tool_results.keys()
            assert all(
                tool_result.content == [text_content("tool-result")]
                for tool_result in response.tool_results.values()
            )
    else:
        assert history == []

    await agent.generate("continue", RequestParams(max_iterations=0, use_history=use_history))

    assert llm.call_count == 4
    requests = [key for message in llm.inputs[-1] for key in (message.tool_calls or {})]
    results = [key for message in llm.inputs[-1] for key in (message.tool_results or {})]
    assert len(requests) == (3 if use_history else 0)
    assert requests == results
    assert llm.inputs[-1][-1].last_text() == "continue"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exhausting_max_iterations_logs_a_warning(monkeypatch) -> None:
    warnings: list[tuple[str, dict]] = []

    def record_warning(message: str, data: dict | None = None, **_kwargs) -> None:
        warnings.append((message, data or {}))

    monkeypatch.setattr(tool_runner._logger, "warning", record_warning)

    llm = AlwaysToolCallingLlm()
    agent = ToolAgent(AgentConfig("looping"), [looping_tool])
    agent._llm = llm

    await agent.generate("go", RequestParams(max_iterations=2))

    matching = [data for message, data in warnings if "maximum iterations reached" in message]
    assert matching == [{"agent_name": "looping", "iterations": 3, "max_iterations": 2}]


@pytest.mark.unit
def test_max_iterations_maps_to_an_acp_stop_reason() -> None:
    assert map_llm_stop_reason_to_acp(LlmStopReason.MAX_ITERATIONS) == "max_turn_requests"
