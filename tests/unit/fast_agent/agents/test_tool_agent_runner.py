import asyncio
from typing import Any, Callable

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.interfaces import FastAgentLLMProtocol
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import UsageAccumulator
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason


class FakeLLM(FastAgentLLMProtocol):
    def __init__(self, responses: list[PromptMessageExtended]):
        self._responses = responses
        self.calls: list[list[PromptMessageExtended]] = []
        self._usage = UsageAccumulator()

    async def generate(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        self.calls.append(messages)
        return self._responses.pop(0)

    async def structured(
        self,
        messages: list[PromptMessageExtended],
        model: Any,
        request_params: RequestParams | None = None,
    ):
        raise NotImplementedError

    async def apply_prompt_template(self, prompt_result, prompt_name: str) -> str:
        return ""

    def get_request_params(self, request_params: RequestParams | None = None) -> RequestParams:
        return request_params or RequestParams()

    def add_stream_listener(self, listener: Callable) -> Callable[[], None]:
        return lambda: None

    def add_tool_stream_listener(self, listener: Callable) -> Callable[[], None]:
        return lambda: None

    @property
    def message_history(self) -> list[PromptMessageExtended]:
        return []

    def pop_last_message(self) -> PromptMessageExtended | None:
        return None

    @property
    def usage_accumulator(self) -> UsageAccumulator | None:
        return self._usage

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def model_name(self) -> str | None:
        return "fake-model"

    @property
    def model_info(self):
        return None

    def clear(self, *, clear_prompts: bool = False) -> None:
        return None


class DummyToolAgent(ToolAgent):
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        text = f"ran:{name}:{arguments or {}}"
        return CallToolResult(content=[], isError=False, structuredContent={"text": text})


def _assistant_tool_call(tool_name: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        tool_calls={
            "tool-1": CallToolRequest(
                params=CallToolRequestParams(name=tool_name, arguments={"x": 1}),
            )
        },
        stop_reason=LlmStopReason.TOOL_USE,
    )


def _assistant_text() -> PromptMessageExtended:
    return PromptMessageExtended(role="assistant", content=[], stop_reason=LlmStopReason.END_TURN)


@pytest.mark.asyncio
async def test_tool_agent_uses_runner_and_parallel_tools():
    # Fake LLM: first call requests a tool, second ends the turn
    fake_llm = FakeLLM(responses=[_assistant_tool_call("do_work"), _assistant_text()])

    agent = DummyToolAgent(config=AgentConfig(name="test-agent"))
    agent._llm = fake_llm  # inject fake LLM

    tools = [Tool(name="do_work", description="", inputSchema={"type": "object"})]

    final = await agent.generate([PromptMessageExtended(role="user", content=[])], tools=tools)

    # Runner path should have executed the tool and reached END_TURN
    assert final.stop_reason == LlmStopReason.END_TURN
    assert len(fake_llm.calls) == 2

    # Ensure the tool result was produced by our overridden executor
    # and not by fallback: the structuredContent marker is present.
    first_turn_user = fake_llm.calls[0][0]
    assert first_turn_user.role == "user"

    # Second turn should carry tool_results back to the model
    second_turn_message = fake_llm.calls[1][0]
    assert second_turn_message.tool_results is not None
    assert "tool-1" in second_turn_message.tool_results
