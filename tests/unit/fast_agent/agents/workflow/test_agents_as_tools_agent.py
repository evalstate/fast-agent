import asyncio
from unittest.mock import AsyncMock

import pytest
from mcp import CallToolRequest, ListToolsResult, Tool
from mcp.types import CallToolRequestParams

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.workflow.agents_as_tools_agent import (
    AgentsAsToolsAgent,
    AgentsAsToolsOptions,
)
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import PromptMessageExtended
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL


class FakeChildAgent(LlmAgent):
    """Minimal child agent stub for Agents-as-Tools tests."""

    def __init__(self, name: str, response_text: str = "ok", delay: float = 0):
        super().__init__(AgentConfig(name))
        self._response_text = response_text
        self._delay = delay

    async def generate(self, messages, request_params=None):
        if self._delay:
            await asyncio.sleep(self._delay)
        return PromptMessageExtended(
            role="assistant",
            content=[text_content(f"{self._response_text}")],
        )

    async def spawn_detached_instance(self, name: str | None = None):
        # Mutate name for instance labelling; reuse self to keep the stub small.
        self._name = name or self.name
        return self


class ErrorChannelChild(FakeChildAgent):
    async def generate(self, messages, request_params=None):
        return PromptMessageExtended(
            role="assistant",
            content=[],
            channels={FAST_AGENT_ERROR_CHANNEL: [text_content("err-block")]},
        )


class StubNestedAgentsAsTools(AgentsAsToolsAgent):
    """Stub AgentsAsToolsAgent that responds without hitting an LLM."""

    async def generate(self, messages, request_params=None):
        return PromptMessageExtended(
            role="assistant",
            content=[text_content(f"{self.name}-reply")],
        )

    async def spawn_detached_instance(self, name: str | None = None):
        self._name = name or self.name
        return self


@pytest.mark.asyncio
async def test_list_tools_merges_base_and_child():
    child = FakeChildAgent("child")
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    # Inject a base MCP tool via the filtered MCP path to ensure merge behavior.
    base_tool = Tool(name="base_tool", description="base", inputSchema={"type": "object"})
    agent._get_filtered_mcp_tools = AsyncMock(return_value=[base_tool])

    result = await agent.list_tools()
    tool_names = {t.name for t in result.tools}

    assert "base_tool" in tool_names
    assert "agent__child" in tool_names


@pytest.mark.asyncio
async def test_run_tools_respects_max_parallel_and_timeout():
    fast_child = FakeChildAgent("fast", response_text="fast")
    slow_child = FakeChildAgent("slow", response_text="slow", delay=0.05)

    options = AgentsAsToolsOptions(max_parallel=1, child_timeout_sec=0.01)
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [fast_child, slow_child], options=options)
    await agent.initialize()

    tool_calls = {
        "1": CallToolRequest(params=CallToolRequestParams(name="agent__fast", arguments={"text": "hi"})),
        "2": CallToolRequest(params=CallToolRequestParams(name="agent__slow", arguments={"text": "hi"})),
    }
    request = PromptMessageExtended(role="assistant", content=[], tool_calls=tool_calls)

    result_message = await agent.run_tools(request)
    assert result_message.tool_results

    fast_result = result_message.tool_results["1"]
    slow_result = result_message.tool_results["2"]

    assert not fast_result.isError
    # Skipped due to max_parallel cap.
    assert slow_result.isError
    assert "Skipped" in slow_result.content[0].text

    # Now ensure timeout path yields an error result when a single slow call runs.
    request_single = PromptMessageExtended(
        role="assistant",
        content=[],
        tool_calls={"3": CallToolRequest(params=CallToolRequestParams(name="agent__slow", arguments={"text": "hi"}))},
    )
    single_result = await agent.run_tools(request_single)
    err_res = single_result.tool_results["3"]
    assert err_res.isError
    assert any("Tool execution failed" in (block.text or "") for block in err_res.content)


@pytest.mark.asyncio
async def test_invoke_child_appends_error_channel():
    child = ErrorChannelChild("err-child")
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    call_result = await agent._invoke_child_agent(child, {"text": "hi"})

    assert call_result.isError
    texts = [block.text for block in call_result.content if hasattr(block, "text")]
    assert "err-block" in texts


@pytest.mark.asyncio
async def test_nested_agents_as_tools_preserves_instance_labels():
    leaf = FakeChildAgent("leaf", response_text="leaf-ok")
    nested = StubNestedAgentsAsTools(AgentConfig("nested"), [leaf])
    parent = AgentsAsToolsAgent(AgentConfig("parent"), [nested])

    await nested.initialize()
    await parent.initialize()

    tool_calls = {
        "1": CallToolRequest(params=CallToolRequestParams(name="agent__nested", arguments={"text": "hi"})),
    }
    request = PromptMessageExtended(role="assistant", content=[], tool_calls=tool_calls)

    result_message = await parent.run_tools(request)
    result = result_message.tool_results["1"]
    assert not result.isError
    # Reply should include the instance-suffixed nested agent name.
    assert any("nested[1]-reply" in (block.text or "") for block in result.content)
