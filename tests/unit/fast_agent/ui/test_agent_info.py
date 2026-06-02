from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from mcp.types import ListToolsResult, Tool

from fast_agent.ui.prompt import agent_info
from fast_agent.ui.prompt.agent_info import (
    AgentSurfaceCounts,
    _child_agent_counts,
    _mcp_server_resource_counts,
)

if TYPE_CHECKING:
    from fast_agent.mcp.types import McpAgentProtocol


class _AggregatorStub:
    server_names = ["hf"]

    async def list_tools(self) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                Tool(name="hf.tool_1", inputSchema={}),
                Tool(name="hf.tool_2", inputSchema={}),
            ]
        )

    async def list_resources(self) -> dict[str, list[str]]:
        return {"hf": ["skill://index.json", "skill://datasets/SKILL.md"]}

    async def list_prompts(self):
        return {"hf": [object()]}


class _McpChildAgentStub:
    name = "child"
    aggregator = _AggregatorStub()


@pytest.mark.asyncio
async def test_mcp_server_resource_counts_use_aggregator_only() -> None:
    agent = cast(
        "McpAgentProtocol",
        SimpleNamespace(
            aggregator=_AggregatorStub(),
        ),
    )

    assert await _mcp_server_resource_counts(agent) == AgentSurfaceCounts(
        tool_count=2,
        prompt_count=1,
        resource_count=2,
    )


@pytest.mark.asyncio
async def test_child_agent_counts_keep_prompt_and_resource_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_info, "McpAgentProtocol", _McpChildAgentStub)

    counts = await _child_agent_counts(cast("McpAgentProtocol", _McpChildAgentStub()))

    assert counts == AgentSurfaceCounts(
        server_count=1,
        tool_count=2,
        prompt_count=1,
        resource_count=2,
    )
