from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import fast_agent.agents.smart_agent as smart_agent
from fast_agent.agents.smart_agent import _enable_smart_tooling
from fast_agent.core.exceptions import AgentConfigError


class _SmartToolHarness:
    def __init__(self) -> None:
        self.tools: list[object] = []

    def add_tool(self, tool: object) -> None:
        self.tools.append(tool)

    async def smart(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def slash_command(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def read_resource(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def smart_list_resources(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def smart_get_resource(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def smart_with_resource(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def smart_complete_resource_argument(self, *args, **kwargs):
        del args, kwargs
        return ""


def test_enable_smart_tooling_registers_minimal_visible_tools() -> None:
    harness = _SmartToolHarness()

    _enable_smart_tooling(harness)

    names = {getattr(tool, "name", "") for tool in harness.tools}
    assert "smart" in names
    assert "slash_command" in names
    assert "get_resource" in names
    assert "validate" not in names
    assert "create_agent_card" not in names
    assert "mcp_connect" not in names
    assert "list_resources" not in names
    assert "attach_resource" not in names

    smart_tool = next(tool for tool in harness.tools if getattr(tool, "name", "") == "smart")
    smart_description = str(getattr(smart_tool, "description", ""))
    assert "action=`validate`" in smart_description

    slash_tool = next(tool for tool in harness.tools if getattr(tool, "name", "") == "slash_command")
    description = str(getattr(slash_tool, "description", ""))
    assert "/skills" in description
    assert "/cards" in description
    assert "/model" in description


@pytest.mark.asyncio
async def test_dispatch_smart_tool_validate_action_uses_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate = AsyncMock(return_value="validated")
    run = AsyncMock(return_value="ran")
    monkeypatch.setattr(smart_agent, "_run_validate_call", validate)
    monkeypatch.setattr(smart_agent, "_run_smart_call", run)

    agent = type("AgentStub", (), {"context": object()})()

    result = await smart_agent._dispatch_smart_tool(
        agent=agent,
        agent_card_path="worker.md",
        action="validate",
    )

    assert result == "validated"
    validate.assert_awaited_once_with(agent.context, "worker.md")
    run.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_smart_tool_run_requires_message() -> None:
    with pytest.raises(AgentConfigError, match="Provide `message` when action=`run`"):
        await smart_agent._dispatch_smart_tool(
            agent=object(),
            agent_card_path="worker.md",
        )


@pytest.mark.asyncio
async def test_dispatch_smart_get_resource_routes_internal_uris(monkeypatch: pytest.MonkeyPatch) -> None:
    internal_read = AsyncMock(return_value="internal result")
    smart_read = AsyncMock(return_value="smart result")
    monkeypatch.setattr(smart_agent, "_run_internal_resource_read_call", internal_read)
    monkeypatch.setattr(smart_agent, "_run_smart_get_resource_call", smart_read)

    result = await smart_agent._dispatch_smart_get_resource_tool(
        agent=object(),
        agent_card_path="worker.md",
        resource_uri="internal://fast-agent/smart-agent-cards",
        server_name="external",
        mcp_connect=["uv run foo.py"],
    )

    assert result == "internal result"
    internal_read.assert_awaited_once_with("internal://fast-agent/smart-agent-cards")
    smart_read.assert_not_awaited()
