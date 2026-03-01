from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import fast_agent.agents.smart_agent as smart_agent
from fast_agent.agents.smart_agent import _enable_smart_tooling


class _SmartToolHarness:
    def __init__(self) -> None:
        self.tools: list[object] = []

    def add_tool(self, tool: object) -> None:
        self.tools.append(tool)

    async def smart(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def validate(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def create_agent_card(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def slash_command(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def mcp_connect(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def resource_list(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def resource_read(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def attach_resource(self, *args, **kwargs):
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


def test_enable_smart_tooling_registers_resource_tools() -> None:
    harness = _SmartToolHarness()

    _enable_smart_tooling(harness)

    names = {getattr(tool, "name", "") for tool in harness.tools}
    assert "smart" in names
    assert "slash_command" in names
    assert "validate" in names
    assert "create_agent_card" in names
    assert "mcp_connect" in names
    assert "list_resources" in names
    assert "get_resource" in names
    assert "attach_resource" in names

    slash_tool = next(tool for tool in harness.tools if getattr(tool, "name", "") == "slash_command")
    description = str(getattr(slash_tool, "description", ""))
    assert "/skills" in description
    assert "/cards" in description
    assert "/models" in description


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
