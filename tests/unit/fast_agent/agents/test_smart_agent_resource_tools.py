from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

import fast_agent.agents.smart_agent as smart_agent
from fast_agent.agents.smart_agent import _enable_smart_tooling
from fast_agent.core.exceptions import AgentConfigError


class _SmartToolHarness:
    def __init__(self) -> None:
        self.tools: list[Any] = []
        self._smart_tool_names: set[str] = set()
        self._parallel_smart_tool_calls = False

    @property
    def smart_tool_names(self) -> set[str]:
        return self._smart_tool_names

    @smart_tool_names.setter
    def smart_tool_names(self, value) -> None:
        self._smart_tool_names = set(value)

    @property
    def parallel_smart_tool_calls(self) -> bool:
        return self._parallel_smart_tool_calls

    @parallel_smart_tool_calls.setter
    def parallel_smart_tool_calls(self, value: bool) -> None:
        self._parallel_smart_tool_calls = value

    def add_tool(self, tool: object, *, replace: bool = True) -> None:
        del replace
        self.tools.append(tool)

    async def smart(
        self,
        agent_card_path: str,
        message: str | None = None,
        mcp_connect: list[str] | None = None,
        action: str = "run",
    ) -> str:
        del agent_card_path, message, mcp_connect, action
        return ""

    async def slash_command(self, command: str) -> str:
        del command
        return ""

    async def read_resource(self, uri: str, server_name: str | None = None) -> str:
        del uri, server_name
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

    _enable_smart_tooling(cast("Any", harness))

    names = {tool.name for tool in harness.tools}
    assert "smart" in names
    assert "slash_command" in names
    assert "get_resource" in names
    assert "validate" not in names
    assert "create_agent_card" not in names
    assert "mcp_connect" not in names
    assert "list_resources" not in names
    assert "attach_resource" not in names

    smart_tool = next(tool for tool in harness.tools if tool.name == "smart")
    smart_description = str(smart_tool.description or "")
    assert "action=`validate`" in smart_description

    slash_tool = next(tool for tool in harness.tools if tool.name == "slash_command")
    description = str(slash_tool.description or "")
    assert "/skills" in description
    assert "/cards" in description
    assert "/model" in description


@pytest.mark.asyncio
async def test_enable_smart_tooling_tools_default_to_content_only() -> None:
    harness = _SmartToolHarness()

    _enable_smart_tooling(cast("Any", harness))

    tools: dict[str, Any] = {tool.name: tool for tool in harness.tools}

    smart_result = await tools["smart"].run({"agent_card_path": "worker.md", "message": "hi"})
    slash_result = await tools["slash_command"].run({"command": "/commands"})
    resource_result = await tools["get_resource"].run(
        {"uri": "internal://fast-agent/smart-agent-cards", "server_name": None}
    )

    assert smart_result.structured_content is None
    assert slash_result.structured_content is None
    assert resource_result.structured_content is None


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
