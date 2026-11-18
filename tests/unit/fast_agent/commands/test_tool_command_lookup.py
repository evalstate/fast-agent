"""Singular ``/tool <name>`` is a plain lookup; ``/tools <action>`` keeps its actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.commands.handlers.tools import handle_list_tools
from fast_agent.core.fastagent import AgentInstance
from fast_agent.tools.tool_surface import SurfaceTool, ToolExposure, ToolSurface
from fast_agent.ui.command_payloads import ListToolsCommand
from fast_agent.ui.prompt.parser import parse_special_input
from tests.unit.fast_agent.commands.test_tools_search import (
    _AcpApp,
    _context,
    _rendered,
    _surface,
    _SurfaceAgent,
    _tool,
)

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol


def _action_named_surface() -> ToolSurface:
    base = _surface()
    action_tools = tuple(
        SurfaceTool(
            tool=_tool(name),
            exposure=ToolExposure.MODEL,
            context_tokens=10,
        )
        for name in ("add", "select")
    )
    return ToolSurface(tools=(*base.tools, *action_tools), programmatic=base.programmatic)


class _FailingSurfaceAgent(_SurfaceAgent):
    async def tool_surface(self) -> ToolSurface:
        raise RuntimeError("server unavailable")


def _acp_handler(agent: _SurfaceAgent) -> SlashCommandHandler:
    return SlashCommandHandler(
        session_id="s1",
        instance=AgentInstance(
            app=cast("AgentApp", _AcpApp(agent)),
            agents={"main": cast("AgentProtocol", agent)},
            registry_version=0,
        ),
        primary_agent_name="main",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["add", "select"])
async def test_tui_singular_tool_inspects_tools_named_like_actions(name: str) -> None:
    command = parse_special_input(f"/tool {name}")
    assert isinstance(command, ListToolsCommand)

    outcome = await handle_list_tools(
        _context(_action_named_surface()), agent_name="main", argument=command.argument
    )

    assert outcome.buffer_prefill is None
    assert f"# Tool schema: {name}" in _rendered(outcome)


@pytest.mark.asyncio
async def test_tui_plural_tools_keeps_actions() -> None:
    command = parse_special_input("/tools add create_issue")
    assert isinstance(command, ListToolsCommand)

    outcome = await handle_list_tools(
        _context(_action_named_surface()), agent_name="main", argument=command.argument
    )

    assert outcome.buffer_prefill == "^tool:create_issue"


@pytest.mark.asyncio
async def test_tui_singular_tool_reports_unknown_and_extra_names() -> None:
    for text, message in [
        ("/tool nope", "Tool not found: nope"),
        ("/tool add extra", "Usage: /tool <tool-name>"),
    ]:
        command = parse_special_input(text)
        assert isinstance(command, ListToolsCommand)
        outcome = await handle_list_tools(
            _context(_action_named_surface()), agent_name="main", argument=command.argument
        )
        assert message in _rendered(outcome)


@pytest.mark.asyncio
async def test_acp_singular_tool_inspects_tools_named_like_actions() -> None:
    handler = _acp_handler(_SurfaceAgent(_action_named_surface()))

    detail = await handler.execute_command("tool", "add")
    added = await handler.execute_command("tools", "add create_issue")

    assert "# Tool schema: add" in detail
    assert "draft text" not in detail
    assert "^tool:create_issue" in added


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["", "add create_issue", "select"])
async def test_acp_tools_reports_surface_failures(arguments: str) -> None:
    handler = _acp_handler(_FailingSurfaceAgent(_surface()))

    rendered = await handler.execute_command("tools", arguments)

    assert "Failed to fetch tools from the agent." in rendered
    assert "server unavailable" in rendered
