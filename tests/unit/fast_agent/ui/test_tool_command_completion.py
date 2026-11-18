"""Singular ``/tool`` completes tool names only; ``/tools`` also completes its actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mcp_types import ListToolsResult
from prompt_toolkit.document import Document

from fast_agent.core.agent_app import AgentApp
from fast_agent.ui.prompt.completer import AgentCompleter
from tests.unit.fast_agent.ui.test_tool_mentions import _ToolAgent

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _ListingToolAgent(_ToolAgent):
    async def list_tools(self) -> ListToolsResult:
        surface = await self.tool_surface()
        return ListToolsResult(tools=[surface_tool.tool for surface_tool in surface.tools])


def _completions(text: str) -> list[str]:
    completer = AgentCompleter(
        agents=["main"],
        current_agent="main",
        agent_provider=AgentApp({"main": cast("AgentProtocol", _ListingToolAgent())}),
    )
    return [entry.text for entry in completer.get_completions(Document(text), None)]


def test_singular_tool_completes_tool_names_without_actions() -> None:
    names = _completions("/tool ")

    assert "jobs-list" in names
    assert not {"--all", "add", "select"} & set(names)
    assert _completions("/tool add ") == []


def test_plural_tools_completes_actions_and_add_arguments() -> None:
    assert _completions("/tools ")[:3] == ["--all", "add", "select"]
    assert _completions("/tools add ") == ["jobs_list", "jobs_status"]
