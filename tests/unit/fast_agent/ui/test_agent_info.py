import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from mcp.types import ListToolsResult, Tool

from fast_agent.interfaces import AgentProtocol
from fast_agent.ui.prompt import agent_info
from fast_agent.ui.prompt.agent_info import (
    AgentResourceCounts,
    ChildAgentCounts,
    _agent_info_text,
    _child_agent_counts_markup,
    _child_agent_info_text,
    _display_child_agent_info,
    _format_dim_count,
    _format_installed_skill_count,
    _format_server_summary,
    _mcp_resource_counts_for_agent,
    _resource_counts_for_agent,
    display_agent_info,
)


def test_format_dim_count_pluralizes_and_keeps_suffix_inside_dim_markup() -> None:
    assert (
        _format_dim_count(1, "tool", suffix=", ")
        == "[bold bright_cyan]1[/bold bright_cyan][dim] tool, [/dim]"
    )
    assert (
        _format_dim_count(2, "tool", suffix=" available")
        == "[bold bright_cyan]2[/bold bright_cyan][dim] tools available[/dim]"
    )
    assert (
        _format_dim_count(1_200, "tool")
        == "[bold bright_cyan]1,200[/bold bright_cyan][dim] tools[/dim]"
    )


def test_format_installed_skill_count_uses_clear_installed_wording() -> None:
    assert (
        _format_installed_skill_count(1)
        == "[bold bright_cyan]1[/bold bright_cyan][dim] skill installed[/dim]"
    )
    assert (
        _format_installed_skill_count(2)
        == "[bold bright_cyan]2[/bold bright_cyan][dim] skills installed[/dim]"
    )


def test_format_server_summary_uses_shared_count_formatting() -> None:
    assert (
        _format_server_summary(
            server_count=2,
            tool_count=1,
            prompt_count=3,
            resource_count=0,
        )
        == "[bold bright_cyan]2[/bold bright_cyan][dim] MCP Servers[/dim][dim] ([/dim][bold bright_cyan]1[/bold bright_cyan][dim] tool[/dim][dim], [/dim][bold bright_cyan]3[/bold bright_cyan][dim] prompts[/dim][dim])[/dim]"
    )


def test_format_server_summary_shows_unknown_failed_resource_counts() -> None:
    assert (
        _format_server_summary(
            server_count=1,
            tool_count=None,
            prompt_count=2,
            resource_count=None,
        )
        == "[bold bright_cyan]1[/bold bright_cyan][dim] MCP Server[/dim][dim] ([/dim][dim]unknown tools[/dim][dim], [/dim][bold bright_cyan]2[/bold bright_cyan][dim] prompts[/dim][dim], [/dim][dim]unknown resources[/dim][dim])[/dim]"
    )


def test_child_agent_counts_markup_omits_zero_resource_counts() -> None:
    markup = _child_agent_counts_markup(
        ChildAgentCounts(
            server_count=1,
            resource_counts=AgentResourceCounts(
                tool_count=0,
                prompt_count=0,
                resource_count=0,
            ),
        )
    )

    assert markup == "[bold bright_cyan]1[/bold bright_cyan][dim] MCP Server[/dim]"


def test_child_agent_counts_markup_includes_nonzero_resource_counts() -> None:
    markup = _child_agent_counts_markup(
        ChildAgentCounts(
            server_count=2,
            resource_counts=AgentResourceCounts(
                tool_count=1,
                prompt_count=3,
                resource_count=0,
            ),
        )
    )

    assert markup == (
        "[bold bright_cyan]2[/bold bright_cyan][dim] MCP Servers[/dim][dim], [/dim]"
        "[bold bright_cyan]1[/bold bright_cyan][dim] tool[/dim][dim], [/dim]"
        "[bold bright_cyan]3[/bold bright_cyan][dim] prompts[/dim]"
    )


def test_child_agent_counts_markup_shows_unknown_failed_resource_counts() -> None:
    markup = _child_agent_counts_markup(
        ChildAgentCounts(
            server_count=1,
            resource_counts=AgentResourceCounts(
                tool_count=None,
                prompt_count=0,
                resource_count=None,
            ),
        )
    )

    assert markup == (
        "[bold bright_cyan]1[/bold bright_cyan][dim] MCP Server[/dim][dim], [/dim]"
        "[dim]unknown tools[/dim][dim], [/dim]"
        "[dim]unknown resources[/dim]"
    )


def test_agent_info_text_treats_agent_name_literally() -> None:
    text = _agent_info_text("[draft]", "1[dim] tool[/dim]")

    assert text.plain == "Agent [draft]: 1 tool"


def test_child_agent_info_text_treats_agent_name_literally() -> None:
    text = _child_agent_info_text("[child]", "└─", "[dim]No MCP Servers[/dim]")

    assert text.plain == "  └─ [child]: No MCP Servers"


def test_display_agent_info_does_not_swallow_content_build_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Provider:
        def _agent(self, _agent_name: str) -> Any:
            return SimpleNamespace(name="demo")

    async def fail_build(_agent: AgentProtocol) -> str | None:
        raise RuntimeError("content build failed")

    monkeypatch.setattr(agent_info, "_build_agent_info_content", fail_build)

    with pytest.raises(RuntimeError, match="content build failed"):
        asyncio.run(display_agent_info("demo", cast("Any", Provider()), shown_agents=set()))


def test_resource_counts_for_agent_marks_failed_resource_probes_unknown() -> None:
    class Agent:
        async def list_tools(self) -> Any:
            raise RuntimeError("tools unavailable")

        async def list_resources(self) -> Any:
            raise RuntimeError("resources unavailable")

        async def list_prompts(self) -> Any:
            return {"demo": [object(), object()]}

    counts = asyncio.run(_resource_counts_for_agent(cast("AgentProtocol", Agent())))

    assert counts == AgentResourceCounts(
        tool_count=None,
        prompt_count=2,
        resource_count=None,
    )


def test_mcp_resource_counts_use_aggregator_only() -> None:
    class Aggregator:
        async def list_tools(self) -> ListToolsResult:
            return ListToolsResult(
                tools=[
                    Tool(name="hf.tool_1", inputSchema={}),
                    Tool(name="hf.tool_2", inputSchema={}),
                ]
            )

        async def list_resources(self) -> dict[str, list[str]]:
            return {"hf": ["skill://index.json", "skill://datasets/SKILL.md"]}

        async def list_prompts(self) -> dict[str, list[object]]:
            return {"hf": [object()]}

    agent = SimpleNamespace(aggregator=Aggregator())

    counts = asyncio.run(_mcp_resource_counts_for_agent(cast("Any", agent)))

    assert counts == AgentResourceCounts(
        tool_count=2,
        prompt_count=1,
        resource_count=2,
    )


def test_display_child_agent_info_does_not_swallow_formatting_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Agent:
        name = "child"

        async def list_tools(self) -> Any:
            return SimpleNamespace(tools=[])

        async def list_resources(self) -> Any:
            return {}

        async def list_prompts(self) -> Any:
            return {}

    def fail_text(_agent_name: str, _prefix: str, _detail_markup: str) -> object:
        raise RuntimeError("child formatting failed")

    monkeypatch.setattr(agent_info, "_child_agent_info_text", fail_text)

    with pytest.raises(RuntimeError, match="child formatting failed"):
        asyncio.run(
            _display_child_agent_info(
                cast("AgentProtocol", Agent()),
                "└─",
                agent_provider=None,
            )
        )
