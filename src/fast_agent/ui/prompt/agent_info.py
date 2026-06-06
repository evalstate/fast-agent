"""Agent info display helpers for prompt startup and hierarchy rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from rich import print as rich_print
from rich.text import Text

from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.agents.workflow.router_agent import RouterAgent
from fast_agent.interfaces import AgentBackedToolProvider, AgentProtocol
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.utils.count_display import plural_label

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fast_agent.core.agent_app import AgentApp


@dataclass(frozen=True, slots=True)
class AgentResourceCounts:
    tool_count: int | None
    prompt_count: int | None
    resource_count: int | None


@dataclass(frozen=True, slots=True)
class ChildAgentCounts:
    server_count: int
    resource_counts: AgentResourceCounts


async def display_agent_info(
    agent_name: str,
    agent_provider: "AgentApp | None",
    *,
    shown_agents: set[str],
) -> None:
    """Display startup info for a single agent once per prompt lifetime."""
    if agent_name in shown_agents or agent_provider is None:
        return

    try:
        agent = agent_provider._agent(agent_name)
    except Exception:
        return

    content = await _build_agent_info_content(agent)
    if content:
        rich_print(_agent_info_text(agent_name, content))
    await _show_skybridge_summary(agent_name, agent)
    shown_agents.add(agent_name)


def _agent_info_text(agent_name: str, content_markup: str) -> Text:
    text = Text()
    text.append("Agent ", style="dim")
    text.append(agent_name, style="blue")
    text.append(": ", style="dim")
    text.append_text(Text.from_markup(content_markup))
    return text


async def _build_agent_info_content(agent: AgentProtocol) -> str | None:
    if isinstance(agent, ParallelAgent):
        return None

    if isinstance(agent, RouterAgent):
        return None

    content_parts = await _build_standard_agent_info_parts(agent)
    if not content_parts:
        return None
    return "[dim]. [/dim]".join(content_parts)


def _format_dim_count(
    count: int,
    singular: str,
    plural: str | None = None,
    *,
    suffix: str = "",
) -> str:
    label = plural_label(count, singular, plural)
    return f"[bold bright_cyan]{count:,}[/bold bright_cyan][dim] {label}{suffix}[/dim]"


async def _build_standard_agent_info_parts(agent: AgentProtocol) -> list[str]:
    content_parts: list[str] = []

    server_count = _server_count_for_agent(agent)
    resource_counts = await _resource_counts_for_agent(agent)
    if server_count > 0:
        content_parts.append(
            _format_server_summary(
                server_count=server_count,
                tool_count=resource_counts.tool_count,
                prompt_count=resource_counts.prompt_count,
                resource_count=resource_counts.resource_count,
            )
        )

    skill_count = _skill_count_for_agent(agent)
    if skill_count > 0:
        content_parts.append(_format_installed_skill_count(skill_count))

    return content_parts


def _format_installed_skill_count(skill_count: int) -> str:
    return _format_dim_count(skill_count, "skill", suffix=" installed")


def _server_count_for_agent(agent: AgentProtocol) -> int:
    if not isinstance(agent, McpAgentProtocol):
        return 0
    server_names = agent.aggregator.server_names
    return len(server_names) if server_names else 0


async def _resource_counts_for_agent(agent: AgentProtocol) -> AgentResourceCounts:
    if isinstance(agent, McpAgentProtocol):
        return await _mcp_resource_counts_for_agent(agent)

    try:
        tools_result = await agent.list_tools()
        tool_count = len(tools_result.tools)
    except Exception:
        tool_count = None

    try:
        resources_dict = await agent.list_resources()
        resource_count = (
            sum(len(resources) for resources in resources_dict.values()) if resources_dict else 0
        )
    except Exception:
        resource_count = None

    try:
        prompts_dict = await agent.list_prompts()
        prompt_count = sum(len(prompts) for prompts in prompts_dict.values()) if prompts_dict else 0
    except Exception:
        prompt_count = None
    return AgentResourceCounts(
        tool_count=tool_count,
        prompt_count=prompt_count,
        resource_count=resource_count,
    )


async def _mcp_resource_counts_for_agent(agent: McpAgentProtocol) -> AgentResourceCounts:
    """Count MCP server surfaces without local runtime or child-agent tools."""
    try:
        tools_result = await agent.aggregator.list_tools()
        tool_count = len(tools_result.tools)
    except Exception:
        tool_count = None

    try:
        resources_dict = await agent.aggregator.list_resources()
        resource_count = (
            sum(len(resources) for resources in resources_dict.values()) if resources_dict else 0
        )
    except Exception:
        resource_count = None

    try:
        prompts_dict = await agent.aggregator.list_prompts()
        prompt_count = sum(len(prompts) for prompts in prompts_dict.values()) if prompts_dict else 0
    except Exception:
        prompt_count = None

    return AgentResourceCounts(
        tool_count=tool_count,
        prompt_count=prompt_count,
        resource_count=resource_count,
    )


def _format_server_summary(
    *,
    server_count: int,
    tool_count: int | None,
    prompt_count: int | None,
    resource_count: int | None,
) -> str:
    sub_parts = _resource_count_parts(
        (
            (tool_count, "tool"),
            (prompt_count, "prompt"),
            (resource_count, "resource"),
        )
    )

    server_text = _format_dim_count(server_count, "MCP Server", "MCP Servers")
    if not sub_parts:
        return server_text
    return f"{server_text}[dim] ([/dim]" + "[dim], [/dim]".join(sub_parts) + "[dim])[/dim]"


def _resource_count_parts(counts: Iterable[tuple[int | None, str]]) -> list[str]:
    parts: list[str] = []
    for count, noun in counts:
        if count is None:
            parts.append(f"[dim]unknown {plural_label(2, noun)}[/dim]")
        elif count > 0:
            parts.append(_format_dim_count(count, noun))
    return parts


def _skill_count_for_agent(agent: AgentProtocol) -> int:
    if not isinstance(agent, McpAgentProtocol) or not agent.skill_manifests:
        return 0
    try:
        return len(list(agent.skill_manifests))
    except TypeError:
        return 0


async def _show_skybridge_summary(agent_name: str, agent: AgentProtocol) -> None:
    if not isinstance(agent, McpAgentProtocol):
        return
    try:
        skybridge_configs = await agent.aggregator.get_skybridge_configs()
    except Exception:
        return
    agent.display.show_skybridge_summary(agent_name, skybridge_configs)


async def display_all_agents_with_hierarchy(
    available_agents: "Iterable[str]",
    agent_provider: "AgentApp | None",
    *,
    shown_agents: set[str],
) -> None:
    """Display all top-level agents and their children with tree structure."""
    if agent_provider is None:
        return

    agent_list = list(available_agents)
    child_agents = await _collect_child_agent_names(agent_list, agent_provider)
    for agent_name in sorted(agent_list):
        if agent_name in child_agents:
            continue
        try:
            agent = agent_provider._agent(agent_name)
        except Exception:
            continue

        await display_agent_info(agent_name, agent_provider, shown_agents=shown_agents)
        await _display_agent_children(agent, agent_provider)


async def _collect_child_agent_names(
    agent_names: list[str],
    agent_provider: "AgentApp",
) -> set[str]:
    child_agents: set[str] = set()
    for agent_name in agent_names:
        try:
            agent = agent_provider._agent(agent_name)
        except Exception:
            continue
        for child_agent in _child_agents_for_display(agent):
            child_agents.add(child_agent.name)
    return child_agents


async def _display_agent_children(agent: AgentProtocol, agent_provider: "AgentApp") -> None:
    if isinstance(agent, ParallelAgent):
        await _display_parallel_children(agent, agent_provider)
        return
    if isinstance(agent, RouterAgent):
        await _display_router_children(agent, agent_provider)
        return

    tool_children = collect_tool_children(agent)
    if tool_children:
        await _display_tool_children(tool_children, agent_provider)


def _child_agents_for_display(agent: AgentProtocol) -> list[AgentProtocol]:
    if isinstance(agent, ParallelAgent):
        children = list(agent.fan_out_agents) if agent.fan_out_agents else []
        if agent.fan_in_agent is not None:
            children.append(agent.fan_in_agent)
        return [cast("AgentProtocol", child) for child in children]
    if isinstance(agent, RouterAgent):
        return [cast("AgentProtocol", child) for child in (agent.agents or [])]
    return collect_tool_children(agent)


async def _display_parallel_children(
    parallel_agent: ParallelAgent, agent_provider: "AgentApp"
) -> None:
    children = _child_agents_for_display(parallel_agent)
    await _display_child_agents(children, agent_provider)


async def _display_router_children(router_agent: RouterAgent, agent_provider: "AgentApp") -> None:
    children = _child_agents_for_display(router_agent)
    await _display_child_agents(children, agent_provider)


async def _display_tool_children(
    tool_children: list[AgentProtocol], agent_provider: "AgentApp"
) -> None:
    await _display_child_agents(tool_children, agent_provider)


async def _display_child_agents(children: list[AgentProtocol], agent_provider: "AgentApp") -> None:
    for index, child_agent in enumerate(children):
        prefix = "└─" if index == len(children) - 1 else "├─"
        await _display_child_agent_info(child_agent, prefix, agent_provider)


def collect_tool_children(agent: object) -> list[AgentProtocol]:
    """Collect child agents exposed as tools."""
    if not isinstance(agent, AgentBackedToolProvider):
        return []

    seen: set[str] = set()
    unique_children: list[AgentProtocol] = []
    for child in agent.agent_backed_tools.values():
        if not isinstance(child, AgentProtocol):
            continue
        if child.name in seen:
            continue
        seen.add(child.name)
        unique_children.append(child)
    return unique_children


async def _display_child_agent_info(
    child_agent: AgentProtocol,
    prefix: str,
    agent_provider: "AgentApp | None",
) -> None:
    del agent_provider
    counts = await _child_agent_counts(child_agent)
    if counts.server_count > 0:
        rich_print(
            _child_agent_info_text(
                child_agent.name,
                prefix,
                _child_agent_counts_markup(counts),
            )
        )
        return
    rich_print(_child_agent_info_text(child_agent.name, prefix, "[dim]No MCP Servers[/dim]"))


def _child_agent_counts_markup(counts: ChildAgentCounts) -> str:
    sub_parts = _resource_count_parts(
        (
            (counts.resource_counts.tool_count, "tool"),
            (counts.resource_counts.resource_count, "resource"),
            (counts.resource_counts.prompt_count, "prompt"),
        )
    )

    server_text = _format_dim_count(counts.server_count, "MCP Server", "MCP Servers")
    if not sub_parts:
        return server_text
    return f"{server_text}[dim], [/dim]" + "[dim], [/dim]".join(sub_parts)


def _child_agent_name_text(agent_name: str, prefix: str) -> Text:
    text = Text()
    text.append(f"  {prefix} ", style="dim")
    text.append(agent_name, style="blue")
    return text


def _child_agent_info_text(agent_name: str, prefix: str, detail_markup: str) -> Text:
    text = _child_agent_name_text(agent_name, prefix)
    text.append(": ", style="dim")
    text.append_text(Text.from_markup(detail_markup))
    return text


async def _child_agent_counts(child_agent: AgentProtocol) -> ChildAgentCounts:
    server_count = 0
    if isinstance(child_agent, McpAgentProtocol):
        try:
            server_names = child_agent.aggregator.server_names
            server_count = len(server_names) if server_names else 0
        except Exception:
            server_count = 0

    return ChildAgentCounts(
        server_count=server_count,
        resource_counts=await _resource_counts_for_agent(child_agent),
    )
