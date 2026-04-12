"""Agent info display helpers for prompt startup and hierarchy rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rich import print as rich_print

from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.agents.workflow.router_agent import RouterAgent
from fast_agent.interfaces import AgentBackedToolProvider, AgentProtocol
from fast_agent.mcp.types import McpAgentProtocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fast_agent.core.agent_app import AgentApp



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

    try:
        content = await _build_agent_info_content(agent)
        if content:
            rich_print(f"[dim]Agent [/dim][blue]{agent_name}[/blue][dim]:[/dim] {content}")
        await _show_skybridge_summary(agent_name, agent)
        shown_agents.add(agent_name)
    except Exception:
        return


async def _build_agent_info_content(agent: AgentProtocol) -> str | None:
    if isinstance(agent, ParallelAgent):
        child_count = len(agent.fan_out_agents) + (1 if agent.fan_in_agent else 0)
        return _format_child_agent_count(child_count)

    if isinstance(agent, RouterAgent):
        child_count = len(agent.agents) if agent.agents else 0
        return _format_child_agent_count(child_count)

    content_parts = await _build_standard_agent_info_parts(agent)
    if not content_parts:
        return None
    return "[dim]. [/dim]".join(content_parts)


def _format_child_agent_count(child_count: int) -> str | None:
    if child_count <= 0:
        return None
    child_word = "child agent" if child_count == 1 else "child agents"
    return f"{child_count:,}[dim] {child_word}[/dim]"


async def _build_standard_agent_info_parts(agent: AgentProtocol) -> list[str]:
    content_parts: list[str] = []
    tool_children = collect_tool_children(agent)
    if tool_children:
        child_count = len(tool_children)
        child_word = "child agent" if child_count == 1 else "child agents"
        content_parts.append(f"{child_count:,}[dim] {child_word}[/dim]")

    server_count = _server_count_for_agent(agent)
    tool_count, prompt_count, resource_count = await _resource_counts_for_agent(agent)
    if server_count > 0:
        content_parts.append(
            _format_server_summary(
                server_count=server_count,
                tool_count=tool_count,
                prompt_count=prompt_count,
                resource_count=resource_count,
            )
        )

    skill_count = _skill_count_for_agent(agent)
    if skill_count > 0:
        skill_word = "skill" if skill_count == 1 else "skills"
        content_parts.append(f"{skill_count:,}[dim] {skill_word}[/dim][dim] available[/dim]")

    return content_parts


def _server_count_for_agent(agent: AgentProtocol) -> int:
    if not isinstance(agent, McpAgentProtocol):
        return 0
    server_names = agent.aggregator.server_names
    return len(server_names) if server_names else 0


async def _resource_counts_for_agent(agent: AgentProtocol) -> tuple[int, int, int]:
    tools_result = await agent.list_tools()
    tool_count = len(tools_result.tools)

    resources_dict = await agent.list_resources()
    resource_count = sum(len(resources) for resources in resources_dict.values()) if resources_dict else 0

    prompts_dict = await agent.list_prompts()
    prompt_count = sum(len(prompts) for prompts in prompts_dict.values()) if prompts_dict else 0
    return tool_count, prompt_count, resource_count


def _format_server_summary(
    *,
    server_count: int,
    tool_count: int,
    prompt_count: int,
    resource_count: int,
) -> str:
    sub_parts: list[str] = []
    if tool_count > 0:
        tool_word = "tool" if tool_count == 1 else "tools"
        sub_parts.append(f"{tool_count:,}[dim] {tool_word}[/dim]")
    if prompt_count > 0:
        prompt_word = "prompt" if prompt_count == 1 else "prompts"
        sub_parts.append(f"{prompt_count:,}[dim] {prompt_word}[/dim]")
    if resource_count > 0:
        resource_word = "resource" if resource_count == 1 else "resources"
        sub_parts.append(f"{resource_count:,}[dim] {resource_word}[/dim]")

    server_word = "Server" if server_count == 1 else "Servers"
    server_text = f"{server_count:,}[dim] MCP {server_word}[/dim]"
    if not sub_parts:
        return server_text
    return f"{server_text}[dim] ([/dim]" + "[dim], [/dim]".join(sub_parts) + "[dim])[/dim]"


def _skill_count_for_agent(agent: AgentProtocol) -> int:
    if not isinstance(agent, McpAgentProtocol) or not agent.skill_manifests:
        return 0
    try:
        return len(list(agent.skill_manifests))
    except TypeError:
        return 0


async def _show_skybridge_summary(agent_name: str, agent: AgentProtocol) -> None:
    try:
        if isinstance(agent, McpAgentProtocol):
            skybridge_configs = await agent.aggregator.get_skybridge_configs()
            agent.display.show_skybridge_summary(agent_name, skybridge_configs)
    except Exception:
        return


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


async def _display_parallel_children(parallel_agent: ParallelAgent, agent_provider: "AgentApp") -> None:
    children = _child_agents_for_display(parallel_agent)
    await _display_child_agents(children, agent_provider)


async def _display_router_children(router_agent: RouterAgent, agent_provider: "AgentApp") -> None:
    children = _child_agents_for_display(router_agent)
    await _display_child_agents(children, agent_provider)


async def _display_tool_children(
    tool_children: list[AgentProtocol], agent_provider: "AgentApp"
) -> None:
    await _display_child_agents(tool_children, agent_provider)


async def _display_child_agents(
    children: list[AgentProtocol], agent_provider: "AgentApp"
) -> None:
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
    try:
        server_count, tool_count, prompt_count, resource_count = await _child_agent_counts(child_agent)
        if server_count > 0:
            rich_print(
                f"[dim]  {prefix} [/dim][blue]{child_agent.name}[/blue][dim]:[/dim] "
                f"{server_count:,}[dim] MCP {'Server' if server_count == 1 else 'Servers'}, [/dim]"
                f"{tool_count:,}[dim] {'tool' if tool_count == 1 else 'tools'}, [/dim]"
                f"{resource_count:,}[dim] {'resource' if resource_count == 1 else 'resources'}, [/dim]"
                f"{prompt_count:,}[dim] {'prompt' if prompt_count == 1 else 'prompts'} available[/dim]"
            )
            return
        rich_print(f"[dim]  {prefix} [/dim][blue]{child_agent.name}[/blue][dim]: No MCP Servers[/dim]")
    except Exception:
        rich_print(f"[dim]  {prefix} [/dim][blue]{child_agent.name}[/blue]")


async def _child_agent_counts(child_agent: AgentProtocol) -> tuple[int, int, int, int]:
    server_count = 0
    if isinstance(child_agent, McpAgentProtocol):
        server_names = child_agent.aggregator.server_names
        server_count = len(server_names) if server_names else 0

    tools_result = await child_agent.list_tools()
    tool_count = len(tools_result.tools)

    resources_dict = await child_agent.list_resources()
    resource_count = sum(len(resources) for resources in resources_dict.values()) if resources_dict else 0

    prompts_dict = await child_agent.list_prompts()
    prompt_count = sum(len(prompts) for prompts in prompts_dict.values()) if prompts_dict else 0
    return server_count, tool_count, prompt_count, resource_count
