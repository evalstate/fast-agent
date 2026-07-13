"""Tools slash command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.commands.renderers.tools_markdown import (
    render_tool_schema_markdown,
    render_tools_markdown,
)
from fast_agent.commands.tool_summaries import build_provider_tool_summaries, build_tool_summaries
from fast_agent.interfaces import AgentProtocol

if TYPE_CHECKING:
    from mcp.types import ListToolsResult

    from fast_agent.acp.slash_commands import SlashCommandHandler


async def handle_tools(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    heading = "tools"

    agent, error = handler._get_current_agent_or_error(f"# {heading}")
    if error:
        return error

    if not isinstance(agent, AgentProtocol):
        return "\n".join(
            [
                f"# {heading}",
                "",
                "This agent does not support tool listing.",
            ]
        )

    try:
        tools_result: "ListToolsResult" = await agent.list_tools()
    except Exception as exc:
        return "\n".join(
            [
                f"# {heading}",
                "",
                "Failed to fetch tools from the agent.",
                f"Details: {exc}",
            ]
        )

    tools = tools_result.tools if tools_result else []
    tool_name = arguments.strip() if arguments else ""
    if tool_name and tool_name.casefold() != "summary":
        tool = next((tool for tool in tools if tool.name == tool_name), None)
        if tool is None:
            return "\n".join([f"# {heading}", "", f"Tool not found: {tool_name}"])
        return render_tool_schema_markdown(tool)

    provider_summaries = build_provider_tool_summaries(agent)
    if not tools and not provider_summaries:
        return "\n".join(
            [
                f"# {heading}",
                "",
                "No tools available for this agent.",
            ]
        )

    summaries = build_tool_summaries(agent, list(tools))
    return render_tools_markdown(
        summaries,
        heading=heading,
        provider_summaries=provider_summaries,
    )
