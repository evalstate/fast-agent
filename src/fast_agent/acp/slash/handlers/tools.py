"""Tools slash command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.commands.handlers.tools import parse_tools_query
from fast_agent.commands.renderers.tools_markdown import (
    render_tool_detail_markdown,
    render_tools_markdown,
)
from fast_agent.commands.tool_summaries import build_tools_listing
from fast_agent.interfaces import AgentProtocol
from fast_agent.tools.tool_surface import resolve_tool_surface

if TYPE_CHECKING:
    from fast_agent.acp.slash_commands import SlashCommandHandler
    from fast_agent.tools.tool_surface import ToolSurface


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
        query = parse_tools_query(arguments)
    except ValueError as exc:
        return "\n".join([f"# {heading}", "", str(exc)])

    try:
        surface: "ToolSurface" = await resolve_tool_surface(agent)
    except Exception as exc:
        return "\n".join(
            [
                f"# {heading}",
                "",
                "Failed to fetch tools from the agent.",
                f"Details: {exc}",
            ]
        )

    if query.tool_name is not None:
        surface_tool = surface.find(query.tool_name)
        if surface_tool is None:
            return "\n".join([f"# {heading}", "", f"Tool not found: {query.tool_name}"])
        return render_tool_detail_markdown(surface_tool, surface)

    listing = build_tools_listing(agent, surface)
    if listing.is_empty:
        return "\n".join(
            [
                f"# {heading}",
                "",
                "No tools available for this agent.",
            ]
        )

    return render_tools_markdown(listing, heading=heading, show_all=query.show_all)
