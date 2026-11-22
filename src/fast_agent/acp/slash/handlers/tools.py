"""Tools slash command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fast_agent.commands.handlers.tools import ToolsQuery, handle_list_tools, parse_tools_query
from fast_agent.commands.renderers.tools_markdown import (
    render_tool_query_markdown,
    render_tools_markdown,
)
from fast_agent.commands.tool_summaries import build_tools_listing
from fast_agent.interfaces import AgentProtocol
from fast_agent.tools.tool_surface import resolve_tool_surface

if TYPE_CHECKING:
    from fast_agent.acp.command_io import ACPCommandIO
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

    if not isinstance(query, ToolsQuery):
        return await _handle_tools_mentions(handler, arguments)

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
        detail = render_tool_query_markdown(query.tool_name, surface)
        if detail is None:
            return "\n".join([f"# {heading}", "", f"Tool not found: {query.tool_name}"])
        return detail

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


async def _handle_tools_mentions(handler: "SlashCommandHandler", arguments: str | None) -> str:
    """``/tools add`` / ``/tools select``: ACP cannot edit the input, so show the draft text."""
    ctx = handler._build_command_context()
    outcome = await handle_list_tools(
        ctx, agent_name=handler.current_agent_name, argument=arguments
    )
    if outcome.buffer_prefill:
        outcome.add_message(
            f"Command produced draft text:\n\n```text\n{outcome.buffer_prefill}\n```",
            render_markdown=True,
        )
    return handler._format_outcome_as_markdown(outcome, "tools", io=cast("ACPCommandIO", ctx.io))
