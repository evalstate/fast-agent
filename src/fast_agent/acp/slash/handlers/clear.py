"""Clear slash command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fast_agent.commands.handlers import history as history_handlers

if TYPE_CHECKING:
    from fast_agent.acp.command_io import ACPCommandIO
    from fast_agent.acp.slash_commands import SlashCommandHandler


async def handle_clear(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    normalized = (arguments or "").strip().lower()
    if normalized == "last":
        return await handle_clear_last(handler)
    return await handle_clear_all(handler)


async def handle_clear_all(handler: "SlashCommandHandler") -> str:
    heading = "# clear conversation"
    agent, error = handler._get_current_agent_or_error(
        heading,
        missing_template=f"Unable to locate agent '{handler.current_agent_name}' for this session.",
    )
    if error:
        return error
    assert agent is not None

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await history_handlers.handle_history_clear_all(
        ctx,
        agent_name=handler.current_agent_name,
    )
    return handler._format_outcome_as_markdown(outcome, "clear conversation", io=io)


async def handle_clear_last(handler: "SlashCommandHandler") -> str:
    heading = "# clear last conversation turn"
    agent, error = handler._get_current_agent_or_error(
        heading,
        missing_template=f"Unable to locate agent '{handler.current_agent_name}' for this session.",
    )
    if error:
        return error
    assert agent is not None

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await history_handlers.handle_history_clear_last(
        ctx,
        agent_name=handler.current_agent_name,
    )
    return handler._format_outcome_as_markdown(outcome, "clear last conversation turn", io=io)
