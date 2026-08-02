"""Subagent slash command handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.commands.handlers import subagents as subagents_handlers
from fast_agent.commands.shared_command_intents import parse_subagents_command_intent

if TYPE_CHECKING:
    from fast_agent.acp.slash_commands import SlashCommandHandler


async def handle_subagents(
    handler: "SlashCommandHandler",
    arguments: str | None = None,
) -> str:
    intent = parse_subagents_command_intent(arguments)
    if intent.error is not None:
        return intent.error
    ctx = handler._build_command_context()
    outcome = await subagents_handlers.handle_subagents_command(
        ctx,
        agent_name=handler.current_agent_name,
        action=intent.action,
    )
    return handler._format_outcome_as_markdown(outcome, "subagents")
