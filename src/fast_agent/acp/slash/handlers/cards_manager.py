"""Card pack slash command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fast_agent.commands.command_catalog import normalize_command_action
from fast_agent.commands.command_discovery import render_direct_command_help
from fast_agent.commands.handlers import cards_manager as cards_handlers
from fast_agent.utils.action_normalization import split_action_arguments

if TYPE_CHECKING:
    from fast_agent.acp.command_io import ACPCommandIO
    from fast_agent.acp.slash_commands import SlashCommandHandler


def _parse_cards_arguments(arguments: str | None) -> tuple[str, str]:
    requested_action, remainder = split_action_arguments(arguments)
    return normalize_command_action("cards", requested_action), remainder


async def handle_cards(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    direct_help = render_direct_command_help("cards", arguments)
    if direct_help is not None:
        return direct_help

    action, remainder = _parse_cards_arguments(arguments)

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    try:
        outcome = await cards_handlers.handle_cards_command(
            ctx,
            agent_name=handler.current_agent_name,
            action=action,
            argument=remainder or None,
        )
    except Exception as exc:
        return f"# cards\n\nFailed to execute /cards: {exc}"

    heading = "cards" if action == "list" else f"cards {action}"
    return handler._format_outcome_as_markdown(outcome, heading, io=io)
