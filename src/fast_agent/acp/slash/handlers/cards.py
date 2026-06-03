"""Agent card slash handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fast_agent.commands.handlers import agent_cards as agent_card_handlers
from fast_agent.commands.shared_command_intents import (
    parse_agent_tool_intent,
    parse_card_load_intent,
)

if TYPE_CHECKING:
    from fast_agent.acp.command_io import ACPCommandIO
    from fast_agent.acp.slash_commands import SlashCommandHandler


async def handle_card(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    intent = parse_card_load_intent(arguments)
    if intent.error:
        return intent.error
    manager = handler._build_card_manager()
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await agent_card_handlers.handle_card_load(
        ctx,
        manager=manager,
        filename=intent.filename,
        add_tool=intent.add_tool,
        remove_tool=intent.remove_tool,
        current_agent=handler.current_agent_name or handler.primary_agent_name,
    )
    return handler._format_outcome_as_markdown(outcome, "card", io=io)


async def handle_agent(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    intent = parse_agent_tool_intent(arguments)
    if intent.error:
        return intent.error

    target_agent = intent.agent_name or handler.current_agent_name or handler.primary_agent_name
    if not target_agent:
        return "No agent available for this session."

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await agent_card_handlers.handle_agent_command(
        ctx,
        manager=handler._build_card_manager(),
        current_agent=handler.current_agent_name or handler.primary_agent_name or target_agent,
        target_agent=intent.agent_name,
        add_tool=intent.add_tool,
        remove_tool=intent.remove_tool,
        dump=intent.dump,
    )
    return handler._format_outcome_as_markdown(outcome, "agent", io=io)


async def handle_reload(handler: "SlashCommandHandler") -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await agent_card_handlers.handle_reload_agents(
        ctx,
        manager=handler._build_card_manager(),
    )
    return handler._format_outcome_as_markdown(outcome, "reload", io=io)
