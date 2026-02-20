"""Agent card slash handlers."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, cast

from fast_agent.commands.handlers import agent_cards as agent_card_handlers

if TYPE_CHECKING:
    from fast_agent.acp.command_io import ACPCommandIO
    from fast_agent.acp.slash_commands import SlashCommandHandler


async def handle_card(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    args = (arguments or "").strip()
    tokens: list[str] = []
    if args:
        try:
            tokens = shlex.split(args)
        except ValueError as exc:
            return f"Invalid arguments: {exc}"

    add_tool = False
    remove_tool = False
    filename = None
    for token in tokens:
        if token in {"tool", "--tool", "--as-tool", "-t"}:
            add_tool = True
            continue
        if token in {"remove", "--remove"}:
            add_tool = True
            remove_tool = True
            continue
        if filename is None:
            filename = token

    manager = handler._build_card_manager()
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await agent_card_handlers.handle_card_load(
        ctx,
        manager=manager,
        filename=filename,
        add_tool=add_tool,
        remove_tool=remove_tool,
        current_agent=handler.current_agent_name or handler.primary_agent_name,
    )
    return handler._format_outcome_as_markdown(outcome, "card", io=io)


async def handle_agent(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    args = (arguments or "").strip()
    if not args:
        return "Usage: /agent <name> --tool | /agent [name] --dump"

    try:
        tokens = shlex.split(args)
    except ValueError as exc:
        return f"Invalid arguments: {exc}"

    add_tool = False
    remove_tool = False
    dump = False
    agent_name = None
    unknown: list[str] = []
    for token in tokens:
        if token in {"tool", "--tool", "--as-tool", "-t"}:
            add_tool = True
            continue
        if token in {"remove", "--remove"}:
            add_tool = True
            remove_tool = True
            continue
        if token in {"dump", "--dump", "-d"}:
            dump = True
            continue
        if agent_name is None:
            agent_name = token[1:] if token.startswith("@") else token
            continue
        unknown.append(token)

    if unknown:
        return f"Unexpected arguments: {', '.join(unknown)}"
    if add_tool and dump:
        return "Use either --tool or --dump, not both."
    if not add_tool and not dump:
        return "Usage: /agent <name> --tool [remove] | /agent [name] --dump"

    target_agent = agent_name or handler.current_agent_name or handler.primary_agent_name
    if not target_agent:
        return "No agent available for this session."

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await agent_card_handlers.handle_agent_command(
        ctx,
        manager=handler._build_card_manager(),
        current_agent=handler.current_agent_name or handler.primary_agent_name or target_agent,
        target_agent=agent_name,
        add_tool=add_tool,
        remove_tool=remove_tool,
        dump=dump,
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
