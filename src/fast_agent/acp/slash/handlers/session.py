"""Session slash command handlers."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, cast

from fast_agent.commands.handlers import sessions as sessions_handlers
from fast_agent.commands.handlers.shared import clear_agent_histories
from fast_agent.commands.renderers.session_markdown import render_session_list_markdown
from fast_agent.commands.session_summaries import build_session_list_summary

if TYPE_CHECKING:
    from fast_agent.acp.command_io import ACPCommandIO
    from fast_agent.acp.slash_commands import SlashCommandHandler


async def handle_session(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    if handler._noenv:
        return "\n".join(
            [
                "# session",
                "",
                "Session commands are disabled in --noenv mode.",
            ]
        )

    remainder = (arguments or "").strip()
    if not remainder:
        return render_session_list(handler)

    try:
        tokens = shlex.split(remainder)
    except ValueError:
        tokens = remainder.split(maxsplit=1)

    if not tokens:
        return render_session_list(handler)

    subcmd = tokens[0].lower()
    argument = remainder[len(tokens[0]) :].strip()

    if subcmd == "list":
        return render_session_list(handler)
    if subcmd == "new":
        return await handle_session_new(handler, argument)
    if subcmd == "resume":
        return await handle_session_resume(handler, argument)
    if subcmd == "title":
        return await handle_session_title(handler, argument)
    if subcmd == "fork":
        return await handle_session_fork(handler, argument)
    if subcmd in {"delete", "clear"}:
        return await handle_session_delete(handler, argument)
    if subcmd == "pin":
        return await handle_session_pin(handler, argument)

    return "\n".join(
        [
            "# session",
            "",
            f"Unknown /session action: {subcmd}",
            "Usage: /session [list|new|resume|title|fork|delete|pin] [args]",
        ]
    )


def render_session_list(handler: "SlashCommandHandler") -> str:
    if handler._noenv:
        return "\n".join(
            [
                "# sessions",
                "",
                "Session commands are disabled in --noenv mode.",
            ]
        )
    summary = build_session_list_summary()
    return render_session_list_markdown(summary, heading="sessions")


async def handle_session_resume(handler: "SlashCommandHandler", argument: str) -> str:
    session_id = argument or None
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await sessions_handlers.handle_resume_session(
        ctx,
        agent_name=handler.current_agent_name,
        session_id=session_id,
    )
    if outcome.switch_agent:
        await handler._switch_current_mode(outcome.switch_agent)
    return handler._format_outcome_as_markdown(outcome, "session resume", io=io)


async def handle_session_title(handler: "SlashCommandHandler", argument: str) -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    title = argument.strip() or None
    outcome = await sessions_handlers.handle_title_session(
        ctx,
        title=title,
        session_id=handler.session_id,
    )
    if title:
        await handler._send_session_info_update()
    return handler._format_outcome_as_markdown(outcome, "session title", io=io)


async def handle_session_fork(handler: "SlashCommandHandler", argument: str) -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await sessions_handlers.handle_fork_session(
        ctx,
        title=argument.strip() or None,
    )
    return handler._format_outcome_as_markdown(outcome, "session fork", io=io)


async def handle_session_new(handler: "SlashCommandHandler", argument: str) -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await sessions_handlers.handle_create_session(
        ctx,
        session_name=argument.strip() or None,
    )
    cleared = clear_agent_histories(handler.instance.agents, handler._logger)
    if cleared:
        outcome.add_message(
            f"Cleared agent history: {', '.join(sorted(cleared))}",
            channel="info",
        )
    return handler._format_outcome_as_markdown(outcome, "session new", io=io)


async def handle_session_delete(handler: "SlashCommandHandler", argument: str) -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await sessions_handlers.handle_clear_sessions(
        ctx,
        target=argument.strip() or None,
    )
    return handler._format_outcome_as_markdown(outcome, "session delete", io=io)


async def handle_session_pin(handler: "SlashCommandHandler", argument: str) -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    pin_argument = argument.strip() if argument else ""
    value: str | None = None
    target: str | None = None
    if pin_argument:
        try:
            pin_tokens = shlex.split(pin_argument)
        except ValueError:
            pin_tokens = pin_argument.split(maxsplit=1)
        if pin_tokens:
            first = pin_tokens[0].lower()
            value_tokens = {
                "on",
                "off",
                "toggle",
                "true",
                "false",
                "yes",
                "no",
                "enable",
                "enabled",
                "disable",
                "disabled",
            }
            if first in value_tokens:
                value = first
                target = " ".join(pin_tokens[1:]).strip() or None
            else:
                target = pin_argument
    outcome = await sessions_handlers.handle_pin_session(
        ctx,
        value=value,
        target=target,
    )
    return handler._format_outcome_as_markdown(outcome, "session pin", io=io)
