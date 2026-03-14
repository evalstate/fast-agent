"""Session slash command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fast_agent.commands.handlers import sessions as sessions_handlers
from fast_agent.commands.handlers.shared import clear_agent_histories
from fast_agent.commands.renderers.session_markdown import render_session_list_markdown
from fast_agent.commands.session_summaries import build_session_list_summary
from fast_agent.commands.shared_command_intents import parse_session_command_intent

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
    intent = parse_session_command_intent(remainder)
    if intent.action == "help":
        return render_session_list(handler)
    if intent.action == "list":
        return render_session_list(handler)
    if intent.action == "new":
        return await handle_session_new(handler, intent.argument)
    if intent.action == "resume":
        return await handle_session_resume(handler, intent.argument)
    if intent.action == "title":
        return await handle_session_title(handler, intent.argument)
    if intent.action == "fork":
        return await handle_session_fork(handler, intent.argument)
    if intent.action == "delete":
        return await handle_session_delete(handler, intent.argument)
    if intent.action == "pin":
        return await handle_session_pin(handler, value=intent.pin_value, target=intent.pin_target)

    return "\n".join(
        [
            "# session",
            "",
            f"Unknown /session action: {intent.raw_subcommand or ''}",
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


async def handle_session_resume(handler: "SlashCommandHandler", argument: str | None) -> str:
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


async def handle_session_title(handler: "SlashCommandHandler", argument: str | None) -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    title = argument.strip() or None if argument is not None else None
    outcome = await sessions_handlers.handle_title_session(
        ctx,
        title=title,
        session_id=handler.session_id,
    )
    if title:
        await handler._send_session_info_update()
    return handler._format_outcome_as_markdown(outcome, "session title", io=io)


async def handle_session_fork(handler: "SlashCommandHandler", argument: str | None) -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await sessions_handlers.handle_fork_session(
        ctx,
        title=argument.strip() or None if argument is not None else None,
    )
    return handler._format_outcome_as_markdown(outcome, "session fork", io=io)


async def handle_session_new(handler: "SlashCommandHandler", argument: str | None) -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await sessions_handlers.handle_create_session(
        ctx,
        session_name=argument.strip() or None if argument is not None else None,
    )
    cleared = clear_agent_histories(handler.instance.agents, handler._logger)
    if cleared:
        outcome.add_message(
            f"Cleared agent history: {', '.join(sorted(cleared))}",
            channel="info",
        )
    return handler._format_outcome_as_markdown(outcome, "session new", io=io)


async def handle_session_delete(handler: "SlashCommandHandler", argument: str | None) -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await sessions_handlers.handle_clear_sessions(
        ctx,
        target=argument.strip() or None if argument is not None else None,
    )
    return handler._format_outcome_as_markdown(outcome, "session delete", io=io)


async def handle_session_pin(
    handler: "SlashCommandHandler",
    argument: str | None = None,
    *,
    value: str | None = None,
    target: str | None = None,
) -> str:
    if argument is not None:
        intent = parse_session_command_intent(f"pin {argument}")
        value = intent.pin_value
        target = intent.pin_target

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await sessions_handlers.handle_pin_session(
        ctx,
        value=value,
        target=target,
    )
    return handler._format_outcome_as_markdown(outcome, "session pin", io=io)
