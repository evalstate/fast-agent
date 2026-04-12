"""History slash command handlers."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fast_agent.commands.handlers import history as history_handlers
from fast_agent.commands.history_summaries import build_history_turn_report
from fast_agent.commands.renderers.history_markdown import (
    render_history_overview_markdown,
    render_history_turn_report_markdown,
)
from fast_agent.commands.shared_command_intents import parse_current_agent_history_intent

if TYPE_CHECKING:
    from fast_agent.acp.command_io import ACPCommandIO
    from fast_agent.acp.slash_commands import SlashCommandHandler


async def handle_history(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    remainder = (arguments or "").strip()
    if not remainder:
        return await render_history_overview(handler)

    try:
        tokens = shlex.split(remainder)
    except ValueError:
        tokens = remainder.split(maxsplit=1)

    if not tokens:
        return await render_history_overview(handler)

    subcmd = tokens[0].lower()
    webclear_enabled = history_handlers.web_tools_enabled_for_agent(handler._get_current_agent())

    if subcmd == "webclear":
        return await _handle_history_webclear_command(
            handler,
            webclear_enabled=webclear_enabled,
        )

    intent = parse_current_agent_history_intent(remainder)
    handled = await _dispatch_shared_history_intent(handler, intent=intent)
    if handled is not None:
        return handled

    return _unknown_history_action_response(
        raw_subcommand=intent.raw_subcommand or subcmd,
        webclear_enabled=webclear_enabled,
    )


async def _handle_history_webclear_command(
    handler: "SlashCommandHandler",
    *,
    webclear_enabled: bool,
) -> str:
    if not webclear_enabled:
        return "\n".join(
            [
                "# history",
                "",
                "Unknown /history action: webclear",
                "Usage: /history [show|detail <turn>|save|load] [args]",
            ]
        )
    return await handle_history_webclear(handler)


async def _dispatch_shared_history_intent(
    handler: "SlashCommandHandler",
    *,
    intent,
) -> str | None:
    if intent.action == "overview":
        return await render_history_overview(handler)
    if intent.action == "show":
        return await handle_show(handler)
    if intent.action == "detail":
        return await handle_detail(
            handler,
            turn_index=intent.turn_index,
            turn_error=intent.turn_error,
        )
    if intent.action == "save":
        return await handle_save(handler, intent.argument)
    if intent.action == "load":
        return await handle_load(handler, intent.argument)
    return None


def _unknown_history_action_response(*, raw_subcommand: str, webclear_enabled: bool) -> str:
    return "\n".join(
        [
            "# history",
            "",
            f"Unknown /history action: {raw_subcommand}",
            (
                "Usage: /history [show|detail <turn>|save|load|webclear] [args]"
                if webclear_enabled
                else "Usage: /history [show|detail <turn>|save|load] [args]"
            ),
        ]
    )


async def render_history_overview(handler: "SlashCommandHandler") -> str:
    heading = "# conversation history"
    agent, error = handler._get_current_agent_or_error(heading)
    if error:
        return error
    assert agent is not None

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    await history_handlers.handle_show_history(
        ctx,
        agent_name=handler.current_agent_name,
    )
    if not io.history_overview:
        return "\n".join([heading, "", "No messages yet."])

    return render_history_overview_markdown(
        io.history_overview,
        heading="conversation history",
    )


async def handle_show(handler: "SlashCommandHandler") -> str:
    heading = "# history show"
    agent, error = handler._get_current_agent_or_error(heading)
    if error:
        return error
    assert agent is not None

    history = list(agent.message_history)
    report = build_history_turn_report(history)
    return render_history_turn_report_markdown(report, heading="history show")


async def handle_detail(
    handler: "SlashCommandHandler",
    *,
    turn_index: int | None,
    turn_error: str | None,
) -> str:
    heading = "# history detail"

    _, error = handler._get_current_agent_or_error(
        heading,
        missing_template=f"Unable to locate agent '{handler.current_agent_name}' for this session.",
    )
    if error:
        return error

    error_message = None
    if turn_error == "missing":
        error_message = "Turn number required for /history detail."
    elif turn_error == "invalid":
        error_message = "Turn number must be an integer."

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await history_handlers.handle_history_review(
        ctx,
        agent_name=handler.current_agent_name,
        turn_index=turn_index,
        error=error_message,
    )
    return handler._format_outcome_as_markdown(outcome, "history detail", io=io)


async def handle_save(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    heading = "# save conversation"

    agent, error = handler._get_current_agent_or_error(
        heading,
        missing_template=f"Unable to locate agent '{handler.current_agent_name}' for this session.",
    )
    if error:
        return error
    assert agent is not None

    filename = arguments.strip() if arguments and arguments.strip() else None

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await history_handlers.handle_history_save(
        ctx,
        agent_name=handler.current_agent_name,
        filename=filename,
        send_func=None,
        history_exporter=handler.history_exporter,
    )
    return handler._format_outcome_as_markdown(outcome, "save conversation", io=io)


async def handle_load(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    heading = "# load conversation"

    agent, error = handler._get_current_agent_or_error(
        heading,
        missing_template=f"Unable to locate agent '{handler.current_agent_name}' for this session.",
    )
    if error:
        return error
    assert agent is not None

    filename = arguments.strip() if arguments and arguments.strip() else None
    error_message = None
    if not filename:
        error_message = "Filename required for /history load."
    else:
        file_path = Path(filename)
        if not file_path.exists():
            error_message = f"File not found: {filename}"

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await history_handlers.handle_history_load(
        ctx,
        agent_name=handler.current_agent_name,
        filename=filename,
        error=error_message,
    )
    return handler._format_outcome_as_markdown(outcome, "load conversation", io=io)


async def handle_history_webclear(handler: "SlashCommandHandler") -> str:
    heading = "# history webclear"

    _, error = handler._get_current_agent_or_error(
        heading,
        missing_template=f"Unable to locate agent '{handler.current_agent_name}' for this session.",
    )
    if error:
        return error

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await history_handlers.handle_history_webclear(
        ctx,
        agent_name=handler.current_agent_name,
        target_agent=None,
    )
    return handler._format_outcome_as_markdown(outcome, "history webclear", io=io)
