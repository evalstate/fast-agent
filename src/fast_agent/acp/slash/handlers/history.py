"""History slash command handlers."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fast_agent.commands.handlers import history as history_handlers
from fast_agent.commands.renderers.history_markdown import render_history_overview_markdown

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
    argument = remainder[len(tokens[0]) :].strip()
    webclear_enabled = history_handlers.web_tools_enabled_for_agent(handler._get_current_agent())

    if subcmd in {"show", "list"}:
        return await render_history_overview(handler)
    if subcmd == "save":
        return await handle_save(handler, argument)
    if subcmd == "load":
        return await handle_load(handler, argument)
    if subcmd == "webclear":
        if not webclear_enabled:
            return "\n".join(
                [
                    "# history",
                    "",
                    "Unknown /history action: webclear",
                    "Usage: /history [show|save|load] [args]",
                ]
            )
        return await handle_history_webclear(handler)

    return "\n".join(
        [
            "# history",
            "",
            f"Unknown /history action: {subcmd}",
            (
                "Usage: /history [show|save|load|webclear] [args]"
                if webclear_enabled
                else "Usage: /history [show|save|load] [args]"
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
