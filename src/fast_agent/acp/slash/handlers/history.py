"""History slash command handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fast_agent.commands.handlers import history as history_handlers
from fast_agent.commands.history_summaries import build_history_turn_report
from fast_agent.commands.renderers.history_markdown import (
    render_history_overview_markdown,
    render_history_turn_report_markdown,
)
from fast_agent.commands.shared_command_intents import (
    HistoryAction,
    HistoryActionIntent,
    HistoryTurnAction,
    HistoryTurnError,
    parse_current_agent_history_intent,
)
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from fast_agent.acp.command_io import ACPCommandIO
    from fast_agent.acp.slash_commands import SlashCommandHandler

_HistoryIntentHandler = Callable[["SlashCommandHandler", HistoryActionIntent], Awaitable[str]]


_HISTORY_USAGE = (
    "Usage: /history [show|detail <turn>|save|load|clear [last]|rewind <turn>|fix] [args]"
)
_HISTORY_WEBCLEAR_USAGE = (
    "Usage: /history "
    "[show|detail <turn>|save|load|clear [last]|rewind <turn>|fix|webclear] [args]"
)
_TURN_ERROR_MESSAGES: dict[HistoryTurnAction, dict[HistoryTurnError, str]] = {
    "detail": {
        "missing": "Turn number required for /history detail.",
        "invalid": "Turn number must be an integer.",
    },
    "rewind": {
        "missing": "Turn number required for /history rewind.",
        "invalid": "Turn number must be an integer.",
    },
}


async def handle_history(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    remainder = strip_to_none(arguments) or ""
    intent = parse_current_agent_history_intent(remainder)
    action_handler = _HISTORY_ACTION_HANDLERS.get(intent.action)
    if action_handler is not None:
        return await action_handler(handler, intent)

    webclear_enabled = history_handlers.web_tools_enabled_for_agent(handler._get_current_agent())
    return _unknown_history_action_response(
        raw_subcommand=intent.raw_subcommand or "",
        webclear_enabled=webclear_enabled,
    )


async def _handle_history_webclear_command(
    handler: "SlashCommandHandler",
    *,
    target_agent: str | None,
) -> str:
    agent_name = target_agent or handler.current_agent_name
    agent = handler.instance.agents.get(agent_name)
    if target_agent is not None and agent is None:
        return _missing_agent_response(heading="# history webclear", agent_name=agent_name)
    webclear_enabled = history_handlers.web_tools_enabled_for_agent(agent)
    if not webclear_enabled:
        return "\n".join(
            [
                "# history",
                "",
                "Unknown /history action: webclear",
                _history_usage(webclear_enabled=False),
            ]
        )
    return await handle_history_webclear(handler, target_agent=target_agent)


async def _handle_history_overview_intent(
    handler: "SlashCommandHandler",
    intent: HistoryActionIntent,
) -> str:
    del intent
    return await render_history_overview(handler)


async def _handle_history_show_intent(
    handler: "SlashCommandHandler",
    intent: HistoryActionIntent,
) -> str:
    return await handle_show(handler, target_agent=intent.argument)


async def _handle_history_detail_intent(
    handler: "SlashCommandHandler",
    intent: HistoryActionIntent,
) -> str:
    return await handle_detail(
        handler,
        turn_index=intent.turn_index,
        turn_error=intent.turn_error,
    )


async def _handle_history_save_intent(
    handler: "SlashCommandHandler",
    intent: HistoryActionIntent,
) -> str:
    return await handle_save(handler, intent.argument)


async def _handle_history_load_intent(
    handler: "SlashCommandHandler",
    intent: HistoryActionIntent,
) -> str:
    return await handle_load(handler, intent.argument)


async def _handle_history_clear_all_intent(
    handler: "SlashCommandHandler",
    intent: HistoryActionIntent,
) -> str:
    return await handle_history_clear_all(handler, target_agent=intent.argument)


async def _handle_history_clear_last_intent(
    handler: "SlashCommandHandler",
    intent: HistoryActionIntent,
) -> str:
    return await handle_history_clear_last(handler, target_agent=intent.argument)


async def _handle_history_rewind_intent(
    handler: "SlashCommandHandler",
    intent: HistoryActionIntent,
) -> str:
    return await handle_history_rewind(
        handler,
        turn_index=intent.turn_index,
        turn_error=intent.turn_error,
    )


async def _handle_history_fix_intent(
    handler: "SlashCommandHandler",
    intent: HistoryActionIntent,
) -> str:
    return await handle_history_fix(handler, target_agent=intent.argument)


async def _handle_history_webclear_intent(
    handler: "SlashCommandHandler",
    intent: HistoryActionIntent,
) -> str:
    return await _handle_history_webclear_command(
        handler,
        target_agent=intent.argument,
    )


_HISTORY_ACTION_HANDLERS: dict[HistoryAction, _HistoryIntentHandler] = {
    "overview": _handle_history_overview_intent,
    "show": _handle_history_show_intent,
    "detail": _handle_history_detail_intent,
    "save": _handle_history_save_intent,
    "load": _handle_history_load_intent,
    "clear_all": _handle_history_clear_all_intent,
    "clear_last": _handle_history_clear_last_intent,
    "rewind": _handle_history_rewind_intent,
    "fix": _handle_history_fix_intent,
    "webclear": _handle_history_webclear_intent,
}


def _unknown_history_action_response(*, raw_subcommand: str, webclear_enabled: bool) -> str:
    return "\n".join(
        [
            "# history",
            "",
            f"Unknown /history action: {raw_subcommand}",
            _history_usage(webclear_enabled=webclear_enabled),
        ]
    )


def _history_usage(*, webclear_enabled: bool) -> str:
    return _HISTORY_WEBCLEAR_USAGE if webclear_enabled else _HISTORY_USAGE


def _missing_agent_response(*, heading: str, agent_name: str) -> str:
    return "\n".join(
        [
            heading,
            "",
            f"Unable to locate agent '{agent_name}' for this session.",
        ]
    )


def _turn_error_message(
    action: HistoryTurnAction,
    turn_error: HistoryTurnError | None,
) -> str | None:
    if turn_error is None:
        return None
    return _TURN_ERROR_MESSAGES[action][turn_error]


async def render_history_overview(handler: "SlashCommandHandler") -> str:
    heading = "# conversation history"
    _, error = handler._get_current_agent_or_error(heading)
    if error:
        return error

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


async def handle_show(handler: "SlashCommandHandler", target_agent: str | None = None) -> str:
    heading = "# history show"
    agent_name = target_agent or handler.current_agent_name
    agent = handler.instance.agents.get(agent_name)
    if agent is None:
        return _missing_agent_response(heading=heading, agent_name=agent_name)

    history = list(agent.message_history)
    report = build_history_turn_report(history)
    return render_history_turn_report_markdown(report, heading="history show")


async def handle_detail(
    handler: "SlashCommandHandler",
    *,
    turn_index: int | None,
    turn_error: HistoryTurnError | None,
) -> str:
    heading = "# history detail"

    _, error = handler._get_current_agent_or_error(
        heading,
        missing_template=f"Unable to locate agent '{handler.current_agent_name}' for this session.",
    )
    if error:
        return error

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await history_handlers.handle_history_review(
        ctx,
        agent_name=handler.current_agent_name,
        turn_index=turn_index,
        error=_turn_error_message("detail", turn_error),
    )
    return handler._format_outcome_as_markdown(outcome, "history detail", io=io)


async def handle_history_clear_all(
    handler: "SlashCommandHandler",
    *,
    target_agent: str | None = None,
) -> str:
    heading = "# history clear"
    agent_name = target_agent or handler.current_agent_name
    if agent_name not in handler.instance.agents:
        return _missing_agent_response(heading=heading, agent_name=agent_name)

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await history_handlers.handle_history_clear_all(
        ctx,
        agent_name=handler.current_agent_name,
        target_agent=target_agent,
    )
    return handler._format_outcome_as_markdown(outcome, "history clear", io=io)


async def handle_history_clear_last(
    handler: "SlashCommandHandler",
    *,
    target_agent: str | None = None,
) -> str:
    heading = "# history clear last"
    agent_name = target_agent or handler.current_agent_name
    if agent_name not in handler.instance.agents:
        return _missing_agent_response(heading=heading, agent_name=agent_name)

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await history_handlers.handle_history_clear_last(
        ctx,
        agent_name=handler.current_agent_name,
        target_agent=target_agent,
    )
    return handler._format_outcome_as_markdown(outcome, "history clear last", io=io)


async def handle_history_rewind(
    handler: "SlashCommandHandler",
    *,
    turn_index: int | None,
    turn_error: HistoryTurnError | None,
) -> str:
    heading = "# history rewind"
    _, error = handler._get_current_agent_or_error(
        heading,
        missing_template=f"Unable to locate agent '{handler.current_agent_name}' for this session.",
    )
    if error:
        return error

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await history_handlers.handle_history_rewind(
        ctx,
        agent_name=handler.current_agent_name,
        turn_index=turn_index,
        error=_turn_error_message("rewind", turn_error),
    )
    return handler._format_outcome_as_markdown(outcome, "history rewind", io=io)


async def handle_history_fix(
    handler: "SlashCommandHandler",
    *,
    target_agent: str | None = None,
) -> str:
    heading = "# history fix"
    agent_name = target_agent or handler.current_agent_name
    if agent_name not in handler.instance.agents:
        return _missing_agent_response(heading=heading, agent_name=agent_name)

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await history_handlers.handle_history_fix(
        ctx,
        agent_name=handler.current_agent_name,
        target_agent=target_agent,
    )
    return handler._format_outcome_as_markdown(outcome, "history fix", io=io)


async def handle_save(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    heading = "# save conversation"

    _, error = handler._get_current_agent_or_error(
        heading,
        missing_template=f"Unable to locate agent '{handler.current_agent_name}' for this session.",
    )
    if error:
        return error

    filename = strip_to_none(arguments)

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

    _, error = handler._get_current_agent_or_error(
        heading,
        missing_template=f"Unable to locate agent '{handler.current_agent_name}' for this session.",
    )
    if error:
        return error

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    filename = strip_to_none(arguments)
    error_message = None
    if not filename:
        error_message = "Filename required for /history load."
    else:
        file_path = Path(filename)
        if not file_path.is_absolute() and ctx.session_cwd is not None:
            file_path = Path(ctx.session_cwd) / file_path
        if not file_path.exists():
            error_message = f"File not found: {filename}"
        else:
            filename = str(file_path)

    outcome = await history_handlers.handle_history_load(
        ctx,
        agent_name=handler.current_agent_name,
        filename=filename,
        error=error_message,
    )
    return handler._format_outcome_as_markdown(outcome, "load conversation", io=io)


async def handle_history_webclear(
    handler: "SlashCommandHandler",
    *,
    target_agent: str | None = None,
) -> str:
    heading = "# history webclear"

    agent_name = target_agent or handler.current_agent_name
    if agent_name not in handler.instance.agents:
        return _missing_agent_response(heading=heading, agent_name=agent_name)

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await history_handlers.handle_history_webclear(
        ctx,
        agent_name=handler.current_agent_name,
        target_agent=target_agent,
    )
    return handler._format_outcome_as_markdown(outcome, "history webclear", io=io)
