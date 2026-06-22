"""Session slash command handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, cast

from fast_agent.commands.handlers import session_export as session_export_handlers
from fast_agent.commands.handlers import sessions as sessions_handlers
from fast_agent.commands.renderers.session_markdown import render_session_list_markdown
from fast_agent.commands.results import CommandOutcome
from fast_agent.commands.session_export_help import render_session_export_help_markdown
from fast_agent.commands.session_summaries import FULL_SESSION_USAGE
from fast_agent.commands.shared_command_intents import (
    SessionAction,
    SessionCommandIntent,
    parse_session_command_intent,
    should_default_export_agent,
)
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from fast_agent.acp.command_io import ACPCommandIO
    from fast_agent.acp.slash_commands import SlashCommandHandler

_SessionActionHandler = Callable[["SlashCommandHandler", SessionCommandIntent], Awaitable[str]]


def _unknown_session_action_response(intent: SessionCommandIntent) -> str:
    raw_subcommand = intent.raw_subcommand or intent.argument or ""
    return "\n".join(
        [
            "# session",
            "",
            f"Unknown /session action: {raw_subcommand}",
            FULL_SESSION_USAGE,
        ]
    )


def _invalid_session_arguments_response(intent: SessionCommandIntent) -> str:
    return "\n".join(
        [
            "# session",
            "",
            f"Invalid /session arguments: {intent.argument or 'parse error'}",
            FULL_SESSION_USAGE,
        ]
    )


async def handle_session(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    if handler._noenv:
        return "\n".join(
            [
                "# session",
                "",
                "Session commands are disabled in --noenv mode.",
            ]
        )

    remainder = strip_to_none(arguments) or ""
    intent = parse_session_command_intent(remainder)
    return await _SESSION_ACTION_HANDLERS[intent.action](handler, intent)


async def _handle_session_list_intent(
    handler: "SlashCommandHandler",
    intent: SessionCommandIntent,
) -> str:
    del intent
    return render_session_list(handler)


async def _handle_session_new_intent(
    handler: "SlashCommandHandler",
    intent: SessionCommandIntent,
) -> str:
    return await handle_session_new(handler, intent.argument)


async def _handle_session_resume_intent(
    handler: "SlashCommandHandler",
    intent: SessionCommandIntent,
) -> str:
    return await handle_session_resume(handler, intent.argument)


async def _handle_session_title_intent(
    handler: "SlashCommandHandler",
    intent: SessionCommandIntent,
) -> str:
    return await handle_session_title(handler, intent.argument)


async def _handle_session_fork_intent(
    handler: "SlashCommandHandler",
    intent: SessionCommandIntent,
) -> str:
    return await handle_session_fork(handler, intent.argument)


async def _handle_session_delete_intent(
    handler: "SlashCommandHandler",
    intent: SessionCommandIntent,
) -> str:
    return await handle_session_delete(handler, intent.argument)


async def _handle_session_pin_intent(
    handler: "SlashCommandHandler",
    intent: SessionCommandIntent,
) -> str:
    return await handle_session_pin(
        handler,
        value=intent.pin_value,
        target=intent.pin_target,
    )


async def _handle_session_export_intent(
    handler: "SlashCommandHandler",
    intent: SessionCommandIntent,
) -> str:
    return await handle_session_export(handler, intent)


async def _handle_session_error_intent(
    handler: "SlashCommandHandler",
    intent: SessionCommandIntent,
) -> str:
    del handler
    return _invalid_session_arguments_response(intent)


async def _handle_unknown_session_intent(
    handler: "SlashCommandHandler",
    intent: SessionCommandIntent,
) -> str:
    del handler
    return _unknown_session_action_response(intent)


_SESSION_ACTION_HANDLERS: dict[SessionAction, _SessionActionHandler] = {
    "help": _handle_session_list_intent,
    "list": _handle_session_list_intent,
    "new": _handle_session_new_intent,
    "resume": _handle_session_resume_intent,
    "title": _handle_session_title_intent,
    "fork": _handle_session_fork_intent,
    "delete": _handle_session_delete_intent,
    "pin": _handle_session_pin_intent,
    "export": _handle_session_export_intent,
    "error": _handle_session_error_intent,
    "unknown": _handle_unknown_session_intent,
}


def render_session_list(handler: "SlashCommandHandler") -> str:
    if handler._noenv:
        return "\n".join(
            [
                "# sessions",
                "",
                "Session commands are disabled in --noenv mode.",
            ]
        )
    ctx = handler._build_command_context()
    if ctx.session_runtime is None:
        return "\n".join(
            [
                "# sessions",
                "",
                "Session commands are unavailable in this context.",
            ]
        )
    summary = ctx.session_runtime.build_list_summary()
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
    title = strip_to_none(argument)
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
        title=strip_to_none(argument),
    )
    return handler._format_outcome_as_markdown(outcome, "session fork", io=io)


async def handle_session_new(handler: "SlashCommandHandler", argument: str | None) -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await sessions_handlers.handle_create_session(
        ctx,
        session_name=strip_to_none(argument),
        session_id=handler.session_id,
        replace_existing=True,
    )
    sessions_handlers.apply_session_new_history_reset(ctx, outcome, logger=handler._logger)
    return handler._format_outcome_as_markdown(outcome, "session new", io=io)


async def handle_session_delete(handler: "SlashCommandHandler", argument: str | None) -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await sessions_handlers.handle_clear_sessions(
        ctx,
        target=strip_to_none(argument),
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


async def handle_session_export(handler: "SlashCommandHandler", intent) -> str:
    if intent.export_help:
        return render_session_export_help_markdown()

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    current_session_id = (
        ctx.session_runtime.active_session_id(fallback_session_id=handler.session_id)
        if ctx.session_runtime is not None
        else None
    )
    if intent.export_target is None and current_session_id is None:
        outcome = CommandOutcome()
        outcome.add_message(
            "No active session to export.",
            channel="error",
            right_info="session",
        )
        return handler._format_outcome_as_markdown(outcome, "session export", io=io)
    agent_name = intent.export_agent
    if agent_name is None and should_default_export_agent(
        intent.export_target,
        current_session_id=current_session_id,
    ):
        agent_name = handler.current_agent_name
    outcome = await session_export_handlers.handle_session_export(
        ctx,
        target=intent.export_target,
        agent_name=agent_name,
        output_path=intent.export_output,
        hf_url=intent.export_hf_url,
        hf_dataset=intent.export_hf_dataset,
        hf_dataset_path=intent.export_hf_dataset_path,
        privacy_filter=intent.export_privacy_filter,
        privacy_filter_path=intent.export_privacy_filter_path,
        download_privacy_filter=intent.export_download_privacy_filter,
        privacy_filter_device=intent.export_privacy_filter_device,
        privacy_filter_variant=intent.export_privacy_filter_variant,
        show_redactions=intent.export_show_redactions,
        current_session_id=current_session_id,
        error=intent.export_error,
    )
    return handler._format_outcome_as_markdown(outcome, "session export", io=io)
