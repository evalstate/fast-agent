"""Shared session command handlers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from rich.text import Text

from fast_agent.commands.handlers._text_formatting import indexed_row, resolve_terminal_width
from fast_agent.commands.handlers.shared import clear_agent_histories
from fast_agent.commands.results import CommandOutcome
from fast_agent.commands.session_summaries import build_session_list_summary
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.session import display_session_name, format_session_agent_label
from fast_agent.session.preview import find_last_assistant_preview_text
from fast_agent.ui.shell_notice import format_shell_notice
from fast_agent.utils.action_normalization import normalize_action_token, parse_boolean_alias
from fast_agent.utils.count_display import format_count
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext
    from fast_agent.core.logging.logger import Logger
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session import ResumeSessionAgentsResult, Session, SessionEntrySummary
    from fast_agent.session.session_manager import SessionManager


NOENV_SESSION_MESSAGE = "Session commands are disabled in --noenv mode."
_PIN_TOGGLE_VALUES = {"", "toggle"}
_PIN_USAGE = "Usage: /session pin [on|off|id|number]"


@dataclass(frozen=True, slots=True)
class _PinState:
    desired: bool | None = None
    error: str | None = None


def _noenv_outcome() -> CommandOutcome:
    outcome = CommandOutcome()
    outcome.add_message(NOENV_SESSION_MESSAGE, channel="warning", right_info="session")
    return outcome


def _append_session_metadata(line: Text, items: list[tuple[str, str]]) -> None:
    for value, style in items:
        line.append(" \u2022 ", style="dim")
        line.append(value, style=style)


def _truncate_summary(summary: str, available: int) -> str | None:
    if available <= 0:
        return None
    if len(summary) <= available:
        return summary
    if available == 1:
        return summary[:available]
    return summary[: max(0, available - 1)].rstrip() + "…"


def _resolve_pin_state(value: str | None, *, current: bool) -> _PinState:
    normalized = normalize_action_token(value)
    if normalized in _PIN_TOGGLE_VALUES:
        return _PinState(desired=not current)
    desired = parse_boolean_alias(normalized, numeric=False)
    if desired is not None:
        return _PinState(desired=desired)
    return _PinState(error=_PIN_USAGE)


def _strip_wrapping_quotes(value: str | None) -> str | None:
    text = strip_to_none(value)
    if text is None:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = strip_to_none(text[1:-1]) or ""
    return strip_to_none(text)


def _session_for_pin(
    manager: "SessionManager",
    outcome: CommandOutcome,
    *,
    target: str | None,
) -> "Session | None":
    target = strip_to_none(target)
    if target:
        resolved = manager.resolve_session_name(target)
        session = manager.get_session(resolved) if resolved else None
        if session is None:
            outcome.add_message(
                f"Session not found: {target}",
                channel="error",
                right_info="session",
            )
        return session

    session = manager.current_session
    if session is not None:
        return session

    sessions = manager.list_sessions()
    if sessions:
        return manager.get_session(sessions[0].name)

    outcome.add_message(
        "No session available to pin.",
        channel="warning",
        right_info="session",
    )
    return None


def _build_session_entries(entries: list[SessionEntrySummary], *, usage: str) -> Text:
    content = Text()
    content.append_text(Text("Sessions:", style="bold"))
    content.append("\n\n")
    terminal_width = resolve_terminal_width()
    bullet_sep = " • "
    for entry in entries:
        line = indexed_row(entry.index)
        name_style = "bold yellow" if entry.is_pinned else "bright_blue bold"
        line.append(entry.display_name, style=name_style)

        if entry.is_current:
            line.append(" ", style="dim")
            line.append("▶", style="bright_green")
            line.append(" ", style="dim")
            line.append(entry.timestamp, style="dim")
        else:
            line.append(bullet_sep, style="dim")
            line.append(entry.timestamp, style="dim")

        metadata_items: list[tuple[str, str]] = []
        agent_label = format_session_agent_label(entry)
        if agent_label:
            metadata_items.append((agent_label, "dim"))

        if entry.is_pinned:
            line.append(bullet_sep, style="dim")
            line.append("(pin)", style="dim")

        if metadata_items:
            _append_session_metadata(line, metadata_items)

        if entry.summary:
            summary_sep = " " if entry.is_pinned and not metadata_items else bullet_sep
            remaining = terminal_width - line.cell_len - len(summary_sep)
            summary_text = _truncate_summary(entry.summary, remaining)
            if summary_text:
                line.append(summary_sep, style="dim")
                line.append(summary_text, style="white")

        content.append_text(line)
        content.append("\n")

    content.append("\n")
    content.append_text(Text(usage, style="dim"))
    return content



async def handle_create_session(
    ctx: CommandContext,
    *,
    session_name: str | None,
    session_id: str | None = None,
    replace_existing: bool = False,
) -> CommandOutcome:
    if ctx.noenv:
        return _noenv_outcome()

    outcome = CommandOutcome()

    manager = ctx.resolve_session_manager()
    session_name = _strip_wrapping_quotes(session_name)
    metadata = {"title": session_name} if session_name else None
    if session_id:
        if replace_existing:
            manager.delete_session(session_id)
        session = manager.create_session_with_id(session_id, metadata=metadata)
    else:
        session = manager.create_session(session_name)
    label = session.info.metadata.get("title") or session.info.name
    outcome.add_message(f"Created session: {label}", channel="info", right_info="session")
    return outcome


def apply_session_new_history_reset(
    ctx: CommandContext,
    outcome: CommandOutcome,
    *,
    logger: "Logger | None" = None,
) -> None:
    """Clear in-memory histories after starting a new conversation session."""
    cleared = clear_agent_histories(ctx.agent_provider.registered_agents(), logger)
    if cleared:
        outcome.add_message(
            f"Cleared agent history: {', '.join(sorted(cleared))}",
            channel="info",
        )


async def handle_list_sessions(
    ctx: CommandContext,
    *,
    show_help: bool = False,
) -> CommandOutcome:
    if ctx.noenv:
        return _noenv_outcome()

    outcome = CommandOutcome()
    summary = build_session_list_summary(
        manager=ctx.resolve_session_manager(),
        show_help=show_help,
    )
    if not summary.entries:
        outcome.add_message("No sessions found.", channel="warning", right_info="session")
        if show_help:
            outcome.add_message(Text(summary.usage, style="dim"), right_info="session")
        return outcome

    outcome.add_message(
        _build_session_entries(summary.entry_summaries, usage=summary.usage),
        right_info="session",
    )
    return outcome


async def handle_pin_session(
    ctx: CommandContext,
    *,
    value: str | None,
    target: str | None,
) -> CommandOutcome:
    if ctx.noenv:
        return _noenv_outcome()

    outcome = CommandOutcome()
    from fast_agent.session import is_session_pinned

    manager = ctx.resolve_session_manager()
    session = _session_for_pin(manager, outcome, target=target)
    if session is None:
        return outcome

    current = is_session_pinned(session.info)
    pin_state = _resolve_pin_state(value, current=current)
    if pin_state.desired is None:
        outcome.add_message(
            pin_state.error or "Usage: /session pin [on|off|id|number]",
            channel="warning",
        )
        return outcome

    session.set_pinned(pin_state.desired)
    label = display_session_name(session.info.name)
    action = "Pinned" if pin_state.desired else "Unpinned"
    outcome.add_message(
        f"{action} session: {label}",
        channel="info",
        right_info="session",
    )
    return outcome


async def handle_clear_sessions(
    ctx: CommandContext,
    *,
    target: str | None,
) -> CommandOutcome:
    if ctx.noenv:
        return _noenv_outcome()

    outcome = CommandOutcome()
    from fast_agent.session import apply_session_window

    target = strip_to_none(target)
    if not target:
        outcome.add_message(
            "Usage: /session delete <id|number|all>",
            channel="warning",
            right_info="session",
        )
        return outcome

    manager = ctx.resolve_session_manager()
    if normalize_action_token(target) == "all":
        all_sessions = manager.list_sessions()
        if not all_sessions:
            outcome.add_message("No sessions found.", channel="warning", right_info="session")
            return outcome
        deleted = 0
        for session_info in all_sessions:
            if manager.delete_session(session_info.name):
                deleted += 1
        outcome.add_message(
            f"Deleted {format_count(deleted, 'session')}.",
            channel="info",
            right_info="session",
        )
        return outcome

    sessions = apply_session_window(manager.list_sessions())
    target_name = target
    if target.isdigit():
        ordinal = int(target)
        if ordinal <= 0 or ordinal > len(sessions):
            outcome.add_message(f"Session not found: {target}", channel="error")
            return outcome
        target_name = sessions[ordinal - 1].name

    if manager.delete_session(target_name):
        outcome.add_message(f"Deleted session: {target_name}", channel="info")
    else:
        outcome.add_message(f"Session not found: {target}", channel="error")
    return outcome


def _add_resume_result_messages(
    outcome: CommandOutcome,
    result: ResumeSessionAgentsResult,
) -> None:
    loaded = result.loaded
    session = result.session
    if loaded:
        loaded_list = ", ".join(sorted(loaded.keys()))
        outcome.add_message(
            f"Resumed session: {session.info.name} ({loaded_list})",
            channel="info",
            right_info="session",
        )
    else:
        outcome.add_message(
            f"Resumed session: {session.info.name} (no history yet)",
            channel="warning",
            right_info="session",
        )

    if result.missing_agents:
        missing_list = ", ".join(sorted(result.missing_agents))
        outcome.add_message(
            f"Missing agents from session: {missing_list}",
            channel="warning",
            right_info="session",
        )

    for warning in result.warnings:
        if warning.code != "missing-agent":
            outcome.add_message(
                warning.message,
                channel="warning",
                right_info="session",
            )

    for usage_notice in result.usage_notices:
        outcome.add_message(
            usage_notice,
            channel="warning",
            right_info="session",
        )


def _add_available_history_summary(
    outcome: CommandOutcome,
    result: ResumeSessionAgentsResult,
) -> None:
    if not result.missing_agents and result.loaded:
        return

    from fast_agent.session import (
        format_history_summary,
        summarize_session_histories,
    )

    summary = summarize_session_histories(result.session)
    summary_text = format_history_summary(summary)
    if summary_text:
        outcome.add_message(
            Text(f"Available histories: {summary_text}", style="dim"),
            right_info="session",
        )


def _active_resume_agent(
    ctx: CommandContext,
    outcome: CommandOutcome,
    *,
    requested_agent_name: str,
    active_agent_name: str,
) -> AgentProtocol:
    if active_agent_name == requested_agent_name:
        return cast("AgentProtocol", ctx.agent_provider._agent(requested_agent_name))

    outcome.switch_agent = active_agent_name
    outcome.add_message(
        f"Switched to agent: {active_agent_name}",
        channel="info",
        right_info="session",
    )
    return cast("AgentProtocol", ctx.agent_provider._agent(active_agent_name))


def _add_shell_notice(outcome: CommandOutcome, agent_obj: AgentProtocol) -> None:
    if isinstance(agent_obj, McpAgentProtocol) and agent_obj.shell_runtime_enabled:
        notice = format_shell_notice(agent_obj.shell_access_modes, agent_obj.shell_runtime)
        outcome.add_message(notice, right_info="session")


def _ensure_usage_model(agent_obj: AgentProtocol) -> None:
    usage = agent_obj.usage_accumulator
    if not usage or usage.model is not None:
        return

    llm = agent_obj.llm
    model_name = llm.model_name if llm is not None else None
    if not model_name:
        model_name = agent_obj.config.model
    if model_name:
        usage.model = model_name


async def _display_resumed_history(
    ctx: CommandContext,
    outcome: CommandOutcome,
    agent_obj: AgentProtocol,
) -> None:
    usage = agent_obj.usage_accumulator
    history = list(agent_obj.message_history)
    await ctx.io.display_history_overview(agent_obj.name, history, usage)

    assistant_text = find_last_assistant_preview_text(history)
    if assistant_text:
        outcome.add_message(
            Text(assistant_text),
            title="Last assistant message",
            right_info="session",
            agent_name=agent_obj.name,
            render_markdown=True,
        )


async def handle_resume_session(
    ctx: CommandContext,
    *,
    agent_name: str,
    session_id: str | None,
) -> CommandOutcome:
    if ctx.noenv:
        return _noenv_outcome()

    outcome = CommandOutcome()
    manager = ctx.resolve_session_manager()
    agents_map = cast("Mapping[str, AgentProtocol]", ctx.agent_provider.registered_agents())
    if not isinstance(agents_map, Mapping):
        outcome.add_message(
            "Session resume is unavailable in this context.",
            channel="error",
            right_info="session",
        )
        return outcome

    fallback_agent_name = ctx.agent_provider.resolve_target_agent_name(agent_name)

    result = await manager.resume_session_agents_async(
        agents_map,
        session_id,
        fallback_agent_name=fallback_agent_name,
    )

    if not result:
        if session_id:
            outcome.add_message(f"Session not found: {session_id}", channel="error")
        else:
            outcome.add_message("No sessions found.", channel="warning")
        return outcome

    active_agent_name = result.active_agent or agent_name
    _add_resume_result_messages(outcome, result)
    _add_available_history_summary(outcome, result)

    agent_obj = _active_resume_agent(
        ctx,
        outcome,
        requested_agent_name=agent_name,
        active_agent_name=active_agent_name,
    )
    _add_shell_notice(outcome, agent_obj)
    _ensure_usage_model(agent_obj)
    await _display_resumed_history(ctx, outcome, agent_obj)
    return outcome


async def handle_title_session(
    ctx: CommandContext,
    *,
    title: str | None,
    session_id: str | None = None,
) -> CommandOutcome:
    if ctx.noenv:
        return _noenv_outcome()

    outcome = CommandOutcome()
    title = _strip_wrapping_quotes(title)
    if not title:
        outcome.add_message("Usage: /session title <text>", channel="error")
        return outcome

    manager = ctx.resolve_session_manager()
    session = manager.current_session
    if session_id:
        if session is None or session.info.name != session_id:
            session = manager.create_session_with_id(session_id)
    elif session is None:
        session = manager.create_session()
    if session is None:
        outcome.add_message("No session available to title.", channel="warning", right_info="session")
        return outcome
    session.set_title(title)
    outcome.add_message(f"Session title set: {title}", channel="info", right_info="session")
    return outcome


async def handle_fork_session(
    ctx: CommandContext,
    *,
    title: str | None,
) -> CommandOutcome:
    if ctx.noenv:
        return _noenv_outcome()

    outcome = CommandOutcome()
    manager = ctx.resolve_session_manager()
    title = _strip_wrapping_quotes(title)
    forked = manager.fork_current_session(title=title)
    if forked is None:
        outcome.add_message("No session available to fork.", channel="warning", right_info="session")
        return outcome
    label = forked.info.metadata.get("title") or forked.info.name
    outcome.add_message(f"Forked session: {label}", channel="info", right_info="session")
    return outcome
