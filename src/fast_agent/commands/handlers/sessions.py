"""Shared session command handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.commands.results import CommandOutcome
from fast_agent.commands.session_summaries import build_session_list_summary
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.ui.shell_notice import format_shell_notice

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext


def _build_session_entries(entries: list[str]) -> Text:
    content = Text()
    content.append("Sessions:\n")
    for line in entries:
        content.append_text(Text(line))
        content.append("\n")
    content.append_text(Text("Usage: /session resume <id|number>", style="dim"))
    return content


async def handle_create_session(
    ctx: CommandContext,
    *,
    session_name: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    from fast_agent.session import get_session_manager

    manager = get_session_manager()
    session = manager.create_session(session_name)
    label = session.info.metadata.get("title") or session.info.name
    outcome.add_message(f"Created session: {label}", channel="info", right_info="session")
    return outcome


async def handle_list_sessions(ctx: CommandContext) -> CommandOutcome:
    outcome = CommandOutcome()
    summary = build_session_list_summary()
    if not summary.entries:
        outcome.add_message("No sessions found.", channel="warning", right_info="session")
        return outcome

    outcome.add_message(_build_session_entries(summary.entries), right_info="session")
    return outcome


async def handle_clear_sessions(
    ctx: CommandContext,
    *,
    target: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    from fast_agent.session import get_session_history_window, get_session_manager

    if not target:
        outcome.add_message(
            "Usage: /session clear <id|number|all>",
            channel="warning",
            right_info="session",
        )
        return outcome

    manager = get_session_manager()
    if target.lower() == "all":
        all_sessions = manager.list_sessions()
        if not all_sessions:
            outcome.add_message("No sessions found.", channel="warning", right_info="session")
            return outcome
        deleted = 0
        for session_info in all_sessions:
            if manager.delete_session(session_info.name):
                deleted += 1
        outcome.add_message(
            f"Deleted {deleted} session(s).",
            channel="info",
            right_info="session",
        )
        return outcome

    sessions = manager.list_sessions()
    limit = get_session_history_window()
    if limit > 0:
        sessions = sessions[:limit]
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


async def handle_resume_session(
    ctx: CommandContext,
    *,
    agent_name: str,
    session_id: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    from fast_agent.session import (
        format_history_summary,
        get_session_manager,
        summarize_session_histories,
    )

    agent_obj = ctx.agent_provider._agent(agent_name)

    manager = get_session_manager()
    agents_map = getattr(ctx.agent_provider, "_agents", None)
    if not isinstance(agents_map, Mapping):
        outcome.add_message(
            "Session resume is unavailable in this context.",
            channel="error",
            right_info="session",
        )
        return outcome

    result = manager.resume_session_agents(
        agents_map,
        session_id,
        default_agent_name=getattr(agent_obj, "name", None),
    )

    if not result:
        if session_id:
            outcome.add_message(f"Session not found: {session_id}", channel="error")
        else:
            outcome.add_message("No sessions found.", channel="warning")
        return outcome

    session, loaded, missing_agents = result
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

    if isinstance(agent_obj, McpAgentProtocol) and agent_obj.shell_runtime_enabled:
        notice = format_shell_notice(agent_obj.shell_access_modes, agent_obj.shell_runtime)
        outcome.add_message(notice, right_info="session")

    if missing_agents:
        missing_list = ", ".join(sorted(missing_agents))
        outcome.add_message(
            f"Missing agents from session: {missing_list}",
            channel="warning",
            right_info="session",
        )

    if missing_agents or not loaded:
        summary = summarize_session_histories(session)
        summary_text = format_history_summary(summary)
        if summary_text:
            outcome.add_message(
                Text(f"Available histories: {summary_text}", style="dim"),
                right_info="session",
            )

    if len(loaded) == 1:
        loaded_agent = next(iter(loaded.keys()))
        if loaded_agent != agent_name:
            outcome.switch_agent = loaded_agent
            agent_obj = ctx.agent_provider._agent(loaded_agent)
            outcome.add_message(
                f"Switched to agent: {loaded_agent}",
                channel="info",
                right_info="session",
            )

    usage = getattr(agent_obj, "usage_accumulator", None)
    if usage and usage.model is None:
        llm = getattr(agent_obj, "llm", None)
        model_name = getattr(llm, "model_name", None)
        if not model_name:
            model_name = getattr(getattr(agent_obj, "config", None), "model", None)
        if model_name:
            usage.model = model_name

    history = getattr(agent_obj, "message_history", [])
    await ctx.io.display_history_overview(agent_obj.name, list(history), usage)
    return outcome


async def handle_title_session(
    ctx: CommandContext,
    *,
    title: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    if not title:
        outcome.add_message("Usage: /session title <text>", channel="error")
        return outcome

    from fast_agent.session import get_session_manager

    manager = get_session_manager()
    session = manager.current_session
    if session is None:
        session = manager.create_session()
    session.set_title(title)
    outcome.add_message(f"Session title set: {title}", channel="info", right_info="session")
    return outcome


async def handle_fork_session(
    ctx: CommandContext,
    *,
    title: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    from fast_agent.session import get_session_manager

    manager = get_session_manager()
    forked = manager.fork_current_session(title=title)
    if forked is None:
        outcome.add_message("No session available to fork.", channel="warning", right_info="session")
        return outcome
    label = forked.info.metadata.get("title") or forked.info.name
    outcome.add_message(f"Forked session: {label}", channel="info", right_info="session")
    return outcome
