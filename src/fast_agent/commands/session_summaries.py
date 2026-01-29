"""Session summary helpers for ACP and CLI rendering."""

from __future__ import annotations

from dataclasses import dataclass

from fast_agent.session import (
    SessionEntrySummary,
    build_session_entry_summaries,
    format_session_entries,
    get_session_history_window,
    get_session_manager,
)


@dataclass(slots=True)
class SessionListSummary:
    entries: list[str]
    usage: str
    entry_summaries: list[SessionEntrySummary]


def build_session_list_summary() -> SessionListSummary:
    manager = get_session_manager()
    sessions = manager.list_sessions()
    limit = get_session_history_window()
    if limit > 0:
        sessions = sessions[:limit]

    current = manager.current_session
    current_name = current.info.name if current else None
    entries = format_session_entries(
        sessions,
        current_name,
        mode="compact",
    )
    entry_summaries = build_session_entry_summaries(
        sessions,
        current_name,
        summary_limit=None,
    )
    return SessionListSummary(
        entries=entries,
        usage="Usage: /session resume <id|number>",
        entry_summaries=entry_summaries,
    )
