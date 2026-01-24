"""Session summary helpers for ACP and CLI rendering."""

from __future__ import annotations

from dataclasses import dataclass

from fast_agent.session import (
    format_session_entries,
    get_session_history_window,
    get_session_manager,
)


@dataclass(slots=True)
class SessionListSummary:
    entries: list[str]
    usage: str


def build_session_list_summary() -> SessionListSummary:
    manager = get_session_manager()
    sessions = manager.list_sessions()
    limit = get_session_history_window()
    if limit > 0:
        sessions = sessions[:limit]

    current = manager.current_session
    entries = format_session_entries(
        sessions,
        current.info.name if current else None,
        mode="compact",
    )
    return SessionListSummary(entries=entries, usage="Usage: /session resume <id|number>")
