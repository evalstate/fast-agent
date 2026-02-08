"""Session management for fast-agent."""

from .formatting import (
    SessionEntrySummary,
    SessionListMode,
    build_session_entry_summaries,
    extract_session_title,
    format_history_summary,
    format_session_entries,
)
from .session_manager import (
    Session,
    SessionInfo,
    SessionManager,
    apply_session_window,
    display_session_name,
    get_session_history_window,
    get_session_manager,
    is_session_pinned,
    reset_session_manager,
    summarize_session_histories,
)

__all__ = [
    "Session",
    "SessionInfo",
    "SessionManager",
    "apply_session_window",
    "display_session_name",
    "SessionListMode",
    "SessionEntrySummary",
    "build_session_entry_summaries",
    "extract_session_title",
    "format_history_summary",
    "format_session_entries",
    "get_session_history_window",
    "get_session_manager",
    "is_session_pinned",
    "reset_session_manager",
    "summarize_session_histories",
]
