"""Shared session list formatting helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from fast_agent.session.session_manager import display_session_name, is_session_pinned
from fast_agent.utils.count_display import format_count

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .session_manager import SessionInfo


class SessionListMode(StrEnum):
    COMPACT = "compact"
    VERBOSE = "verbose"


def _metadata_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized or None


def extract_session_title(metadata: Mapping[str, object] | None) -> str | None:
    """Extract a display-friendly session title from metadata."""
    if not isinstance(metadata, Mapping):
        return None
    for key in ("title", "label", "first_user_preview"):
        title_text = _metadata_text(metadata.get(key))
        if title_text:
            return title_text
    return None


def _history_agent_names(metadata: Mapping[str, object]) -> list[str]:
    history_map = metadata.get("last_history_by_agent")
    if not isinstance(history_map, Mapping):
        return []
    return sorted(
        agent_name for key in history_map if (agent_name := _metadata_text(key)) is not None
    )


@dataclass(slots=True)
class SessionEntrySummary:
    index: int
    display_name: str
    is_current: bool
    is_pinned: bool
    timestamp: str
    agent_count: int | None = None
    agent_label: str | None = None
    summary: str | None = None


def format_session_agent_label(entry: SessionEntrySummary) -> str | None:
    if not entry.agent_count or not entry.agent_label:
        return None
    return f"{format_count(entry.agent_count, 'agent')}: {entry.agent_label}"


def build_session_entry_summaries(
    sessions: "Iterable[SessionInfo]",
    current_session_name: str | None,
    *,
    summary_limit: int | None = 30,
) -> list[SessionEntrySummary]:
    session_list = list(sessions)
    if not session_list:
        return []

    entries: list[SessionEntrySummary] = []
    for index, session_info in enumerate(session_list, 1):
        is_current = current_session_name == session_info.name if current_session_name else False
        display_name = display_session_name(session_info.name)
        timestamp = session_info.last_activity.strftime("%b %d %H:%M")
        pinned = is_session_pinned(session_info)

        metadata = session_info.metadata or {}
        summary_text = extract_session_title(metadata)
        if not summary_text:
            summary_value = None
        elif summary_limit is None:
            summary_value = summary_text
        else:
            summary_value = summary_text[:summary_limit]

        agent_count = None
        agent_label = None
        agent_names = _history_agent_names(metadata)
        if len(agent_names) > 1:
            display_names = agent_names
            if len(agent_names) > 3:
                display_names = [*agent_names[:3], f"+{len(agent_names) - 3}"]
            agent_count = len(agent_names)
            agent_label = ", ".join(display_names)

        entries.append(
            SessionEntrySummary(
                index=index,
                display_name=display_name,
                is_current=is_current,
                is_pinned=pinned,
                timestamp=timestamp,
                agent_count=agent_count,
                agent_label=agent_label,
                summary=summary_value,
            )
        )

    return entries


def format_session_entries(
    sessions: Iterable[SessionInfo],
    current_session_name: str | None,
    *,
    mode: SessionListMode | Literal["compact", "verbose"],
) -> list[str]:
    """Format session entries for display in CLI or ACP outputs."""
    list_mode = SessionListMode(mode)
    session_list = list(sessions)
    if not session_list:
        return []

    max_index_width = len(str(len(session_list)))
    lines: list[str] = []

    if list_mode is SessionListMode.COMPACT:
        for entry in build_session_entry_summaries(
            session_list,
            current_session_name,
            summary_limit=30,
        ):
            index_str = f"{entry.index}.".rjust(max_index_width + 1)
            separator = " \u25b6 " if entry.is_current else " - "
            line = f"{index_str} {entry.display_name}{separator}{entry.timestamp}"
            if entry.is_pinned:
                line = f"{line} - pin"
            agent_label = format_session_agent_label(entry)
            if agent_label:
                line = f"{line} - {agent_label}"
            if entry.summary:
                line = f"{line} - {entry.summary}"
            lines.append(line)
        return lines

    if list_mode is not SessionListMode.VERBOSE:
        raise ValueError(f"Unsupported session list mode: {list_mode}")

    for index, session_info in enumerate(session_list, 1):
        is_current = current_session_name == session_info.name if current_session_name else False
        index_str = f"{index}.".rjust(max_index_width + 1)

        display_name = display_session_name(session_info.name)

        current_marker = " \U0001f7e2" if is_current else ""
        pin_marker = " \U0001f4cc" if is_session_pinned(session_info) else ""
        created = session_info.created_at.strftime("%Y-%m-%d %H:%M")
        last_activity = session_info.last_activity.strftime("%Y-%m-%d %H:%M")
        history_count = len(session_info.history_files)
        lines.append(f"  {index_str} {display_name}{current_marker}{pin_marker}")
        lines.append(
            f"     Created: {created} | Last: {last_activity} | Histories: {history_count}"
        )

    return lines


def format_history_summary(summary: "Mapping[str, int]") -> str:
    """Format a history summary for display."""
    if not summary:
        return ""
    return ", ".join(
        f"{agent} ({format_count(count, 'message')})" for agent, count in sorted(summary.items())
    )
