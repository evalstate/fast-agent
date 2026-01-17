"""Shared session list formatting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from fast_agent.session.session_manager import display_session_name

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .session_manager import SessionInfo

SessionListMode = Literal["compact", "verbose"]


def format_session_entries(
    sessions: Iterable[SessionInfo],
    current_session_name: str | None,
    *,
    mode: SessionListMode,
) -> list[str]:
    """Format session entries for display in CLI or ACP outputs."""
    session_list = list(sessions)
    if not session_list:
        return []

    max_index_width = len(str(len(session_list)))
    lines: list[str] = []

    for index, session_info in enumerate(session_list, 1):
        is_current = current_session_name == session_info.name if current_session_name else False
        index_str = f"{index}.".rjust(max_index_width + 1)

        display_name = display_session_name(session_info.name)

        if mode == "compact":
            separator = " \u25b6 " if is_current else " - "
            timestamp = session_info.last_activity.strftime("%b %d %H:%M")
            line = f"{index_str} {display_name}{separator}{timestamp}"
            metadata = session_info.metadata or {}
            summary = (
                metadata.get("title")
                or metadata.get("label")
                or metadata.get("first_user_preview")
                or ""
            )
            summary = " ".join(str(summary).split())

            history_map = metadata.get("last_history_by_agent")
            if isinstance(history_map, dict) and history_map:
                agent_names = sorted(history_map.keys())
                if len(agent_names) > 1:
                    display_names = agent_names
                    if len(agent_names) > 3:
                        display_names = agent_names[:3] + [f"+{len(agent_names) - 3}"]
                    agent_label = ", ".join(display_names)
                    line = f"{line} - {len(agent_names)} agents: {agent_label}"

            if summary:
                line = f"{line} - {summary[:30]}"

            lines.append(line)
            continue

        current_marker = " \U0001F7E2" if is_current else ""
        created = session_info.created_at.strftime("%Y-%m-%d %H:%M")
        last_activity = session_info.last_activity.strftime("%Y-%m-%d %H:%M")
        history_count = len(session_info.history_files)
        lines.append(f"  {index_str} {display_name}{current_marker}")
        lines.append(
            f"     Created: {created} | Last: {last_activity} | Histories: {history_count}"
        )

    return lines
