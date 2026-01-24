"""Markdown renderers for session summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.commands.session_summaries import SessionListSummary


def render_session_list_markdown(
    summary: "SessionListSummary",
    *,
    heading: str,
) -> str:
    lines = [f"# {heading}", ""]

    if not summary.entries:
        lines.extend(["No sessions found.", "", summary.usage])
        return "\n".join(lines)

    lines.extend(summary.entries)
    lines.extend(["", summary.usage])
    return "\n".join(lines)
