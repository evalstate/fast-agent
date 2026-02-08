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

    for entry, entry_summary in zip(summary.entries, summary.entry_summaries, strict=False):
        if entry_summary.is_pinned:
            lines.append(entry.replace(entry_summary.display_name, f"**{entry_summary.display_name}**", 1))
        else:
            lines.append(entry)
    lines.extend(["", summary.usage])
    return "\n".join(lines)
