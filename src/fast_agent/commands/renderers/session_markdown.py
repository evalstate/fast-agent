"""Markdown renderers for session summaries."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from fast_agent.commands.renderers.markdown_blocks import markdown_heading
from fast_agent.utils.markdown import escape_markdown_text
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from fast_agent.commands.session_summaries import SessionListSummary
    from fast_agent.session import SessionEntrySummary


def _format_session_entry(entry: str, summary: "SessionEntrySummary") -> str:
    if not summary.is_pinned:
        return escape_markdown_text(entry)

    display_name = strip_to_none(summary.display_name)
    if display_name is None:
        return escape_markdown_text(entry)

    match = re.match(
        rf"^(?P<prefix>\s*{summary.index}\.\s+)(?P<name>{re.escape(display_name)})(?=$|\s)",
        entry,
    )
    if match is None:
        return escape_markdown_text(entry)

    return (
        escape_markdown_text(match.group("prefix"))
        + f"**{escape_markdown_text(match.group('name'))}**"
        + escape_markdown_text(entry[match.end("name") :])
    )


def render_session_list_markdown(
    summary: "SessionListSummary",
    *,
    heading: str,
) -> str:
    lines = [markdown_heading(heading), ""]

    if not summary.entries:
        lines.extend(["No sessions found.", "", summary.usage])
        return "\n".join(lines)

    lines.extend(
        _format_session_entry(entry, entry_summary)
        for entry, entry_summary in zip(
            summary.entries, summary.entry_summaries, strict=False
        )
    )
    lines.extend(["", summary.usage])
    return "\n".join(lines)
