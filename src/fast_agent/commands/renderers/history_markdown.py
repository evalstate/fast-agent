"""Markdown renderers for history summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.commands.history_summaries import HistoryOverview


def render_history_overview_markdown(
    overview: "HistoryOverview",
    *,
    heading: str,
) -> str:
    lines = [f"# {heading}", ""]
    lines.append(
        "Messages: "
        f"{overview.message_count} (user: {overview.user_message_count}, "
        f"assistant: {overview.assistant_message_count})"
    )
    lines.append(
        "Tool Calls: "
        f"{overview.tool_calls} (successes: {overview.tool_successes}, "
        f"errors: {overview.tool_errors})"
    )

    if overview.recent_messages:
        lines.append("")
        lines.append(f"Recent {len(overview.recent_messages)} messages:")
        for message in overview.recent_messages:
            lines.append(f"- {message.role}: {message.snippet}")
    else:
        lines.append("")
        lines.append("No messages yet.")

    return "\n".join(lines)
