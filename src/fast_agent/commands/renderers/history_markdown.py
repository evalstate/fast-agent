"""Markdown renderers for history summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.commands.renderers.markdown_blocks import markdown_heading
from fast_agent.utils.count_display import format_count, format_count_breakdown
from fast_agent.utils.markdown import escape_markdown_table_cell, escape_markdown_text
from fast_agent.utils.text import strip_to_none
from fast_agent.utils.timing_display import format_duration_ms, format_rate_per_second

if TYPE_CHECKING:
    from fast_agent.commands.history_summaries import (
        HistoryMessageSnippet,
        HistoryOverview,
        HistoryTurnReport,
        HistoryTurnSummary,
    )


def _format_labeled_parts(parts: dict[str, str]) -> str:
    return ", ".join(f"{label} {value}" for label, value in parts.items())


def _format_positive_duration_ms(value: float) -> str:
    return format_duration_ms(value if value > 0 else None)


def _format_turn_table_row(turn: "HistoryTurnSummary") -> str:
    turn_text = escape_markdown_table_cell(
        f"{turn.user_snippet} → {turn.assistant_snippet}"
    )
    return (
        "| "
        f"{turn.turn_index} | "
        f"{turn_text} | "
        f"{format_duration_ms(turn.turn_time_ms)} | "
        f"{format_duration_ms(turn.tool_time_ms)} | "
        f"{format_duration_ms(turn.ttft_ms)} | "
        f"{format_duration_ms(turn.response_ms)} | "
        f"{format_rate_per_second(turn.tps)} |"
    )


def _format_recent_message_line(message: "HistoryMessageSnippet") -> str:
    role = strip_to_none(message.role) or "unknown"
    snippet = strip_to_none(message.snippet)
    if snippet is None:
        return f"- {escape_markdown_text(role)}:"
    return f"- {escape_markdown_text(role)}: {escape_markdown_text(snippet)}"


def render_history_overview_markdown(
    overview: "HistoryOverview",
    *,
    heading: str,
) -> str:
    lines = [markdown_heading(heading), ""]
    lines.append(
        format_count_breakdown(
            "Messages",
            overview.message_count,
            user=overview.user_message_count,
            assistant=overview.assistant_message_count,
        )
    )
    lines.append(
        format_count_breakdown(
            "Tool Calls",
            overview.tool_calls,
            successes=overview.tool_successes,
            errors=overview.tool_errors,
        )
    )

    if overview.recent_messages:
        lines.append("")
        lines.append(f"Recent {format_count(len(overview.recent_messages), 'message')}:")
        lines.extend(_format_recent_message_line(message) for message in overview.recent_messages)
    else:
        lines.append("")
        lines.append("No messages yet.")

    return "\n".join(lines)


def render_history_turn_report_markdown(
    report: "HistoryTurnReport",
    *,
    heading: str,
) -> str:
    lines = [markdown_heading(heading), ""]

    lines.append(f"Turns: {report.turn_count}")
    lines.append(
        format_count_breakdown(
            "Tools",
            report.total_tool_calls,
            errors=report.total_tool_errors,
        )
    )
    lines.append(
        "Totals: "
        + _format_labeled_parts(
            {
                "turn": _format_positive_duration_ms(report.total_turn_time_ms),
                "llm": _format_positive_duration_ms(report.total_llm_time_ms),
                "tool": _format_positive_duration_ms(report.total_tool_time_ms),
            }
        )
    )
    lines.append(
        "Averages: "
        + _format_labeled_parts(
            {
                "turn": format_duration_ms(report.average_turn_time_ms),
                "tool": format_duration_ms(report.average_tool_time_ms),
                "ttft": format_duration_ms(report.average_ttft_ms),
                "resp": format_duration_ms(report.average_response_ms),
                "tps": format_rate_per_second(report.average_tps),
            }
        )
    )

    if not report.turns:
        lines.extend(["", "No user turns yet."])
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "| # | Turn | Time | Tool | TTFT | Resp | TPS |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    lines.extend(_format_turn_table_row(turn) for turn in report.turns)

    return "\n".join(lines)
