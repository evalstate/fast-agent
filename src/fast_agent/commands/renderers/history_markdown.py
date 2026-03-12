"""Markdown renderers for history summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.commands.history_summaries import HistoryOverview, HistoryTurnReport


def _format_time(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1000:
        return f"{value:.0f}ms"
    return f"{value / 1000:.1f}s"


def _format_tps(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def _escape_table_cell(value: str) -> str:
    return value.replace("|", r"\|")


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


def render_history_turn_report_markdown(
    report: "HistoryTurnReport",
    *,
    heading: str,
) -> str:
    lines = [f"# {heading}", ""]

    lines.append(f"Turns: {report.turn_count}")
    lines.append(
        "Tools: "
        f"{report.total_tool_calls} (errors: {report.total_tool_errors})"
    )
    lines.append(
        "Totals: "
        f"turn {_format_time(report.total_turn_time_ms if report.total_turn_time_ms > 0 else None)}"
        f", llm {_format_time(report.total_llm_time_ms if report.total_llm_time_ms > 0 else None)}"
        f", tool {_format_time(report.total_tool_time_ms if report.total_tool_time_ms > 0 else None)}"
    )
    lines.append(
        "Averages: "
        f"turn {_format_time(report.average_turn_time_ms)}, "
        f"tool {_format_time(report.average_tool_time_ms)}, "
        f"ttft {_format_time(report.average_ttft_ms)}, "
        f"resp {_format_time(report.average_response_ms)}, "
        f"tps {_format_tps(report.average_tps)}"
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

    for turn in report.turns:
        lines.append(
            "| "
            f"{turn.turn_index} | "
            f"{_escape_table_cell(f'{turn.user_snippet} → {turn.assistant_snippet}')} | "
            f"{_format_time(turn.turn_time_ms)} | "
            f"{_format_time(turn.tool_time_ms)} | "
            f"{_format_time(turn.ttft_ms)} | "
            f"{_format_time(turn.response_ms)} | "
            f"{_format_tps(turn.tps)} |"
        )

    return "\n".join(lines)
