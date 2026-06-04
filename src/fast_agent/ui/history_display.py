"""Display helpers for agent conversation history."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from shutil import get_terminal_size
from typing import TYPE_CHECKING

from rich import print as rich_print
from rich.text import Text

from fast_agent.commands.history_summaries import build_history_turn_report
from fast_agent.types.conversation_summary import ConversationSummary
from fast_agent.ui import history_display_rows as _row_extraction
from fast_agent.ui.history_display_models import (
    HistoryChromeBar,
    HistoryDisplayRow,
    HistoryTimelineEntry,
)
from fast_agent.utils.count_display import format_count
from fast_agent.utils.timing_display import format_duration_ms, format_rate_per_second

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rich.console import Console

    from fast_agent.commands.history_summaries import HistoryTurnSummary
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.types import PromptMessageExtended


NON_TEXT_MARKER = "^"
TIMELINE_WIDTH = 20
SUMMARY_COUNT = 12
ROLE_COLUMN_WIDTH = 17

_SHADE_BLOCK_THRESHOLDS: tuple[tuple[int, str, str], ...] = (
    (50, "░", "dim {color}"),
    (200, "▒", "dim {color}"),
    (500, "▒", "{color}"),
    (2000, "▓", "{color}"),
)


@dataclass(frozen=True, slots=True)
class _SummaryTextWidths:
    preview: int
    detail: int


@dataclass(frozen=True, slots=True)
class _OverviewColumns:
    total_width: int
    show_time: bool
    show_chars: bool


class Colours:
    """Central colour palette for history display output."""

    USER = "blue"
    ASSISTANT = "green"
    TOOL = "magenta"
    TOOL_ERROR = "red"
    HEADER = USER
    TIMELINE_EMPTY = "dim default"
    CONTEXT_SAFE = "green"
    CONTEXT_CAUTION = "yellow"
    CONTEXT_ALERT = "bright_red"
    TOOL_DETAIL = "dim magenta"


def _build_history_rows(history: Sequence[PromptMessageExtended]) -> list[HistoryDisplayRow]:
    return _row_extraction.build_history_rows(history)


def _extract_tool_result_summary(result, *, limit: int = 80):
    return _row_extraction._extract_tool_result_summary(result, limit=limit)


def _message_role(message: PromptMessageExtended) -> str:
    return _row_extraction._message_role(message)


def _ensure_text(value: object | None) -> Text:
    """Coerce various value types into a Rich Text instance."""

    if isinstance(value, Text):
        return value.copy()
    if value is None:
        return Text("")
    if isinstance(value, str):
        return Text(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Text)):
        return Text(", ".join(str(item) for item in value if item))
    return Text(str(value))


def _truncate_text_segment(segment: Text, width: int) -> Text:
    if width <= 0 or segment.cell_len == 0:
        return Text("")
    if segment.cell_len <= width:
        return segment.copy()
    truncated = segment.copy()
    truncated.truncate(width, overflow="ellipsis")
    return truncated


def _summary_marker(include_non_text: bool) -> Text:
    marker_component = Text()
    if include_non_text:
        marker_component.append(" ")
        marker_component.append(NON_TEXT_MARKER, style="dim")
    return marker_component


def _append_detail_if_present(combined: Text, detail: Text | None) -> None:
    if detail and detail.cell_len > 0:
        if combined.cell_len > 0:
            combined.append(" ")
        combined.append_text(detail)


def _compose_unbounded_summary_text(
    preview: Text,
    detail: Text | None,
    marker_component: Text,
) -> Text:
    combined = Text()
    combined.append_text(preview)
    _append_detail_if_present(combined, detail)
    combined.append_text(marker_component)
    return combined


def _minimum_detail_width(detail: Text) -> int:
    detail_plain = detail.plain
    for prefix in ("tool→", "result→"):
        if detail_plain.startswith(prefix):
            return min(detail.cell_len, len(prefix))
    return 1


def _fit_summary_widths(
    *,
    preview_len: int,
    detail_len: int,
    width_after_marker: int,
    min_detail_width: int,
) -> _SummaryTextWidths:
    preview_allow = min(preview_len, width_after_marker)
    detail_allow = min(detail_len, max(0, width_after_marker - preview_allow))

    if detail_allow < min_detail_width:
        needed = min_detail_width - detail_allow
        reduction = min(preview_allow, needed)
        preview_allow -= reduction
        detail_allow += reduction

    space = 1 if preview_allow > 0 and detail_allow > 0 else 0
    total = preview_allow + detail_allow + space
    if total > width_after_marker:
        overflow = total - width_after_marker
        reduction = min(preview_allow, overflow)
        preview_allow -= reduction
        overflow -= reduction
        if overflow > 0:
            detail_allow = max(0, detail_allow - overflow)

    return _SummaryTextWidths(
        preview=max(0, preview_allow),
        detail=max(0, min(detail_allow, detail_len)),
    )


def _summary_text_widths(
    preview: Text,
    detail: Text,
    width_after_marker: int,
) -> _SummaryTextWidths:
    if detail.cell_len <= 0 or width_after_marker <= 0:
        return _SummaryTextWidths(
            preview=min(preview.cell_len, width_after_marker),
            detail=0,
        )

    return _fit_summary_widths(
        preview_len=preview.cell_len,
        detail_len=detail.cell_len,
        width_after_marker=width_after_marker,
        min_detail_width=_minimum_detail_width(detail),
    )


def _compose_bounded_summary_text(
    preview: Text,
    detail: Text,
    marker_component: Text,
    max_width: int,
) -> Text:
    if max_width <= 0:
        return Text("")
    if marker_component.cell_len > max_width:
        marker_component = Text("")
    marker_width = marker_component.cell_len
    widths = _summary_text_widths(
        preview,
        detail,
        max(0, max_width - marker_width),
    )

    preview_segment = _truncate_text_segment(preview, widths.preview)
    detail_segment = (
        _truncate_text_segment(detail, widths.detail) if widths.detail > 0 else Text("")
    )

    combined = Text()
    combined.append_text(preview_segment)
    if preview_segment.cell_len > 0 and detail_segment.cell_len > 0:
        combined.append(" ")
    combined.append_text(detail_segment)

    if marker_component.cell_len > 0 and combined.cell_len + marker_component.cell_len <= max_width:
        combined.append_text(marker_component)

    return combined


def _compose_summary_text(
    preview: Text,
    detail: Text | None,
    *,
    include_non_text: bool,
    max_width: int | None,
) -> Text:
    marker_component = _summary_marker(include_non_text)
    if max_width is None:
        return _compose_unbounded_summary_text(preview, detail, marker_component)

    return _compose_bounded_summary_text(
        preview,
        detail.copy() if detail else Text(""),
        marker_component,
        max_width,
    )


def format_chars(value: int) -> str:
    if value <= 0:
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _aggregate_timeline_entries(
    rows: Sequence[HistoryDisplayRow],
) -> list[HistoryTimelineEntry]:
    return [
        HistoryTimelineEntry(
            role=row.timeline_role,
            chars=row.chars,
            non_text=row.non_text,
            is_error=row.is_error,
        )
        for row in rows
        if row.include_in_timeline
    ]


def _get_role_color(role: str, *, is_error: bool = False) -> str:
    """Get the display color for a role, accounting for error states."""
    color_map = {"user": Colours.USER, "assistant": Colours.ASSISTANT, "tool": Colours.TOOL}

    if role == "tool" and is_error:
        return Colours.TOOL_ERROR

    return color_map.get(role, "white")


def _shade_block(chars: int, *, non_text: bool, color: str) -> Text:
    if non_text:
        return Text(NON_TEXT_MARKER, style=f"bold {color}")
    if chars <= 0:
        return Text("·", style="dim")
    for max_chars, marker, style_template in _SHADE_BLOCK_THRESHOLDS:
        if chars < max_chars:
            return Text(marker, style=style_template.format(color=color))
    return Text("█", style=f"bold {color}")


def _build_history_bar(
    entries: Sequence[HistoryTimelineEntry],
    width: int = TIMELINE_WIDTH,
) -> HistoryChromeBar:
    recent = list(entries[-width:])
    bar = Text(" history |", style="dim")
    for entry in recent:
        color = _get_role_color(entry.role, is_error=entry.is_error)
        bar.append_text(
            _shade_block(entry.chars, non_text=entry.non_text, color=color)
        )
    remaining = width - len(recent)
    if remaining > 0:
        bar.append("░" * remaining, style=Colours.TIMELINE_EMPTY)
    bar.append("|", style="dim")

    detail = Text(format_count(len(entries), "turn"), style="dim")
    return HistoryChromeBar(bar=bar, detail=detail)


def _build_context_bar_line(
    current: int,
    window: int | None,
    width: int = TIMELINE_WIDTH,
) -> HistoryChromeBar:
    bar = Text(" context |", style="dim")

    if not window or window <= 0:
        bar.append("░" * width, style=Colours.TIMELINE_EMPTY)
        bar.append("|", style="dim")
        detail = Text(f"{format_chars(current)} tokens (unknown window)", style="dim")
        return HistoryChromeBar(bar=bar, detail=detail)

    if current <= 0:
        bar.append("░" * width, style=Colours.TIMELINE_EMPTY)
        bar.append("|", style="dim")
        bar.append(" pending", style="dim")
        detail = Text(f"pending / {format_chars(window)} →", style="dim")
        return HistoryChromeBar(bar=bar, detail=detail)

    percent = current / window if window else 0.0
    filled = min(width, round(min(percent, 1.0) * width))

    def color_for(pct: float) -> str:
        if pct >= 0.9:
            return Colours.CONTEXT_ALERT
        if pct >= 0.7:
            return Colours.CONTEXT_CAUTION
        return Colours.CONTEXT_SAFE

    color = color_for(percent)
    if filled > 0:
        bar.append("█" * filled, style=color)
    if filled < width:
        bar.append("░" * (width - filled), style=Colours.TIMELINE_EMPTY)
    bar.append("|", style="dim")
    bar.append(f" {percent * 100:5.1f}%", style="dim")
    if percent > 1.0:
        bar.append(f" +{(percent - 1) * 100:.0f}%", style="bold bright_red")

    detail = Text(f"{format_chars(current)} / {format_chars(window)} →", style="dim")
    return HistoryChromeBar(bar=bar, detail=detail)


def _render_header_line(agent_name: str, *, console: Console | None, printer) -> None:
    header = Text()
    header.append("▎", style=Colours.HEADER)
    header.append(" [ 1] ", style=Colours.HEADER)
    header.append(str(agent_name), style=f"bold {Colours.USER}")

    line = Text()
    line.append_text(header)
    line.append(" ")

    try:
        total_width = console.width if console else get_terminal_size().columns
    except Exception:
        total_width = 80

    separator_width = max(1, total_width - line.cell_len)
    line.append("─" * separator_width, style="dim")

    printer("")
    printer(line)
    printer("")


def _render_statistics(
    summary: ConversationSummary,
    *,
    printer,
) -> None:
    """Render compact conversation statistics section."""

    llm_time = format_duration_ms(
        summary.total_elapsed_time_ms if summary.total_elapsed_time_ms > 0 else None
    )
    runtime = format_duration_ms(
        summary.conversation_span_ms if summary.conversation_span_ms > 0 else None
    )

    # Build compact statistics lines
    stats_lines = []

    if summary.total_elapsed_time_ms > 0 or summary.conversation_span_ms > 0:
        timing_line = Text("  ", style="dim")
        timing_line.append("LLM Time: ", style="dim")
        timing_line.append(llm_time, style="default")
        timing_line.append("  •  ", style="dim")
        timing_line.append("Runtime: ", style="dim")
        timing_line.append(runtime, style="default")
        stats_lines.append(timing_line)

    tool_counts = Text("  ", style="dim")
    tool_counts.append("Tool Calls: ", style="dim")
    tool_counts.append(str(summary.tool_calls), style="default")
    if summary.tool_calls > 0:
        tool_counts.append(
            f" (successes: {summary.tool_successes}, errors: {summary.tool_errors})", style="dim"
        )
    stats_lines.append(tool_counts)

    # Tool Usage Breakdown (if tools were used)
    if summary.tool_calls > 0 and summary.tool_call_map:
        # Get top tools sorted by count
        sorted_tools = sorted(summary.tool_call_map.items(), key=lambda x: x[1], reverse=True)

        # Show compact breakdown
        tool_details = Text("  ", style="dim")
        tool_details.append("Tools: ", style="dim")

        tool_parts = []
        for tool_name, count in sorted_tools[:5]:  # Show max 5 tools
            tool_parts.append(f"{tool_name} ({count})")

        tool_details.append(", ".join(tool_parts), style=Colours.TOOL_DETAIL)
        stats_lines.append(tool_details)

    # Print all statistics lines
    for line in stats_lines:
        printer(line)

    printer("")


def _render_turn_statistics(
    *,
    turn_count: int,
    total_turn_time_ms: float,
    total_tool_time_ms: float,
    average_ttft_ms: float | None,
    average_response_ms: float | None,
    average_tps: float | None,
    printer,
) -> None:
    summary_line = Text("  ", style="dim")
    summary_line.append("Turns: ", style="dim")
    summary_line.append(str(turn_count), style="default")
    summary_line.append("  •  ", style="dim")
    summary_line.append("Turn Time: ", style="dim")
    summary_line.append(
        format_duration_ms(total_turn_time_ms if total_turn_time_ms > 0 else None),
        style="default",
    )
    summary_line.append("  •  ", style="dim")
    summary_line.append("Tool Time: ", style="dim")
    summary_line.append(
        format_duration_ms(total_tool_time_ms if total_tool_time_ms > 0 else None),
        style="default",
    )
    printer(summary_line)

    detail_line = Text("  ", style="dim")
    detail_line.append("Avg TTFT: ", style="dim")
    detail_line.append(format_duration_ms(average_ttft_ms), style="default")
    detail_line.append("  •  ", style="dim")
    detail_line.append("Avg Resp: ", style="dim")
    detail_line.append(format_duration_ms(average_response_ms), style="default")
    detail_line.append("  •  ", style="dim")
    detail_line.append("Avg TPS: ", style="dim")
    detail_line.append(format_rate_per_second(average_tps), style="default")
    printer(detail_line)
    printer("")


def _format_turn_report_row(turn: "HistoryTurnSummary", *, preview_width: int) -> Text:
    turn_preview = Text()
    turn_preview.append(turn.user_snippet, style=Colours.USER)
    turn_preview.append(" → ", style="dim")
    turn_preview.append(turn.assistant_snippet, style=Colours.ASSISTANT)
    preview_text = _truncate_text_segment(turn_preview, preview_width)

    line = Text(" ")
    line.append(f"{turn.turn_index:>2}", style="dim")
    line.append(" ")
    line.append_text(preview_text)
    if preview_text.cell_len < preview_width:
        line.append(" " * (preview_width - preview_text.cell_len))
    line.append(f" {format_duration_ms(turn.turn_time_ms):>7}", style="dim")
    line.append(f" {format_duration_ms(turn.tool_time_ms):>7}", style="dim")
    line.append(f" {format_duration_ms(turn.ttft_ms):>7}", style="dim")
    line.append(f" {format_duration_ms(turn.response_ms):>7}", style="dim")
    line.append(f" {format_rate_per_second(turn.tps):>6}", style="dim")
    return line


def _append_padded_detail_segment(
    line: Text,
    *,
    label_width: int,
    available_width: int,
    detail: Text,
) -> None:
    line.append(" " * label_width, style="dim")
    line.append_text(detail)
    if available_width > detail.cell_len:
        line.append(" " * (available_width - detail.cell_len), style="dim")


def _render_history_chrome(
    history: Sequence[PromptMessageExtended],
    usage_accumulator: "UsageAccumulator" | None,
    *,
    printer,
) -> None:
    rows = _build_history_rows(history)
    timeline_entries = _aggregate_timeline_entries(rows)

    history_bar = _build_history_bar(timeline_entries)
    if usage_accumulator:
        current_tokens = usage_accumulator.current_context_tokens
        window = usage_accumulator.context_window_size
    else:
        current_tokens = 0
        window = None
    context_bar = _build_context_bar_line(current_tokens, window)

    gap = Text("   ")
    combined_line = Text()
    combined_line.append_text(history_bar.bar)
    combined_line.append_text(gap)
    combined_line.append_text(context_bar.bar)
    printer(combined_line)

    history_label_len = len(" history |")
    context_label_len = len(" context |")

    history_available = history_bar.bar.cell_len - history_label_len
    context_available = context_bar.bar.cell_len - context_label_len

    detail_line = Text()
    _append_padded_detail_segment(
        detail_line,
        label_width=history_label_len,
        available_width=history_available,
        detail=history_bar.detail,
    )
    detail_line.append_text(gap)
    _append_padded_detail_segment(
        detail_line,
        label_width=context_label_len,
        available_width=context_available,
        detail=context_bar.detail,
    )
    printer(detail_line)

    printer("")
    printer(
        Text(
            " " + "─" * (history_bar.bar.cell_len + context_bar.bar.cell_len + gap.cell_len),
            style="dim",
        )
    )


def _terminal_width(console: Console | None, *, fallback: int) -> int:
    try:
        return console.width if console else get_terminal_size().columns
    except Exception:
        return fallback


def _overview_columns(console: Console | None) -> _OverviewColumns:
    total_width = _terminal_width(console, fallback=80)
    return _OverviewColumns(
        total_width=total_width,
        show_time=total_width >= 60,
        show_chars=total_width >= 50,
    )


def _overview_summary_rows(
    rows: Sequence[HistoryDisplayRow],
) -> tuple[list[HistoryDisplayRow], int]:
    summary_candidates = [row for row in rows if not row.hide_summary]
    summary_rows = summary_candidates[-SUMMARY_COUNT:]
    start_index = len(summary_candidates) - len(summary_rows) + 1
    return summary_rows, start_index


def _render_overview_table_header(columns: _OverviewColumns, printer) -> None:
    header_line = Text(" ")
    header_line.append(" #", style="dim")
    header_line.append(" ", style="dim")
    header_line.append(f"    {'Role':<{ROLE_COLUMN_WIDTH}}", style="dim")
    if columns.show_time:
        header_line.append(f" {'Time':>7}", style="dim")
    if columns.show_chars:
        header_line.append(f" {'Chars':>7}", style="dim")
    header_line.append("  ", style="dim")
    header_line.append("Summary", style="dim")
    printer(header_line)


def _overview_role_label(row: HistoryDisplayRow) -> tuple[str, str, str]:
    role_arrows = {"user": "▶", "assistant": "◀", "tool": "▶"}
    role_labels = {"user": "user", "assistant": "assistant", "tool": "tool result"}
    role = row.role
    arrow = row.arrow or role_arrows.get(role, "▶")
    label = row.label or role_labels.get(role, role)
    if role == "assistant" and row.has_tool_request:
        label = f"{label}*"
    return role, arrow, label


def _overview_detail_text(row: HistoryDisplayRow) -> Text | None:
    details = row.details
    detail_text = _ensure_text(details) if details else Text("")
    return detail_text if detail_text.cell_len > 0 else None


def _format_overview_row(
    *,
    row: HistoryDisplayRow,
    index: int,
    columns: _OverviewColumns,
) -> Text:
    role, arrow, label = _overview_role_label(row)
    color = _get_role_color(role, is_error=row.is_error)
    chars = row.chars
    block = _shade_block(chars, non_text=row.non_text, color=color)

    line = Text(" ")
    line.append(f"{index:>2}", style="dim")
    line.append(" ")
    line.append_text(block)
    line.append(" ")
    line.append(arrow, style=color)
    line.append(" ")
    line.append(f"{label:<{ROLE_COLUMN_WIDTH}}", style=color)
    if columns.show_time:
        line.append(f" {format_duration_ms(row.timing_ms):>7}", style="dim")
    if columns.show_chars:
        line.append(f" {format_chars(chars):>7}", style="dim")
    line.append("  ")

    summary_text = _compose_summary_text(
        _ensure_text(row.preview),
        _overview_detail_text(row),
        include_non_text=row.non_text,
        max_width=max(0, columns.total_width - line.cell_len),
    )
    line.append_text(summary_text)
    return line


def _render_overview_rows(
    rows: Sequence[HistoryDisplayRow],
    *,
    console: Console | None,
    printer,
) -> None:
    summary_rows, start_index = _overview_summary_rows(rows)
    columns = _overview_columns(console)
    _render_overview_table_header(columns, printer)
    for offset, row in enumerate(summary_rows):
        printer(
            _format_overview_row(
                row=row,
                index=start_index + offset,
                columns=columns,
            )
        )
    printer("")


def display_history_overview(
    agent_name: str,
    history: Sequence[PromptMessageExtended],
    usage_accumulator: "UsageAccumulator" | None = None,
    *,
    console: Console | None = None,
) -> None:
    if not history:
        printer = console.print if console else rich_print
        printer("[dim]No conversation history yet[/dim]")
        return

    printer = console.print if console else rich_print

    # Create conversation summary for statistics
    summary = ConversationSummary(messages=list(history))
    rows = _build_history_rows(history)

    # Render conversation statistics
    _render_header_line(agent_name, console=console, printer=printer)
    _render_statistics(summary, printer=printer)
    _render_history_chrome(
        history,
        usage_accumulator,
        printer=printer,
    )
    _render_overview_rows(rows, console=console, printer=printer)


def display_history_show(
    agent_name: str,
    history: Sequence[PromptMessageExtended],
    usage_accumulator: "UsageAccumulator" | None = None,
    *,
    console: Console | None = None,
) -> None:
    if not history:
        printer = console.print if console else rich_print
        printer("[dim]No conversation history yet[/dim]")
        return

    printer = console.print if console else rich_print

    turn_report = build_history_turn_report(list(history))
    _render_header_line(agent_name, console=console, printer=printer)
    _render_turn_statistics(
        turn_count=turn_report.turn_count,
        total_turn_time_ms=turn_report.total_turn_time_ms,
        total_tool_time_ms=turn_report.total_tool_time_ms,
        average_ttft_ms=turn_report.average_ttft_ms,
        average_response_ms=turn_report.average_response_ms,
        average_tps=turn_report.average_tps,
        printer=printer,
    )
    _render_history_chrome(
        history,
        usage_accumulator,
        printer=printer,
    )

    if not turn_report.turns:
        printer("[dim]No user turns yet[/dim]")
        printer("")
        return

    total_width = _terminal_width(console, fallback=100)

    fixed_columns = 3 + 8 + 8 + 8 + 8 + 7
    preview_width = max(24, total_width - fixed_columns - 10)

    header_line = Text(" ")
    header_line.append(f"{'#':>2}", style="dim")
    header_line.append(" ", style="dim")
    header_line.append(f"{'Turn':<{preview_width}}", style="dim")
    header_line.append(f" {'Turn':>7}", style="dim")
    header_line.append(f" {'Tool':>7}", style="dim")
    header_line.append(f" {'TTFT':>7}", style="dim")
    header_line.append(f" {'Resp':>7}", style="dim")
    header_line.append(f" {'TPS':>6}", style="dim")
    printer(header_line)

    for turn in turn_report.turns:
        printer(_format_turn_report_row(turn, preview_width=preview_width))

    printer("")
