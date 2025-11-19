"""Display helpers for agent conversation history."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from shutil import get_terminal_size
from typing import TYPE_CHECKING

from rich import print as rich_print
from rich.text import Text

from fast_agent.constants import FAST_AGENT_TIMING, FAST_AGENT_TOOL_TIMING
from fast_agent.mcp.helpers.content_helpers import get_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rich.console import Console

    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.types import PromptMessageExtended


NON_TEXT_MARKER = "^"
TIMELINE_WIDTH = 20
SUMMARY_COUNT = 8
ROLE_COLUMN_WIDTH = 17


def _normalize_text(value: str | None) -> str:
    return "" if not value else " ".join(value.split())


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


def _char_count(value: str | None) -> int:
    return len(_normalize_text(value))


def _format_tool_detail(prefix: str, names: Sequence[str]) -> Text:
    detail = Text(prefix, style=Colours.TOOL_DETAIL)
    if names:
        detail.append(", ".join(names), style=Colours.TOOL_DETAIL)
    return detail


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


def _compose_summary_text(
    preview: Text,
    detail: Text | None,
    *,
    include_non_text: bool,
    max_width: int | None,
) -> Text:
    marker_component = Text()
    if include_non_text:
        marker_component.append(" ")
        marker_component.append(NON_TEXT_MARKER, style="dim")

    if max_width is None:
        combined = Text()
        combined.append_text(preview)
        if detail and detail.cell_len > 0:
            if combined.cell_len > 0:
                combined.append(" ")
            combined.append_text(detail)
        combined.append_text(marker_component)
        return combined

    width_available = max_width
    if width_available <= 0:
        return Text("")

    if marker_component.cell_len > width_available:
        marker_component = Text("")
    marker_width = marker_component.cell_len
    width_after_marker = max(0, width_available - marker_width)

    preview_len = preview.cell_len
    detail_component = detail.copy() if detail else Text("")
    detail_len = detail_component.cell_len
    detail_plain = detail_component.plain

    preview_allow = min(preview_len, width_after_marker)
    detail_allow = 0
    if detail_len > 0 and width_after_marker > 0:
        detail_allow = min(detail_len, max(0, width_after_marker - preview_allow))

        if width_after_marker > 0:
            min_detail_allow = 1
            for prefix in ("tool→", "result→"):
                if detail_plain.startswith(prefix):
                    min_detail_allow = min(detail_len, len(prefix))
                    break
        else:
            min_detail_allow = 0
        if detail_allow < min_detail_allow:
            needed = min_detail_allow - detail_allow
            reduction = min(preview_allow, needed)
            preview_allow -= reduction
            detail_allow += reduction

        preview_allow = max(0, preview_allow)
        detail_allow = max(0, min(detail_allow, detail_len))

        space = 1 if preview_allow > 0 and detail_allow > 0 else 0
        total = preview_allow + detail_allow + space
        if total > width_after_marker:
            overflow = total - width_after_marker
            reduction = min(preview_allow, overflow)
            preview_allow -= reduction
            overflow -= reduction
            if overflow > 0:
                detail_allow = max(0, detail_allow - overflow)

        preview_allow = max(0, preview_allow)
        detail_allow = max(0, min(detail_allow, detail_len))
    else:
        preview_allow = min(preview_len, width_after_marker)
        detail_allow = 0

    preview_segment = _truncate_text_segment(preview, preview_allow)
    detail_segment = (
        _truncate_text_segment(detail_component, detail_allow) if detail_allow > 0 else Text("")
    )

    combined = Text()
    combined.append_text(preview_segment)
    if preview_segment.cell_len > 0 and detail_segment.cell_len > 0:
        combined.append(" ")
    combined.append_text(detail_segment)

    if marker_component.cell_len > 0:
        if combined.cell_len + marker_component.cell_len <= max_width:
            combined.append_text(marker_component)

    return combined


def _preview_text(value: str | None, limit: int = 80) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return "<no text>"
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def _has_non_text_content(message: PromptMessageExtended) -> bool:
    for block in getattr(message, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type and block_type != "text":
            return True
    return False


def _extract_tool_result_summary(result, *, limit: int = 80) -> tuple[str, int, bool]:
    preview: str | None = None
    total_chars = 0
    saw_non_text = False

    for block in getattr(result, "content", []) or []:
        text = get_text(block)
        if text:
            normalized = _normalize_text(text)
            if preview is None:
                preview = _preview_text(normalized, limit=limit)
            total_chars += len(normalized)
        else:
            saw_non_text = True

    if preview is not None:
        return preview, total_chars, saw_non_text
    return f"{NON_TEXT_MARKER} non-text tool result", 0, True


def format_chars(value: int) -> str:
    if value <= 0:
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _extract_timing_ms(message: PromptMessageExtended) -> float | None:
    """Extract timing duration in milliseconds from message channels."""
    channels = getattr(message, "channels", None)
    if not channels:
        return None

    timing_blocks = channels.get(FAST_AGENT_TIMING, [])
    if not timing_blocks:
        return None

    timing_text = get_text(timing_blocks[0])
    if not timing_text:
        return None

    try:
        timing_data = json.loads(timing_text)
        return timing_data.get("duration_ms")
    except (json.JSONDecodeError, AttributeError, KeyError):
        return None


def _extract_tool_timings(message: PromptMessageExtended) -> dict[str, float]:
    """Extract tool timing data from message channels."""
    channels = getattr(message, "channels", None)
    if not channels:
        return {}

    timing_blocks = channels.get(FAST_AGENT_TOOL_TIMING, [])
    if not timing_blocks:
        return {}

    timing_text = get_text(timing_blocks[0])
    if not timing_text:
        return {}

    try:
        return json.loads(timing_text)
    except (json.JSONDecodeError, TypeError):
        return {}


def format_time(value: float | None) -> str:
    """Format timing value for display."""
    if value is None:
        return "-"
    if value < 1000:
        return f"{value:.0f}ms"
    return f"{value / 1000:.1f}s"


def _build_history_rows(history: Sequence[PromptMessageExtended]) -> list[dict]:
    rows: list[dict] = []
    call_name_lookup: dict[str, str] = {}

    for message in history:
        role_raw = getattr(message, "role", "assistant")
        role_value = getattr(role_raw, "value", role_raw)
        role = str(role_value).lower() if role_value else "assistant"

        text = ""
        if hasattr(message, "first_text"):
            try:
                text = message.first_text() or ""
            except Exception:  # pragma: no cover - defensive
                text = ""
        normalized_text = _normalize_text(text)
        chars = len(normalized_text)
        preview = _preview_text(text)
        non_text = _has_non_text_content(message) or chars == 0

        # Extract timing data
        timing_ms = _extract_timing_ms(message)
        tool_timings = _extract_tool_timings(message)

        tool_calls: Mapping[str, object] | None = getattr(message, "tool_calls", None)
        tool_results: Mapping[str, object] | None = getattr(message, "tool_results", None)

        detail_sections: list[Text] = []
        row_non_text = non_text
        has_tool_request = False
        hide_in_summary = False
        timeline_role = role
        include_in_timeline = True
        result_rows: list[dict] = []
        tool_result_total_chars = 0
        tool_result_has_non_text = False
        tool_result_has_error = False

        if tool_calls:
            names: list[str] = []
            for call_id, call in tool_calls.items():
                params = getattr(call, "params", None)
                name = getattr(params, "name", None) or getattr(call, "name", None) or call_id
                call_name_lookup[call_id] = name
                names.append(name)
            if names:
                detail_sections.append(_format_tool_detail("tool→", names))
                row_non_text = row_non_text and chars == 0  # treat call as activity
            has_tool_request = True
        if not normalized_text and tool_calls:
            preview = "(issuing tool request)"

        if tool_results:
            result_names: list[str] = []
            for call_id, result in tool_results.items():
                tool_name = call_name_lookup.get(call_id, call_id)
                result_names.append(tool_name)
                summary, result_chars, result_non_text = _extract_tool_result_summary(result)
                tool_result_total_chars += result_chars
                tool_result_has_non_text = tool_result_has_non_text or result_non_text
                detail = _format_tool_detail("result→", [tool_name])
                is_error = getattr(result, "isError", False)
                tool_result_has_error = tool_result_has_error or is_error
                # Get timing for this specific tool call
                tool_timing = tool_timings.get(call_id)
                result_rows.append(
                    {
                        "role": "tool",
                        "timeline_role": "tool",
                        "chars": result_chars,
                        "preview": summary,
                        "details": detail,
                        "non_text": result_non_text,
                        "has_tool_request": False,
                        "hide_summary": False,
                        "include_in_timeline": False,
                        "is_error": is_error,
                        "timing_ms": tool_timing,
                    }
                )
            if role == "user":
                timeline_role = "tool"
                hide_in_summary = True
            if result_names:
                detail_sections.append(_format_tool_detail("result→", result_names))

        if detail_sections:
            if len(detail_sections) == 1:
                details: Text | None = detail_sections[0]
            else:
                details = Text()
                for index, section in enumerate(detail_sections):
                    if index > 0:
                        details.append(" ")
                    details.append_text(section)
        else:
            details = None

        row_chars = chars
        if timeline_role == "tool" and tool_result_total_chars > 0:
            row_chars = tool_result_total_chars
        row_non_text = row_non_text or tool_result_has_non_text
        row_is_error = tool_result_has_error

        rows.append(
            {
                "role": role,
                "timeline_role": timeline_role,
                "chars": row_chars,
                "preview": preview,
                "details": details,
                "non_text": row_non_text,
                "has_tool_request": has_tool_request,
                "hide_summary": hide_in_summary,
                "include_in_timeline": include_in_timeline,
                "is_error": row_is_error,
                "timing_ms": timing_ms,
            }
        )
        rows.extend(result_rows)

    return rows


def _aggregate_timeline_entries(rows: Sequence[dict]) -> list[dict]:
    return [
        {
            "role": row.get("timeline_role", row["role"]),
            "chars": row["chars"],
            "non_text": row["non_text"],
            "is_error": row.get("is_error", False),
        }
        for row in rows
        if row.get("include_in_timeline", True)
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
    if chars < 50:
        return Text("░", style=f"dim {color}")
    if chars < 200:
        return Text("▒", style=f"dim {color}")
    if chars < 500:
        return Text("▒", style=color)
    if chars < 2000:
        return Text("▓", style=color)
    return Text("█", style=f"bold {color}")


def _build_history_bar(entries: Sequence[dict], width: int = TIMELINE_WIDTH) -> tuple[Text, Text]:
    recent = list(entries[-width:])
    bar = Text(" history |", style="dim")
    for entry in recent:
        color = _get_role_color(entry["role"], is_error=entry.get("is_error", False))
        bar.append_text(
            _shade_block(entry["chars"], non_text=entry.get("non_text", False), color=color)
        )
    remaining = width - len(recent)
    if remaining > 0:
        bar.append("░" * remaining, style=Colours.TIMELINE_EMPTY)
    bar.append("|", style="dim")

    detail = Text(f"{len(entries)} turns", style="dim")
    return bar, detail


def _build_context_bar_line(
    current: int,
    window: int | None,
    width: int = TIMELINE_WIDTH,
) -> tuple[Text, Text]:
    bar = Text(" context |", style="dim")

    if not window or window <= 0:
        bar.append("░" * width, style=Colours.TIMELINE_EMPTY)
        bar.append("|", style="dim")
        detail = Text(f"{format_chars(current)} tokens (unknown window)", style="dim")
        return bar, detail

    percent = current / window if window else 0.0
    filled = min(width, int(round(min(percent, 1.0) * width)))

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
    return bar, detail


def _render_header_line(agent_name: str, *, console: Console | None, printer) -> None:
    header = Text()
    header.append("▎", style=Colours.HEADER)
    header.append("●", style=f"dim {Colours.HEADER}")
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

    rows = _build_history_rows(history)
    timeline_entries = _aggregate_timeline_entries(rows)

    history_bar, history_detail = _build_history_bar(timeline_entries)
    if usage_accumulator:
        current_tokens = getattr(usage_accumulator, "current_context_tokens", 0)
        window = getattr(usage_accumulator, "context_window_size", None)
    else:
        current_tokens = 0
        window = None
    context_bar, context_detail = _build_context_bar_line(current_tokens, window)

    _render_header_line(agent_name, console=console, printer=printer)

    gap = Text("   ")
    combined_line = Text()
    combined_line.append_text(history_bar)
    combined_line.append_text(gap)
    combined_line.append_text(context_bar)
    printer(combined_line)

    history_label_len = len(" history |")
    context_label_len = len(" context |")

    history_available = history_bar.cell_len - history_label_len
    context_available = context_bar.cell_len - context_label_len

    detail_line = Text()
    detail_line.append(" " * history_label_len, style="dim")
    detail_line.append_text(history_detail)
    if history_available > history_detail.cell_len:
        detail_line.append(" " * (history_available - history_detail.cell_len), style="dim")
    detail_line.append_text(gap)
    detail_line.append(" " * context_label_len, style="dim")
    detail_line.append_text(context_detail)
    if context_available > context_detail.cell_len:
        detail_line.append(" " * (context_available - context_detail.cell_len), style="dim")
    printer(detail_line)

    printer("")
    printer(
        Text(" " + "─" * (history_bar.cell_len + context_bar.cell_len + gap.cell_len), style="dim")
    )

    summary_candidates = [row for row in rows if not row.get("hide_summary")]
    summary_rows = summary_candidates[-SUMMARY_COUNT:]
    start_index = len(summary_candidates) - len(summary_rows) + 1

    role_arrows = {"user": "▶", "assistant": "◀", "tool": "▶"}
    role_labels = {"user": "user", "assistant": "assistant", "tool": "tool result"}

    try:
        total_width = console.width if console else get_terminal_size().columns
    except Exception:
        total_width = 80

    # Responsive column layout based on terminal width
    show_time = total_width >= 60
    show_chars = total_width >= 50

    header_line = Text(" ")
    header_line.append(" #", style="dim")
    header_line.append(" ", style="dim")
    header_line.append(f"    {'Role':<{ROLE_COLUMN_WIDTH}}", style="dim")
    if show_time:
        header_line.append(f" {'Time':>7}", style="dim")
    if show_chars:
        header_line.append(f" {'Chars':>7}", style="dim")
    header_line.append("  ", style="dim")
    header_line.append("Summary", style="dim")
    printer(header_line)

    for offset, row in enumerate(summary_rows):
        role = row["role"]
        color = _get_role_color(role, is_error=row.get("is_error", False))
        arrow = role_arrows.get(role, "▶")
        label = role_labels.get(role, role)
        if role == "assistant" and row.get("has_tool_request"):
            label = f"{label}*"
        chars = row["chars"]
        block = _shade_block(chars, non_text=row.get("non_text", False), color=color)

        details = row.get("details")
        preview_value = row["preview"]
        preview_text = _ensure_text(preview_value)
        detail_text = _ensure_text(details) if details else Text("")
        if detail_text.cell_len == 0:
            detail_text = None

        timing_ms = row.get("timing_ms")
        timing_str = format_time(timing_ms)

        line = Text(" ")
        line.append(f"{start_index + offset:>2}", style="dim")
        line.append(" ")
        line.append_text(block)
        line.append(" ")
        line.append(arrow, style=color)
        line.append(" ")
        line.append(f"{label:<{ROLE_COLUMN_WIDTH}}", style=color)
        if show_time:
            line.append(f" {timing_str:>7}", style="dim")
        if show_chars:
            line.append(f" {format_chars(chars):>7}", style="dim")
        line.append("  ")
        summary_width = max(0, total_width - line.cell_len)
        summary_text = _compose_summary_text(
            preview_text,
            detail_text,
            include_non_text=row.get("non_text", False),
            max_width=summary_width,
        )
        line.append_text(summary_text)
        printer(line)

    printer("")
