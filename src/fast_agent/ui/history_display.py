"""Display helpers for agent conversation history."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from rich import print as rich_print
from rich.console import Console
from rich.text import Text

from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

    from fast_agent.llm.usage_tracking import UsageAccumulator


NON_TEXT_MARKER = "^"
TIMELINE_WIDTH = 20
SUMMARY_COUNT = 8


def _normalize_text(value: Optional[str]) -> str:
    return "" if not value else " ".join(value.split())


def _char_count(value: Optional[str]) -> int:
    return len(_normalize_text(value))


def _preview_text(value: Optional[str], limit: int = 80) -> str:
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
    for block in getattr(result, "content", []) or []:
        text = get_text(block)
        if text:
            normalized = _normalize_text(text)
            return _preview_text(normalized, limit=limit), len(normalized), False
    return f"{NON_TEXT_MARKER} non-text tool result", 0, True


def format_chars(value: int) -> str:
    if value <= 0:
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


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

        if role == "user":
            rows.append(
                {
                    "role": "user",
                    "chars": chars,
                    "preview": preview,
                    "details": None,
                    "non_text": non_text,
                }
            )
            continue

        tool_calls: Optional[Mapping[str, object]] = getattr(message, "tool_calls", None)
        details = None
        row_non_text = non_text

        if tool_calls:
            names: list[str] = []
            for call_id, call in tool_calls.items():
                params = getattr(call, "params", None)
                name = getattr(params, "name", None) or getattr(call, "name", None) or call_id
                call_name_lookup[call_id] = name
                names.append(name)
            if names:
                details = "tool→" + ", ".join(names)
                row_non_text = row_non_text and chars == 0  # treat call as activity
        if not normalized_text and tool_calls:
            preview = "(issuing tool request)"

        rows.append(
            {
                "role": "assistant",
                "chars": chars,
                "preview": preview,
                "details": details,
                "non_text": row_non_text,
            }
        )

        tool_results: Optional[Mapping[str, object]] = getattr(message, "tool_results", None)
        if tool_results:
            for call_id, result in tool_results.items():
                summary, result_chars, result_non_text = _extract_tool_result_summary(result)
                tool_name = call_name_lookup.get(call_id, call_id)
                rows.append(
                    {
                        "role": "tool",
                        "chars": result_chars,
                        "preview": summary,
                        "details": f"result of {tool_name}",
                        "non_text": result_non_text,
                    }
                )

    return rows


def _aggregate_timeline_entries(rows: Sequence[dict]) -> list[dict]:
    return [
        {
            "role": row["role"],
            "chars": row["chars"],
            "non_text": row["non_text"],
        }
        for row in rows
    ]


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
        return Text("▓", style=color)
    if chars < 1000:
        return Text("█", style=f"dim {color}")
    return Text("█", style=f"bold {color}")


def _build_history_bar(entries: Sequence[dict], width: int = TIMELINE_WIDTH) -> tuple[Text, Text]:
    color_map = {"user": "cyan", "assistant": "green", "tool": "magenta"}

    recent = list(entries[-width:])
    bar = Text(" history |", style="dim")
    for entry in recent:
        color = color_map.get(entry["role"], "white")
        bar.append_text(
            _shade_block(entry["chars"], non_text=entry.get("non_text", False), color=color)
        )
    remaining = width - len(recent)
    if remaining > 0:
        bar.append("░" * remaining, style="grey58")
    bar.append("|", style="dim")

    detail = Text(f"{len(entries)} turns", style="dim")
    return bar, detail


def _build_context_bar_line(
    current: int,
    window: Optional[int],
    width: int = TIMELINE_WIDTH,
) -> tuple[Text, Text]:
    bar = Text(" context |", style="dim")

    if not window or window <= 0:
        bar.append("░" * width, style="grey58")
        bar.append("|", style="dim")
        detail = Text(f"{format_chars(current)} tokens (unknown window)", style="dim")
        return bar, detail

    percent = current / window if window else 0.0
    filled = min(width, int(round(min(percent, 1.0) * width)))

    def color_for(pct: float) -> str:
        if pct >= 0.9:
            return "bright_red"
        if pct >= 0.7:
            return "#af8700"
        return "ansigreen"

    color = color_for(percent)
    if filled > 0:
        bar.append("█" * filled, style=color)
    if filled < width:
        bar.append("░" * (width - filled), style="grey58")
    bar.append("|", style="dim")
    bar.append(f" {percent*100:5.1f}%", style="dim")
    if percent > 1.0:
        bar.append(f" +{(percent-1)*100:.0f}%", style="bold bright_red")

    detail = Text(f"{format_chars(current)} / {format_chars(window)} →", style="dim")
    return bar, detail


def display_history_overview(
    agent_name: str,
    history: Sequence[PromptMessageExtended],
    usage_accumulator: Optional["UsageAccumulator"] = None,
    *,
    console: Optional[Console] = None,
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

    header = Text()
    header.append("▎", style="cyan")
    header.append(f" Conversation for {agent_name}", style="bold")
    printer("")
    printer(header)

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
    printer(Text(" " + "─" * (history_bar.cell_len + context_bar.cell_len + gap.cell_len), style="dim"))

    header_line = Text(" ")
    header_line.append("#  ", style="dim")
    header_line.append("Role           ", style="dim")
    header_line.append("Chars    ", style="dim")
    header_line.append("Summary", style="dim")
    printer(header_line)

    summary_rows = rows[-SUMMARY_COUNT:]
    start_index = len(rows) - len(summary_rows) + 1

    role_arrows = {"user": "▶", "assistant": "◀", "tool": "▶"}
    role_styles = {"user": "cyan", "assistant": "green", "tool": "magenta"}
    role_labels = {"user": "user", "assistant": "assistant", "tool": "tool result"}

    for offset, row in enumerate(summary_rows):
        role = row["role"]
        color = role_styles.get(role, "white")
        arrow = role_arrows.get(role, "▶")
        label = role_labels.get(role, role)
        chars = row["chars"]
        block = _shade_block(chars, non_text=row.get("non_text", False), color=color)

        details = row.get("details")
        if isinstance(details, list):
            details = ", ".join(filter(None, details))
        summary = row["preview"]
        if details:
            summary = f"{summary} {details}"
        if row.get("non_text") and chars == 0:
            summary = f"{summary} [dim]{NON_TEXT_MARKER} non-text[/dim]"

        line = Text(" ")
        line.append(f"{start_index + offset:>2} ", style="dim")
        line.append_text(block)
        line.append(" ")
        line.append(arrow, style=color)
        line.append(" ")
        line.append(f"{label:<12}", style=color)
        line.append(f" {format_chars(chars):>7}", style="dim")
        line.append("  ")
        line.append(summary)
        printer(line)

    printer("")
