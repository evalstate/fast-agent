"""Rendering helpers for history compaction output (shared by /compact and auto-compact)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.ui.history_display import Colours, format_chars

if TYPE_CHECKING:
    from fast_agent.history.compaction import CompactionResult

_BAR_WIDTH = 20


def _context_color(percent: float) -> str:
    if percent >= 0.9:
        return Colours.CONTEXT_ALERT
    if percent >= 0.7:
        return Colours.CONTEXT_CAUTION
    return Colours.CONTEXT_SAFE


def context_bar(tokens: int | None, window: int | None, *, width: int = _BAR_WIDTH) -> Text:
    """Render a context usage bar like ``|████░░░░| 42.1%``."""
    bar = Text("|", style="dim")
    if not window or window <= 0 or tokens is None or tokens < 0:
        bar.append("░" * width, style="dim default")
        bar.append("|", style="dim")
        bar.append(" ──.─%", style="dim")
        return bar

    percent = tokens / window
    filled = min(width, round(min(percent, 1.0) * width))
    color = _context_color(percent)
    if filled > 0:
        bar.append("█" * filled, style=color)
    if filled < width:
        bar.append("░" * (width - filled), style="dim default")
    bar.append("|", style="dim")
    bar.append(f" {min(percent, 9.99) * 100:5.1f}%", style=color)
    return bar


def compaction_summary_lines(result: "CompactionResult") -> list[Text]:
    """Render the before → after context visualization for a compaction result."""
    window = result.context_window

    context_line = Text("context ", style="dim")
    context_line.append_text(context_bar(result.tokens_before, window))
    context_line.append("  →  ", style="dim")
    context_line.append_text(context_bar(result.tokens_after_estimate, window))
    context_line.append(" est", style="dim")

    detail_line = Text()
    before_tokens = format_chars(result.tokens_before) if result.tokens_before else "unknown"
    detail_line.append(f"{before_tokens} → ~{format_chars(result.tokens_after_estimate)} tokens")
    if window:
        detail_line.append(f" of {format_chars(window)} window", style="dim")
    detail_line.append("  •  ", style="dim")
    detail_line.append(f"{result.messages_before} → {result.messages_after} messages")

    lines = [context_line, detail_line]
    if result.archive_file:
        archive_line = Text("archived ", style="dim")
        archive_line.append(result.archive_file, style="dim")
        lines.append(archive_line)
    return lines
