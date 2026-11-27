"""Render the live inline input dock shown while a turn is running.

The dock renders at the bottom of the active Rich live surface (either the
progress display or the streaming display) so the user can see what they are
typing and which messages are queued without leaving the primary screen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from rich.cells import cell_len, chop_cells
from rich.console import Group
from rich.text import Text

from fast_agent.ui.message_styles import SHORT_RULE_WIDTH

DEFAULT_PROMPT_PREFIX = "❯"
_MAX_VISIBLE_QUEUED = 3
_MAX_VISIBLE_DRAFT_ROWS = 6
_MIN_RENDER_WIDTH = 12
_EOL_CURSOR_GLYPH = "▏"

# Matches ANSI CSI, OSC, and 2-byte ESC sequences so pasted terminal output
# cannot emit raw cursor-control codes into the live renderer.
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])")


@dataclass(frozen=True)
class TurnInputSnapshot:
    """Immutable view of the mid-turn input buffer and submission queue."""

    prompt_prefix: str
    draft_text: str
    cursor_position: int
    queued_messages: tuple[str, ...]
    multiline: bool


@dataclass(frozen=True)
class RenderedTurnInputDock:
    """Rendered dock together with its exact physical row count."""

    renderable: Group
    height: int


def _sanitize_line_with_cursor(
    raw_line: str,
    cursor_col: int | None,
) -> tuple[str, int | None]:
    """Strip ANSI escapes and control chars while mapping ``cursor_col`` cleanly."""
    parts: list[str] = []
    mapped_cursor: int | None = None
    out_len = 0
    pos = 0
    length = len(raw_line)

    for match in _ANSI_ESCAPE_RE.finditer(raw_line):
        start, end = match.span()
        while pos < start:
            if cursor_col is not None and pos == cursor_col:
                mapped_cursor = out_len
            ch = raw_line[pos]
            if ch == "\t":
                parts.append("    ")
                out_len += 4
            elif ch.isspace():
                parts.append(" ")
                out_len += 1
            elif ch.isprintable():
                parts.append(ch)
                out_len += 1
            pos += 1
        if cursor_col is not None and start <= cursor_col < end:
            mapped_cursor = out_len
        pos = end

    while pos < length:
        if cursor_col is not None and pos == cursor_col:
            mapped_cursor = out_len
        ch = raw_line[pos]
        if ch == "\t":
            parts.append("    ")
            out_len += 4
        elif ch.isspace():
            parts.append(" ")
            out_len += 1
        elif ch.isprintable():
            parts.append(ch)
            out_len += 1
        pos += 1

    if cursor_col is not None and mapped_cursor is None:
        mapped_cursor = out_len

    return "".join(parts), mapped_cursor


def _sanitize_single_line(text: str) -> str:
    sanitized, _ = _sanitize_line_with_cursor(text, None)
    return " ".join(sanitized.split())


def _single_line_preview(text: str, *, max_cells: int) -> str:
    collapsed = _sanitize_single_line(text)
    if max_cells <= 0:
        return ""
    if cell_len(collapsed) <= max_cells:
        return collapsed
    if max_cells == 1:
        return "…"
    out: list[str] = []
    used = 0
    limit = max_cells - 1
    for ch in collapsed:
        w = cell_len(ch)
        if used + w > limit:
            break
        out.append(ch)
        used += w
    return "".join(out) + "…"


def _split_draft_with_cursor(
    draft_text: str,
    cursor_position: int,
) -> tuple[list[str], int, int]:
    """Return ``(logical_lines, cursor_row, cursor_col)``.

    Rich's background refresh thread may read a snapshot while the event loop is
    mutating the buffer, so clamping ``cursor_position`` keeps slicing safe.
    """
    clamped = max(0, min(cursor_position, len(draft_text)))
    lines = draft_text.split("\n")
    offset = 0
    for row, line in enumerate(lines):
        line_end = offset + len(line)
        if clamped <= line_end:
            return lines, row, clamped - offset
        offset = line_end + 1
    return lines, len(lines) - 1, len(lines[-1])


def _visible_row_window(total_rows: int, cursor_row: int, max_rows: int) -> tuple[int, int]:
    """Return ``(start_row, end_row_exclusive)`` keeping ``cursor_row`` visible."""
    if total_rows <= max_rows:
        return 0, total_rows
    half = max_rows // 2
    start = max(0, cursor_row - half)
    end = start + max_rows
    if end > total_rows:
        end = total_rows
        start = max(0, end - max_rows)
    return start, end


def _prompt_prefixes(prompt_prefix: str) -> tuple[str, str]:
    first = f"{prompt_prefix} "
    continuation = " " * cell_len(first)
    return first, continuation


def _placeholder_text(multiline: bool) -> str:
    if multiline:
        return "Type to queue… (Ctrl+J to send, Esc to interrupt)"
    return "Type to queue a message… (Esc to interrupt)"


def _build_queued_rows(
    snapshot: TurnInputSnapshot,
    *,
    width: int,
    max_queued_rows: int,
) -> list[Text]:
    queued = snapshot.queued_messages
    if not queued or max_queued_rows <= 0:
        return []
    rows: list[Text] = []
    lead = "  ↳ "
    lead_cells = cell_len(lead)
    show_edit_hint = not snapshot.draft_text

    if max_queued_rows == 1 and len(queued) > 1:
        summary = Text(no_wrap=True, overflow="crop")
        summary.append(lead, style="yellow")
        summary.append(f"{len(queued)} queued", style="dim italic")
        if show_edit_hint:
            hint = "  (↑ edit last)"
            if cell_len(summary.plain) + cell_len(hint) <= width:
                summary.append(hint, style="dim italic")
        return [summary]

    visible_count = len(queued) if len(queued) <= max_queued_rows else max_queued_rows - 1
    for idx in range(visible_count):
        is_last_visible = idx == visible_count - 1 and visible_count == len(queued)
        suffix = "  (↑ edit)" if (show_edit_hint and is_last_visible) else ""
        available = width - lead_cells - cell_len(suffix)
        if suffix and available < 4:
            suffix = ""
            available = width - lead_cells
        preview = _single_line_preview(queued[idx], max_cells=max(1, available))
        line = Text(no_wrap=True, overflow="crop")
        line.append(lead, style="yellow")
        line.append(preview, style="dim")
        if suffix:
            line.append(suffix, style="dim italic")
        rows.append(line)

    remaining = len(queued) - visible_count
    if remaining > 0:
        more = Text(no_wrap=True, overflow="crop")
        more.append(lead, style="yellow")
        more.append(f"+{remaining} more queued", style="dim italic")
        if show_edit_hint:
            hint = "  (↑ edit last)"
            if cell_len(more.plain) + cell_len(hint) <= width:
                more.append(hint, style="dim italic")
        rows.append(more)
    return rows


def _chop_line_with_cursor(
    sanitized: str,
    cursor_col: int | None,
    *,
    first_prefix: str,
    first_prefix_style: str,
    cont_prefix: str,
    content_width: int,
) -> tuple[list[Text], int | None]:
    if cursor_col is None:
        chunks = chop_cells(sanitized, content_width) if sanitized else [""]
        rows: list[Text] = []
        for idx, chunk in enumerate(chunks):
            row = Text(no_wrap=True, overflow="crop")
            if idx == 0:
                row.append(first_prefix, style=first_prefix_style)
            else:
                row.append(cont_prefix, style="dim")
            if chunk:
                row.append(chunk)
            rows.append(row)
        return rows, None

    if cursor_col >= len(sanitized):
        display_text = sanitized + _EOL_CURSOR_GLYPH
        cursor_idx = len(sanitized)
    else:
        display_text = sanitized
        cursor_idx = cursor_col

    chunks = chop_cells(display_text, content_width)
    rows = []
    cursor_chunk_idx = len(chunks) - 1
    offset = 0
    for idx, chunk in enumerate(chunks):
        row = Text(no_wrap=True, overflow="crop")
        if idx == 0:
            row.append(first_prefix, style=first_prefix_style)
        else:
            row.append(cont_prefix, style="dim")
        chunk_end = offset + len(chunk)
        if offset <= cursor_idx < chunk_end:
            local = cursor_idx - offset
            cursor_chunk_idx = idx
            if local > 0:
                row.append(chunk[:local])
            row.append(chunk[local], style="reverse")
            if local + 1 < len(chunk):
                row.append(chunk[local + 1 :])
        elif chunk:
            row.append(chunk)
        rows.append(row)
        offset = chunk_end
    return rows, cursor_chunk_idx


def _build_draft_physical_rows(
    snapshot: TurnInputSnapshot,
    *,
    width: int,
    max_rows: int,
) -> list[Text]:
    first_prefix, cont_prefix = _prompt_prefixes(snapshot.prompt_prefix)

    if not snapshot.draft_text:
        input_line = Text(no_wrap=True, overflow="crop")
        input_line.append(first_prefix, style="bold cyan")
        input_line.append(_EOL_CURSOR_GLYPH, style="reverse")
        placeholder = _placeholder_text(snapshot.multiline)
        max_placeholder = max(0, width - cell_len(first_prefix) - 2)
        if max_placeholder > 0:
            input_line.append(" ", style="default")
            input_line.append(
                _single_line_preview(placeholder, max_cells=max_placeholder),
                style="dim",
            )
        return [input_line]

    lines, cursor_row, cursor_col = _split_draft_with_cursor(
        snapshot.draft_text, snapshot.cursor_position
    )
    # Only wrap logical lines within max_rows of the cursor so huge pastes stay O(1).
    slice_start = max(0, cursor_row - max_rows)
    slice_end = min(len(lines), cursor_row + max_rows + 1)
    content_width = max(1, width - cell_len(first_prefix))

    physical_rows: list[Text] = []
    cursor_physical_row = 0

    for row_index in range(slice_start, slice_end):
        raw_line = lines[row_index]
        is_first_logical = row_index == 0
        line_prefix = first_prefix if is_first_logical else cont_prefix
        line_prefix_style = "bold cyan" if is_first_logical else "dim"
        sanitized, mapped_cursor = _sanitize_line_with_cursor(
            raw_line, cursor_col if row_index == cursor_row else None
        )
        wrapped, cursor_in_line = _chop_line_with_cursor(
            sanitized,
            mapped_cursor,
            first_prefix=line_prefix,
            first_prefix_style=line_prefix_style,
            cont_prefix=cont_prefix,
            content_width=content_width,
        )
        if cursor_in_line is not None:
            cursor_physical_row = len(physical_rows) + cursor_in_line
        physical_rows.extend(wrapped)

    start, end = _visible_row_window(len(physical_rows), cursor_physical_row, max_rows)
    return physical_rows[start:end]


@lru_cache(maxsize=8)
def _render_dock_cached(
    snapshot: TurnInputSnapshot,
    safe_width: int,
    effective_max_height: int | None,
) -> RenderedTurnInputDock:
    if effective_max_height is None:
        max_queued_rows = _MAX_VISIBLE_QUEUED
    else:
        max_queued_rows = min(_MAX_VISIBLE_QUEUED, max(0, effective_max_height - 2))

    queued_rows = _build_queued_rows(snapshot, width=safe_width, max_queued_rows=max_queued_rows)
    divider = Text(
        "─" * min(safe_width, SHORT_RULE_WIDTH), style="dim", no_wrap=True, overflow="crop"
    )

    fixed_rows = len(queued_rows) + 1
    max_draft_rows = _MAX_VISIBLE_DRAFT_ROWS
    if effective_max_height is not None:
        max_draft_rows = max(1, min(_MAX_VISIBLE_DRAFT_ROWS, effective_max_height - fixed_rows))

    draft_rows = _build_draft_physical_rows(
        snapshot,
        width=safe_width,
        max_rows=max_draft_rows,
    )
    all_rows = [*queued_rows, divider, *draft_rows]
    return RenderedTurnInputDock(renderable=Group(*all_rows), height=len(all_rows))


def render_turn_input_dock(
    snapshot: TurnInputSnapshot,
    *,
    width: int,
    terminal_height: int | None = None,
    max_height: int | None = None,
) -> RenderedTurnInputDock:
    """Build the Rich renderable for the live turn input dock and its exact row count."""
    safe_width = max(_MIN_RENDER_WIDTH, width)
    effective_max_height = max_height
    if effective_max_height is None and terminal_height is not None:
        effective_max_height = max(3, terminal_height // 3)
    return _render_dock_cached(snapshot, safe_width, effective_max_height)
