"""High performance truncation for plain text streaming displays."""

from __future__ import annotations

import math


class PlainTextTruncator:
    """Trim plain text content to fit within a target terminal window."""

    def __init__(self, target_height_ratio: float = 0.7) -> None:
        if not 0 < target_height_ratio <= 1:
            raise ValueError("target_height_ratio must be between 0 and 1")
        self.target_height_ratio = target_height_ratio

    def truncate(self, text: str, *, terminal_height: int, terminal_width: int) -> str:
        """Return the most recent portion of text that fits the terminal window.

        Args:
            text: Full text buffer accumulated during streaming.
            terminal_height: Terminal height in rows.
            terminal_width: Terminal width in columns.

        Returns:
            Tail portion of the text that fits within the target height ratio.
        """
        if not text:
            return text

        if terminal_height <= 0 or terminal_width <= 0:
            return text

        target_rows = max(1, int(terminal_height * self.target_height_ratio))
        width = max(1, terminal_width)

        idx = len(text)
        rows_used = 0
        start_idx = 0

        while idx > 0 and rows_used < target_rows:
            prev_newline = text.rfind("\n", 0, idx)
            line_start = prev_newline + 1 if prev_newline != -1 else 0
            line = text[line_start:idx]
            expanded = line.expandtabs()
            line_len = len(expanded)
            line_rows = max(1, math.ceil(line_len / width)) if line_len else 1

            if rows_used + line_rows >= target_rows:
                rows_remaining = target_rows - rows_used
                if rows_remaining <= 0:
                    start_idx = idx
                    break

                if line_rows <= rows_remaining:
                    start_idx = line_start
                else:
                    approx_chars = width * rows_remaining
                    keep_chars = min(len(line), approx_chars)
                    start_idx = idx - keep_chars
                break

            rows_used += line_rows
            start_idx = line_start
            if prev_newline == -1:
                break
            idx = prev_newline

        return text[start_idx:]
