"""Markdown truncation optimized for streaming displays.

This module keeps the most recent portion of a markdown stream within a
viewport budget. It preserves code block fences and table headers without
requiring expensive render passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.ui.streaming_buffer import StreamBuffer

if TYPE_CHECKING:
    from rich.console import Console


class MarkdownTruncator:
    """Handles lightweight markdown truncation for streaming output."""

    def __init__(self, target_height_ratio: float = 0.8) -> None:
        if not 0 < target_height_ratio <= 1:
            raise ValueError("target_height_ratio must be between 0 and 1")
        self.target_height_ratio = target_height_ratio
        self._buffer = StreamBuffer(target_height_ratio=target_height_ratio)

    def truncate(
        self,
        text: str,
        terminal_height: int,
        console: Console | None,
        code_theme: str = "monokai",
        prefer_recent: bool = False,
    ) -> str:
        """Return the most recent portion of text that fits the viewport.

        Args:
            text: The markdown text to truncate.
            terminal_height: Height of the terminal in lines.
            console: Rich Console instance used to derive width.
            code_theme: Unused; kept for compatibility.
            prefer_recent: Unused; kept for compatibility.
        """
        del code_theme, prefer_recent
        if not text:
            return text
        terminal_width = console.size.width if console else None
        return self._buffer.truncate_text(
            text,
            terminal_height=terminal_height,
            terminal_width=terminal_width,
            add_closing_fence=False,
        )

    def measure_rendered_height(
        self, text: str, console: Console, code_theme: str = "monokai"
    ) -> int:
        """Estimate how many terminal rows the markdown will occupy."""
        del code_theme
        if not text:
            return 0
        width = console.size.width
        if width <= 0:
            return len(text.split("\n"))
        return self._buffer.estimate_display_lines(text, width)

    def truncate_to_height(
        self,
        text: str,
        *,
        terminal_height: int,
        console: Console | None,
    ) -> str:
        """Truncate markdown to a specific display height."""
        if not text:
            return text
        terminal_width = console.size.width if console else None
        return self._buffer.truncate_text(
            text,
            terminal_height=terminal_height,
            terminal_width=terminal_width,
            add_closing_fence=False,
            target_ratio=1.0,
        )

    def _ensure_table_header_if_needed(self, original_text: str, truncated_text: str) -> str:
        """Ensure table header is prepended if truncation removed it."""
        if not truncated_text or truncated_text == original_text:
            return truncated_text

        truncation_pos = original_text.rfind(truncated_text)
        if truncation_pos == -1:
            truncation_pos = max(0, len(original_text) - len(truncated_text))

        tables = self._buffer._find_tables(original_text)
        if not tables:
            return truncated_text

        lines = original_text.split("\n")
        for table in tables:
            if not (table.start_pos < truncation_pos < table.end_pos):
                continue

            table_start_line = original_text[: table.start_pos].count("\n")
            data_start_line = table_start_line + len(table.header_lines)
            data_start_pos = sum(len(line) + 1 for line in lines[:data_start_line])

            if truncation_pos >= data_start_pos:
                header_text = "\n".join(table.header_lines) + "\n"
                if truncated_text.startswith(header_text):
                    return truncated_text
                truncated_lines = truncated_text.splitlines()
                header_lines = [line.rstrip() for line in table.header_lines]
                if len(truncated_lines) >= len(header_lines):
                    candidate = [
                        line.rstrip() for line in truncated_lines[: len(header_lines)]
                    ]
                    if candidate == header_lines:
                        return truncated_text
                return header_text + truncated_text

            return truncated_text

        return truncated_text


__all__ = ["MarkdownTruncator"]
