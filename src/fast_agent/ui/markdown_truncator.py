"""Markdown truncation optimized for streaming displays.

This module keeps the most recent portion of a markdown stream within a
viewport budget. It preserves code block fences and table headers without
requiring expensive render passes.
"""

from __future__ import annotations

from collections import OrderedDict
from hashlib import blake2b
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
        self._height_cache: OrderedDict[tuple[int, int, str, int, str], int] = OrderedDict()
        self._height_cache_limit = 128
        self._truncate_cache: OrderedDict[tuple[int, int, int, int, str], str] = (
            OrderedDict()
        )
        self._truncate_cache_limit = 32

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
        """Measure how many terminal rows the markdown will occupy."""
        if not text:
            return 0
        width = console.size.width
        if width <= 0:
            return len(text.split("\n"))
        text_len, text_digest = self._fingerprint(text)
        cache_key = (id(console), width, code_theme, text_len, text_digest)
        cached = self._height_cache.get(cache_key)
        if cached is not None:
            self._height_cache.move_to_end(cache_key)
            return cached
        try:
            from rich.markdown import Markdown

            options = console.options.update(width=width)
            lines = console.render_lines(
                Markdown(text, code_theme=code_theme),
                options=options,
                pad=False,
            )
        except Exception:
            height = self._buffer.estimate_display_lines(text, width)
        else:
            height = len(lines)
        self._height_cache[cache_key] = height
        if len(self._height_cache) > self._height_cache_limit:
            self._height_cache.popitem(last=False)
        return height

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
        cache_key: tuple[int, int, int, int, str] | None = None
        if console and terminal_width:
            text_len, text_digest = self._fingerprint(text)
            cache_key = (id(console), terminal_width, terminal_height, text_len, text_digest)
            cached = self._truncate_cache.get(cache_key)
            if cached is not None:
                self._truncate_cache.move_to_end(cache_key)
                return cached
        truncated = self._buffer.truncate_text(
            text,
            terminal_height=terminal_height,
            terminal_width=terminal_width,
            add_closing_fence=False,
            target_ratio=1.0,
        )
        if not console or terminal_height <= 0:
            return truncated
        if self.measure_rendered_height(truncated, console) <= terminal_height:
            if cache_key:
                self._truncate_cache[cache_key] = truncated
                if len(self._truncate_cache) > self._truncate_cache_limit:
                    self._truncate_cache.popitem(last=False)
            return truncated

        best = ""
        low = 1
        high = max(1, terminal_height - 1)
        while low <= high:
            mid = (low + high) // 2
            candidate = self._buffer.truncate_text(
                text,
                terminal_height=mid,
                terminal_width=terminal_width,
                add_closing_fence=False,
                target_ratio=1.0,
            )
            if not candidate:
                high = mid - 1
                continue
            candidate_height = self.measure_rendered_height(candidate, console)
            if candidate_height <= terminal_height:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        result = best or truncated
        if cache_key:
            self._truncate_cache[cache_key] = result
            if len(self._truncate_cache) > self._truncate_cache_limit:
                self._truncate_cache.popitem(last=False)
        return result

    def _fingerprint(self, text: str) -> tuple[int, str]:
        digest = blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
        return len(text), digest

    def cache_sizes(self) -> dict[str, int]:
        return {
            "height_entries": len(self._height_cache),
            "truncate_entries": len(self._truncate_cache),
        }

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
