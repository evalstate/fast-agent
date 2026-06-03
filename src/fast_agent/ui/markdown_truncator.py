"""Markdown truncation optimized for streaming displays.

This module keeps the most recent portion of a markdown stream within a
viewport budget. It preserves code block fences and table headers without
requiring expensive render passes.
"""

from __future__ import annotations

from collections import OrderedDict
from hashlib import blake2b
from typing import TYPE_CHECKING

from fast_agent.ui.markdown_renderables import build_markdown_renderable
from fast_agent.ui.streaming_buffer import StreamBuffer, Table

if TYPE_CHECKING:
    from rich.console import Console


class MarkdownTruncator:
    """Handles lightweight markdown truncation for streaming output."""

    def __init__(
        self,
        target_height_ratio: float = 0.8,
        *,
        code_word_wrap: bool = True,
        render_fences_with_syntax: bool = True,
    ) -> None:
        if not 0 < target_height_ratio <= 1:
            raise ValueError("target_height_ratio must be between 0 and 1")
        self.target_height_ratio = target_height_ratio
        self.code_word_wrap = code_word_wrap
        self.render_fences_with_syntax = render_fences_with_syntax
        self._buffer = StreamBuffer(target_height_ratio=target_height_ratio)
        self._height_cache: OrderedDict[tuple[int, int, str, int, str], int] = OrderedDict()
        self._height_cache_limit = 128
        self._truncate_cache: OrderedDict[tuple[int, int, int, str, int, str], str] = (
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
            options = console.options.update(width=width)
            lines = console.render_lines(
                build_markdown_renderable(
                    text,
                    code_theme=code_theme,
                    escape_xml=False,
                    close_incomplete_fences=True,
                    render_fences_with_syntax=self.render_fences_with_syntax,
                    code_word_wrap=self.code_word_wrap,
                ),
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

    def estimate_rendered_height(self, text: str, terminal_width: int) -> int:
        """Cheap width-based estimate for markdown display height."""
        if not text:
            return 0
        if terminal_width <= 0:
            return len(text.split("\n"))
        return self._buffer.estimate_display_lines(text, terminal_width)

    def truncate_to_height(
        self,
        text: str,
        *,
        terminal_height: int,
        console: Console | None,
        code_theme: str = "monokai",
    ) -> str:
        """Truncate markdown to a specific display height."""
        if not text:
            return text
        terminal_width = console.size.width if console else None
        cache_key = self._truncate_cache_key(
            text,
            console=console,
            terminal_width=terminal_width,
            terminal_height=terminal_height,
            code_theme=code_theme,
        )
        cached = self._cached_truncation(cache_key)
        if cached is not None:
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
        if self._fits_height(truncated, console, terminal_height, code_theme):
            self._store_truncation(cache_key, truncated)
            return truncated

        result = self._best_fit_truncation(
            text,
            terminal_height=terminal_height,
            terminal_width=terminal_width,
            console=console,
            code_theme=code_theme,
            fallback=truncated,
        )
        self._store_truncation(cache_key, result)
        return result

    def _truncate_cache_key(
        self,
        text: str,
        *,
        console: Console | None,
        terminal_width: int | None,
        terminal_height: int,
        code_theme: str,
    ) -> tuple[int, int, int, str, int, str] | None:
        if not console or not terminal_width:
            return None
        text_len, text_digest = self._fingerprint(text)
        return (
            id(console),
            terminal_width,
            terminal_height,
            code_theme,
            text_len,
            text_digest,
        )

    def _cached_truncation(
        self,
        cache_key: tuple[int, int, int, str, int, str] | None,
    ) -> str | None:
        if cache_key is None:
            return None
        cached = self._truncate_cache.get(cache_key)
        if cached is not None:
            self._truncate_cache.move_to_end(cache_key)
        return cached

    def _store_truncation(
        self,
        cache_key: tuple[int, int, int, str, int, str] | None,
        value: str,
    ) -> None:
        if cache_key is None:
            return
        self._truncate_cache[cache_key] = value
        if len(self._truncate_cache) > self._truncate_cache_limit:
            self._truncate_cache.popitem(last=False)

    def _fits_height(
        self,
        text: str,
        console: Console,
        terminal_height: int,
        code_theme: str,
    ) -> bool:
        return self.measure_rendered_height(text, console, code_theme=code_theme) <= terminal_height

    def _best_fit_truncation(
        self,
        text: str,
        *,
        terminal_height: int,
        terminal_width: int | None,
        console: Console,
        code_theme: str,
        fallback: str,
    ) -> str:
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
            candidate_height = self.measure_rendered_height(
                candidate,
                console,
                code_theme=code_theme,
            )
            if candidate_height <= terminal_height:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        return best or fallback

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

        header_text = self._table_header_prefix_for_truncation(
            original_text,
            truncated_text,
            truncation_pos,
        )
        if header_text is None:
            return truncated_text
        return header_text + truncated_text

    def _table_header_prefix_for_truncation(
        self,
        original_text: str,
        truncated_text: str,
        truncation_pos: int,
    ) -> str | None:
        lines = original_text.split("\n")
        for table in self._buffer._find_tables(original_text):
            if not (table.start_pos < truncation_pos < table.end_pos):
                continue

            if truncation_pos < _table_data_start_pos(original_text, lines, table):
                return None

            header_text = "\n".join(table.header_lines) + "\n"
            if _truncated_text_has_table_header(truncated_text, header_text, table):
                return None
            return header_text

        return None


def _table_data_start_pos(original_text: str, lines: list[str], table: Table) -> int:
    table_start_line = original_text[: table.start_pos].count("\n")
    data_start_line = table_start_line + len(table.header_lines)
    return sum(len(line) + 1 for line in lines[:data_start_line])


def _truncated_text_has_table_header(
    truncated_text: str,
    header_text: str,
    table: Table,
) -> bool:
    if truncated_text.startswith(header_text):
        return True
    truncated_lines = truncated_text.splitlines()
    header_lines = [line.rstrip() for line in table.header_lines]
    if len(truncated_lines) < len(header_lines):
        return False
    candidate = [line.rstrip() for line in truncated_lines[: len(header_lines)]]
    return candidate == header_lines


__all__ = ["MarkdownTruncator"]
