"""Viewport calculations for streaming segment windows."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from rich.console import Console

    from fast_agent.ui.markdown_truncator import MarkdownTruncator
    from fast_agent.ui.plain_text_truncator import PlainTextTruncator
    from fast_agent.ui.stream_segments import StreamSegment


def estimate_plain_text_height(text: str, width: int) -> int:
    """Estimate how many terminal rows the plain text will occupy."""
    if not text:
        return 0
    width = max(1, width)
    total = 0
    for line in text.split("\n"):
        expanded_len = len(line.expandtabs())
        total += max(1, math.ceil(expanded_len / width)) if expanded_len else 1
    return total


class StreamViewport:
    """Select a tail window of segments that fit within the viewport budget."""

    def __init__(
        self,
        *,
        markdown_truncator: MarkdownTruncator,
        plain_truncator: PlainTextTruncator,
    ) -> None:
        self._markdown_truncator = markdown_truncator
        self._plain_truncator = plain_truncator

    def slice_segments(
        self,
        segments: Iterable[StreamSegment],
        *,
        terminal_height: int,
        console: Console,
        target_ratio: float,
    ) -> list[StreamSegment]:
        if terminal_height <= 0:
            return list(segments)

        width = max(1, console.size.width)
        segments_list = [segment for segment in segments if segment.text]
        if not segments_list:
            return []

        max_lines = max(1, int(terminal_height * target_ratio))

        heights = [
            self._segment_height(segment, console=console, width=width)
            for segment in segments_list
        ]
        total_height = sum(heights)
        if total_height <= max_lines:
            return segments_list

        remaining = max_lines
        window: list[StreamSegment] = []
        for segment, height in zip(reversed(segments_list), reversed(heights)):
            if remaining <= 0:
                break
            if height <= remaining:
                window.append(segment)
                remaining -= height
                continue

            trimmed = self._truncate_segment(
                segment,
                terminal_height=remaining,
                terminal_width=width,
                console=console,
            )
            if trimmed.text:
                window.append(trimmed)
            break

        window.reverse()
        return window

    def _segment_height(self, segment: StreamSegment, *, console: Console, width: int) -> int:
        if segment.kind in ("markdown", "reasoning"):
            return self._markdown_truncator.measure_rendered_height(
                segment.text, console, code_theme="monokai"
            )
        return estimate_plain_text_height(segment.text, width)

    def _truncate_segment(
        self,
        segment: StreamSegment,
        *,
        terminal_height: int,
        terminal_width: int,
        console: Console,
    ) -> StreamSegment:
        if terminal_height <= 0 or not segment.text:
            return segment.copy_with_text("")
        if segment.kind in ("markdown", "reasoning"):
            truncated = self._markdown_truncator.truncate_to_height(
                segment.text,
                terminal_height=terminal_height,
                console=console,
            )
        else:
            truncated = self._plain_truncator.truncate(
                segment.text,
                terminal_height=terminal_height,
                terminal_width=terminal_width,
            )
        return segment.copy_with_text(truncated)


__all__ = ["StreamViewport", "estimate_plain_text_height"]
