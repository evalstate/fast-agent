"""Viewport calculations for streaming segment windows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rich.console import Console

    from fast_agent.ui.markdown_truncator import MarkdownTruncator
    from fast_agent.ui.plain_text_truncator import PlainTextTruncator
    from fast_agent.ui.stream_segments import StreamSegment


def estimate_plain_text_height(text: str, width: int) -> int:
    """Estimate how many terminal rows the plain text will occupy."""
    if not text:
        return 0
    width = max(1, width)
    return sum(_estimate_plain_text_line_height(line, width) for line in text.split("\n"))


def _estimate_plain_text_line_height(line: str, width: int) -> int:
    expanded_len = len(line.expandtabs())
    if not expanded_len:
        return 1
    return max(1, math.ceil(expanded_len / width))


@dataclass(frozen=True, slots=True)
class StreamViewportWindow:
    segments: list[StreamSegment]
    heights: list[int]


class StreamViewport:
    """Select a tail window of segments that fit within the viewport budget."""

    def __init__(
        self,
        *,
        markdown_truncator: MarkdownTruncator,
        plain_truncator: PlainTextTruncator,
        code_theme: str = "monokai",
    ) -> None:
        self._markdown_truncator = markdown_truncator
        self._plain_truncator = plain_truncator
        self._code_theme = code_theme

    def slice_segments(
        self,
        segments: Iterable[StreamSegment],
        *,
        terminal_height: int,
        console: Console,
        target_ratio: float,
    ) -> list[StreamSegment]:
        window = self.slice_segments_with_heights(
            segments,
            terminal_height=terminal_height,
            console=console,
            target_ratio=target_ratio,
        )
        return window.segments

    def slice_segments_with_heights(
        self,
        segments: Iterable[StreamSegment],
        *,
        terminal_height: int,
        console: Console,
        target_ratio: float,
    ) -> StreamViewportWindow:
        if terminal_height <= 0:
            segments_list = list(segments)
            width = max(1, console.size.width)
            heights = [
                self._segment_height(segment, console=console, width=width)
                for segment in segments_list
            ]
            return StreamViewportWindow(segments=segments_list, heights=heights)

        width = max(1, console.size.width)
        segments_list = [segment for segment in segments if segment.text]
        if not segments_list:
            return StreamViewportWindow(segments=[], heights=[])

        max_lines = max(1, int(terminal_height * target_ratio))

        heights = [
            self._segment_height(segment, console=console, width=width) for segment in segments_list
        ]
        total_height = sum(heights)
        if total_height <= max_lines:
            return StreamViewportWindow(segments=segments_list, heights=heights)

        remaining = max_lines
        window: list[StreamSegment] = []
        window_heights: list[int] = []
        for segment, height in zip(reversed(segments_list), reversed(heights), strict=True):
            if remaining <= 0:
                break
            if height <= remaining:
                window.append(segment)
                window_heights.append(height)
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
                window_heights.append(self._segment_height(trimmed, console=console, width=width))
            break

        window.reverse()
        window_heights.reverse()
        return StreamViewportWindow(segments=window, heights=window_heights)

    def _segment_height(self, segment: StreamSegment, *, console: Console, width: int) -> int:
        if segment.uses_markdown_layout:
            return self._markdown_truncator.measure_rendered_height(
                segment.text,
                console,
                code_theme=self._code_theme,
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
        if segment.uses_markdown_layout:
            truncated = self._markdown_truncator.truncate_to_height(
                segment.text,
                terminal_height=terminal_height,
                console=console,
                code_theme=self._code_theme,
            )
        else:
            truncated = self._plain_truncator.truncate(
                segment.text,
                terminal_height=terminal_height,
                terminal_width=terminal_width,
            )
        return segment.copy_with_text(truncated)


__all__ = ["StreamViewport", "StreamViewportWindow", "estimate_plain_text_height"]
