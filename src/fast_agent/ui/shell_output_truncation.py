from __future__ import annotations

from dataclasses import dataclass

from fast_agent.utils.count_display import format_count

SHELL_OUTPUT_TRUNCATION_MARKER = "△▽.........."


@dataclass(frozen=True, slots=True)
class ShellOutputLineWindow:
    head_lines: int
    tail_lines: int


def format_shell_output_line_count(line_count: int) -> str:
    """Return a human-friendly shell output line-count label."""
    return format_count(line_count, "line")


def split_shell_output_line_limit(line_limit: int) -> ShellOutputLineWindow:
    """Split a line limit into head/tail windows.

    For limits greater than one, the tail window receives the extra line when
    the limit is odd so users see slightly more of the most recent output.
    """
    normalized_limit = max(0, line_limit)
    head_lines, extra_tail_line = divmod(normalized_limit, 2)
    if normalized_limit == 1:
        return ShellOutputLineWindow(head_lines=1, tail_lines=0)
    tail_lines = head_lines + extra_tail_line
    return ShellOutputLineWindow(head_lines=head_lines, tail_lines=tail_lines)


def _head_marker_tail(
    lines: list[str],
    line_window: ShellOutputLineWindow,
    *,
    marker: str,
) -> list[str]:
    head = lines[: line_window.head_lines]
    tail = lines[-line_window.tail_lines :] if line_window.tail_lines > 0 else []
    return [*head, marker, *tail]


def truncate_shell_output_lines(
    lines: list[str],
    line_limit: int,
    *,
    marker: str = SHELL_OUTPUT_TRUNCATION_MARKER,
) -> tuple[list[str], bool]:
    """Truncate shell output to head + marker + tail windows.

    Returns the potentially truncated lines and whether truncation occurred.
    """
    all_lines = list(lines)
    if line_limit >= 0 and len(all_lines) <= line_limit:
        return all_lines, False

    line_window = split_shell_output_line_limit(line_limit)
    return _head_marker_tail(all_lines, line_window, marker=marker), True
