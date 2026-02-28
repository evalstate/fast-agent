from __future__ import annotations


def split_shell_output_line_limit(line_limit: int) -> tuple[int, int]:
    """Split a line limit into head/tail windows.

    For limits greater than one, the tail window receives the extra line when
    the limit is odd so users see slightly more of the most recent output.
    """
    if line_limit <= 0:
        return 0, 0
    if line_limit == 1:
        return 1, 0

    head_lines = line_limit // 2
    tail_lines = line_limit - head_lines
    return head_lines, tail_lines


def truncate_shell_output_lines(
    lines: list[str],
    line_limit: int,
    *,
    marker: str = "...",
) -> tuple[list[str], bool]:
    """Truncate shell output to head + marker + tail windows.

    Returns the potentially truncated lines and whether truncation occurred.
    """
    all_lines = list(lines)
    if line_limit < 0 or len(all_lines) <= line_limit:
        return all_lines, False

    head_lines, tail_lines = split_shell_output_line_limit(line_limit)
    truncated: list[str] = []

    if head_lines > 0:
        truncated.extend(all_lines[:head_lines])

    truncated.append(marker)

    if tail_lines > 0:
        truncated.extend(all_lines[-tail_lines:])

    return truncated, True
