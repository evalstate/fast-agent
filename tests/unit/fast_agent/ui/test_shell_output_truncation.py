from fast_agent.ui.shell_output_truncation import (
    split_shell_output_line_limit,
    truncate_shell_output_lines,
)


def test_split_shell_output_line_limit_even() -> None:
    assert split_shell_output_line_limit(6) == (3, 3)


def test_split_shell_output_line_limit_odd_biases_tail() -> None:
    assert split_shell_output_line_limit(7) == (3, 4)


def test_truncate_shell_output_lines_uses_head_marker_tail() -> None:
    lines = [f"line-{i}" for i in range(1, 11)]
    truncated, was_truncated = truncate_shell_output_lines(lines, 6)

    assert was_truncated is True
    assert truncated == [
        "line-1",
        "line-2",
        "line-3",
        "...",
        "line-8",
        "line-9",
        "line-10",
    ]


def test_truncate_shell_output_lines_returns_original_when_within_limit() -> None:
    lines = ["line-1", "line-2"]
    truncated, was_truncated = truncate_shell_output_lines(lines, 6)

    assert was_truncated is False
    assert truncated == lines
