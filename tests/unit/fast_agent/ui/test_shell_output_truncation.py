from fast_agent.ui.shell_output_truncation import (
    SHELL_OUTPUT_TRUNCATION_MARKER,
    ShellOutputLineWindow,
    format_shell_output_line_count,
    split_shell_output_line_limit,
    truncate_shell_output_lines,
)


def test_format_shell_output_line_count_pluralizes_lines() -> None:
    assert format_shell_output_line_count(1) == "1 line"
    assert format_shell_output_line_count(2) == "2 lines"


def test_split_shell_output_line_limit_even() -> None:
    assert split_shell_output_line_limit(6) == ShellOutputLineWindow(
        head_lines=3,
        tail_lines=3,
    )


def test_split_shell_output_line_limit_odd_biases_tail() -> None:
    assert split_shell_output_line_limit(7) == ShellOutputLineWindow(
        head_lines=3,
        tail_lines=4,
    )


def test_split_shell_output_line_limit_ignores_nonpositive_limits() -> None:
    assert split_shell_output_line_limit(0) == ShellOutputLineWindow(
        head_lines=0,
        tail_lines=0,
    )
    assert split_shell_output_line_limit(-1) == ShellOutputLineWindow(
        head_lines=0,
        tail_lines=0,
    )


def test_truncate_shell_output_lines_uses_head_marker_tail() -> None:
    lines = [f"line-{i}" for i in range(1, 11)]
    truncated, was_truncated = truncate_shell_output_lines(lines, 6)

    assert was_truncated is True
    assert truncated == [
        "line-1",
        "line-2",
        "line-3",
        SHELL_OUTPUT_TRUNCATION_MARKER,
        "line-8",
        "line-9",
        "line-10",
    ]


def test_truncate_shell_output_lines_uses_custom_marker() -> None:
    truncated, was_truncated = truncate_shell_output_lines(
        ["line-1", "line-2", "line-3"],
        2,
        marker="<cut>",
    )

    assert was_truncated is True
    assert truncated == ["line-1", "<cut>", "line-3"]


def test_truncate_shell_output_lines_with_zero_limit_returns_marker_only() -> None:
    truncated, was_truncated = truncate_shell_output_lines(["line-1", "line-2"], 0)

    assert was_truncated is True
    assert truncated == [SHELL_OUTPUT_TRUNCATION_MARKER]


def test_truncate_shell_output_lines_with_negative_limit_returns_marker_only() -> None:
    truncated, was_truncated = truncate_shell_output_lines(["line-1", "line-2"], -1)

    assert was_truncated is True
    assert truncated == [SHELL_OUTPUT_TRUNCATION_MARKER]


def test_truncate_shell_output_lines_returns_original_when_within_limit() -> None:
    lines = ["line-1", "line-2"]
    truncated, was_truncated = truncate_shell_output_lines(lines, 6)

    assert was_truncated is False
    assert truncated == lines
