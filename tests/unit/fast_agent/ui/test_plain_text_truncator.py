from fast_agent.ui.plain_text_truncator import PlainTextTruncator


def test_truncate_keeps_small_text() -> None:
    truncator = PlainTextTruncator()
    text = "hello\nworld"

    result = truncator.truncate(text, terminal_height=20, terminal_width=80)

    assert result == text


def test_truncate_preserves_recent_lines() -> None:
    truncator = PlainTextTruncator()
    text = "\n".join(str(i) for i in range(50))

    result = truncator.truncate(text, terminal_height=10, terminal_width=10)

    assert result.splitlines() == [str(i) for i in range(43, 50)]


def test_truncate_handles_long_single_line() -> None:
    truncator = PlainTextTruncator()
    text = "x" * 500

    result = truncator.truncate(text, terminal_height=10, terminal_width=50)

    assert len(result) == 350  # width (50) * target rows (0.7 * 10 = 7)


def test_truncate_counts_exact_width_lines_as_one_row() -> None:
    truncator = PlainTextTruncator(target_height_ratio=1.0)
    text = "abcde\n123456"

    result = truncator.truncate(text, terminal_height=1, terminal_width=5)

    assert result == "23456"


def test_truncate_counts_expanded_tabs_when_estimating_rows() -> None:
    truncator = PlainTextTruncator(target_height_ratio=1.0)
    text = "first\n\tsecond\nthird"

    result = truncator.truncate(text, terminal_height=2, terminal_width=8)

    assert result == "\tsecond\nthird"
