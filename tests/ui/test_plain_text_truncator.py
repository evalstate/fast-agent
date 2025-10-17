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
