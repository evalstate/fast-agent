from fast_agent.ui.stream_viewport import estimate_plain_text_height


def test_estimate_plain_text_height_wraps_long_line() -> None:
    assert estimate_plain_text_height("hello world again", 6) == 3


def test_estimate_plain_text_height_handles_long_word() -> None:
    assert estimate_plain_text_height("abcdefghij", 4) == 3


def test_estimate_plain_text_height_counts_lines() -> None:
    text = "abc\ndefghij\n"
    assert estimate_plain_text_height(text, 4) == 4
