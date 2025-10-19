from fast_agent.ui import console
from fast_agent.ui.console_display import _StreamingMessageHandle


def test_wrap_plain_line_breaks_on_space() -> None:
    result = _StreamingMessageHandle._wrap_plain_line("hello world again", 6)
    assert result == ["hello", "world", "again"]


def test_wrap_plain_line_handles_long_word() -> None:
    result = _StreamingMessageHandle._wrap_plain_line("abcdefghij", 4)
    assert result == ["abcd", "efgh", "ij"]


def test_wrap_plain_chunk_inserts_newlines() -> None:
    handle = object.__new__(_StreamingMessageHandle)
    handle._use_plain_text = True
    original_width = getattr(console.console, "_width", None)
    try:
        console.console._width = 8
        wrapped = _StreamingMessageHandle._wrap_plain_chunk(handle, "abcdefghijk")
        assert wrapped == "abcdefgh\nijk"
    finally:
        if original_width is None:
            delattr(console.console, "_width")
        else:
            console.console._width = original_width
