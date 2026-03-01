from __future__ import annotations

from typing import TextIO, cast

from fast_agent.ui.prompt import session_runtime


class _FakeStream:
    def __init__(self, *, tty: bool) -> None:
        self._tty = tty
        self.writes: list[str] = []
        self.flush_calls = 0

    def isatty(self) -> bool:
        return self._tty

    def write(self, value: str) -> None:
        self.writes.append(value)

    def flush(self) -> None:
        self.flush_calls += 1


def test_clear_prompt_echo_line_erases_regular_input() -> None:
    stream = _FakeStream(tty=True)

    session_runtime._clear_prompt_echo_line("good morning", stream=cast("TextIO", stream))

    assert stream.writes == [session_runtime._ERASE_PREVIOUS_LINE_SEQ]
    assert stream.flush_calls == 1


def test_clear_prompt_echo_line_keeps_slash_and_shell_commands_visible() -> None:
    stream = _FakeStream(tty=True)

    session_runtime._clear_prompt_echo_line("/help", stream=cast("TextIO", stream))
    session_runtime._clear_prompt_echo_line("!pwd", stream=cast("TextIO", stream))

    assert stream.writes == []
    assert stream.flush_calls == 0


def test_clear_prompt_echo_line_skips_multiline_and_non_tty() -> None:
    tty_stream = _FakeStream(tty=True)
    non_tty_stream = _FakeStream(tty=False)

    session_runtime._clear_prompt_echo_line("line1\nline2", stream=cast("TextIO", tty_stream))
    session_runtime._clear_prompt_echo_line("hello", stream=cast("TextIO", non_tty_stream))

    assert tty_stream.writes == []
    assert tty_stream.flush_calls == 0
    assert non_tty_stream.writes == []
    assert non_tty_stream.flush_calls == 0
