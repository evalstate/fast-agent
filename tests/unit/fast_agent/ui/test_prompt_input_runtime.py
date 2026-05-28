from __future__ import annotations

from typing import TYPE_CHECKING, TextIO, cast

import pytest

from fast_agent.ui.prompt import input_runtime
from fast_agent.ui.prompt.keybindings import PromptInputInterrupt

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession


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

    input_runtime._clear_prompt_echo_line("good morning", stream=cast("TextIO", stream))

    assert stream.writes == [input_runtime._ERASE_PREVIOUS_LINE_SEQ]
    assert stream.flush_calls == 1


def test_clear_prompt_echo_line_keeps_slash_and_shell_commands_visible() -> None:
    stream = _FakeStream(tty=True)

    input_runtime._clear_prompt_echo_line("/help", stream=cast("TextIO", stream))
    input_runtime._clear_prompt_echo_line("!pwd", stream=cast("TextIO", stream))

    assert stream.writes == []
    assert stream.flush_calls == 0


def test_clear_prompt_echo_line_skips_multiline_and_non_tty() -> None:
    tty_stream = _FakeStream(tty=True)
    non_tty_stream = _FakeStream(tty=False)

    input_runtime._clear_prompt_echo_line("line1\nline2", stream=cast("TextIO", tty_stream))
    input_runtime._clear_prompt_echo_line("hello", stream=cast("TextIO", non_tty_stream))

    assert tty_stream.writes == []
    assert tty_stream.flush_calls == 0
    assert non_tty_stream.writes == []
    assert non_tty_stream.flush_calls == 0


def test_format_prompt_prefix_omits_default_agent_name() -> None:
    assert input_runtime._format_prompt_prefix("dev", default_agent_name="dev") == "❯"
    assert input_runtime._format_prompt_prefix("default") == "default ❯"
    assert input_runtime._format_prompt_prefix("dev") == "dev ❯"


@pytest.mark.asyncio
async def test_run_prompt_once_converts_prompt_input_interrupt_to_interrupt_command() -> None:
    class _Buffer:
        def __init__(self) -> None:
            self.accept_handler = None

    class _Session:
        def __init__(self) -> None:
            self.default_buffer = _Buffer()

        async def prompt_async(self, *_args, **_kwargs):
            raise PromptInputInterrupt()

    result = await input_runtime.run_prompt_once(
        session=cast("PromptSession", _Session()),
        agent_name="agent",
        default_agent_name="agent",
        default_buffer="",
        resolve_prompt_text=lambda: "❯ ",
        parse_special_input=lambda value: value,
    )

    assert type(result).__name__ == "InterruptCommand"
