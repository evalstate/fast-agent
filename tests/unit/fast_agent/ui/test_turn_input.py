from __future__ import annotations

import asyncio
import os
import sys
import termios

import pytest
from prompt_toolkit.keys import Keys

from fast_agent.ui.prompt import turn_input
from fast_agent.ui.prompt.turn_input import TurnInputReader

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX terminal modes")


class Recorder:
    def __init__(self) -> None:
        self.cancelled = 0
        self.queued: list[int] = []
        self.written: list[str] = []

    def reader(self, fd: int) -> TurnInputReader:
        return TurnInputReader(
            fd=fd,
            on_cancel=self.cancel,
            on_queued=self.queued.append,
            write=self.written.append,
        )

    def cancel(self) -> None:
        self.cancelled += 1


def keys(reader: TurnInputReader) -> list[tuple[str, str]]:
    return [(str(k.key), k.data) for k in reader.held_keys]


def test_paste_and_enter_are_queued_like_hc_send() -> None:
    recorder = Recorder()
    reader = recorder.reader(fd=-1)
    reader.feed('\x1b[200~<hc from="t7">two\nlines</hc>\x1b[201~')
    reader.feed("\r")
    reader.flush()
    assert recorder.queued == [1]
    assert [k.key for k in reader.held_keys] == [Keys.BracketedPaste, Keys.ControlM]
    assert recorder.cancelled == 0


def test_queued_count_is_published_and_cleared_when_the_turn_ends() -> None:
    recorder = Recorder()
    reader = recorder.reader(fd=-1)
    reader.feed("one\r\rtwo\r")  # an empty Enter queues nothing
    reader.flush()
    assert recorder.queued == [1, 2]
    reader.stop()
    assert recorder.queued == [1, 2, 0]
    # Still held: both messages replay into the next prompts.
    assert [k.key for k in reader.held_keys].count(Keys.ControlM) == 3


def test_typed_text_keeps_editing_but_drops_prompt_controls() -> None:
    recorder = Recorder()
    reader = recorder.reader(fd=-1)
    reader.feed("hx\x7fi\x04\x1b[A\tthere")
    reader.flush()
    assert "".join(
        data for _key, data in keys(reader) if len(data) == 1 and data.isprintable()
    ) == ("hxithere")
    assert Keys.ControlD not in [k.key for k in reader.held_keys]
    assert Keys.Up not in [k.key for k in reader.held_keys]
    assert recorder.queued == []  # no Enter yet: becomes a draft in the next prompt


def test_lone_escape_cancels_but_alt_chords_do_not() -> None:
    recorder = Recorder()
    reader = recorder.reader(fd=-1)
    reader.feed("\x1bv")
    reader.flush()
    assert recorder.cancelled == 0
    assert reader.held_keys == []
    reader.feed("\x1b")
    assert recorder.cancelled == 0  # may still be the start of a sequence
    reader.flush()
    assert recorder.cancelled == 1


@pytest.mark.asyncio
async def test_pty_mode_is_held_during_turn_and_restored(monkeypatch) -> None:
    master, slave = os.openpty()
    try:
        original = termios.tcgetattr(slave)
        recorder = Recorder()
        reader = recorder.reader(slave)
        assert reader.start()
        held = termios.tcgetattr(slave)
        assert not held[0] & termios.ICRNL  # Enter stays Enter, not Ctrl+J
        assert not held[3] & (termios.ICANON | termios.ECHO)
        assert held[3] & termios.ISIG  # Ctrl+C still interrupts
        os.write(master, b"\x1b[200~queued\x1b[201~\r")
        for _ in range(50):
            await asyncio.sleep(0.01)
            if recorder.queued:
                break
        assert recorder.queued == [1]

        with reader.suspended():
            assert termios.tcgetattr(slave) == original
            os.write(master, b"for the form\n")
            await asyncio.sleep(0.05)
        assert recorder.queued == [1]  # the form's input was not taken
        assert not termios.tcgetattr(slave)[3] & termios.ICANON

        reader.stop()
        assert termios.tcgetattr(slave) == original
        assert recorder.written == ["\x1b[?2004h", "\x1b[?2004l", "\x1b[?2004h", "\x1b[?2004l"]
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.asyncio
async def test_held_keys_replay_into_successive_prompts() -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.input.typeahead import store_typeahead
    from prompt_toolkit.output import DummyOutput

    from fast_agent.ui.prompt.keybindings import create_keybindings

    recorder = Recorder()
    reader = recorder.reader(fd=-1)
    reader.feed("\x1b[200~first message\x1b[201~\r\x1b[200~second\x1b[201~\rdraft")
    reader.flush()

    with create_pipe_input() as pipe:
        store_typeahead(pipe, reader.held_keys)
        session = PromptSession(input=pipe, output=DummyOutput(), key_bindings=create_keybindings())
        assert await session.prompt_async() == "first message"
        assert await session.prompt_async() == "second"
        pipe.send_text("\r")
        assert await session.prompt_async() == "draft"


def test_disabled_by_environment_or_without_a_terminal(monkeypatch) -> None:
    monkeypatch.setenv("FAST_AGENT_DISABLE_TURN_INPUT", "1")
    assert not turn_input.turn_input_enabled()
    monkeypatch.delenv("FAST_AGENT_DISABLE_TURN_INPUT")
    monkeypatch.setattr(sys, "stdin", None)
    assert not turn_input.turn_input_enabled()
    with turn_input.hold_turn_input() as reader:
        assert reader is None
    with turn_input.suspend_turn_input():
        pass


def test_default_publisher_feeds_overlays_and_status(monkeypatch) -> None:
    from fast_agent.integrations import herdr_lifecycle

    reported: list[int] = []
    monkeypatch.setattr(herdr_lifecycle, "report_queued_input", reported.append)
    reader = TurnInputReader(fd=-1, write=lambda _s: None)
    reader.feed("hello\r")
    reader.flush()
    assert turn_input.queued_input_count() == 1
    reader.stop()
    assert turn_input.queued_input_count() == 0
    assert reported == [1, 0]
