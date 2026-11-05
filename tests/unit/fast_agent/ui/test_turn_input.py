from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING

import pytest
from prompt_toolkit.input import Input, PipeInput, create_input, create_pipe_input
from prompt_toolkit.keys import Keys

from fast_agent.ui.prompt import turn_input
from fast_agent.ui.prompt.turn_input import TurnInputReader

if TYPE_CHECKING:
    from collections.abc import Iterator

termios = pytest.importorskip("termios", reason="POSIX terminal modes")


@pytest.fixture
def pipe() -> Iterator[PipeInput]:
    with create_pipe_input() as input:
        yield input


class Recorder:
    def __init__(self) -> None:
        self.cancelled = 0
        self.queued: list[int] = []
        self.written: list[str] = []

    def reader(self, input: Input, *, multiline: bool = False) -> TurnInputReader:
        return TurnInputReader(
            input=input,
            multiline=multiline,
            on_cancel=self.cancel,
            on_queued=self.queued.append,
            write=self.written.append,
        )

    def cancel(self) -> None:
        self.cancelled += 1


def keys(reader: TurnInputReader) -> list[tuple[str, str]]:
    return [(str(k.key), k.data) for k in reader.held_keys]


def test_paste_and_enter_are_queued_like_hc_send(pipe: PipeInput) -> None:
    recorder = Recorder()
    reader = recorder.reader(pipe)
    pipe.send_text('\x1b[200~<hc from="t7">two\nlines</hc>\x1b[201~')
    reader._readable()
    pipe.send_text("\r")
    reader._readable()
    reader.flush()
    assert recorder.queued == [1]
    assert [k.key for k in reader.held_keys] == [Keys.BracketedPaste, Keys.ControlM]
    assert recorder.cancelled == 0


def test_queued_count_is_published_and_cleared_when_the_turn_ends(pipe: PipeInput) -> None:
    recorder = Recorder()
    reader = recorder.reader(pipe)
    pipe.send_text("one\r\rtwo\r")  # an empty Enter queues nothing
    reader._readable()
    reader.flush()
    assert recorder.queued == [1, 2]
    reader.stop()
    assert recorder.queued == [1, 2, 0]
    # Still held: both messages replay into the next prompts.
    assert [k.key for k in reader.held_keys].count(Keys.ControlM) == 3


def test_typed_text_keeps_editing_but_drops_prompt_controls(pipe: PipeInput) -> None:
    recorder = Recorder()
    reader = recorder.reader(pipe)
    pipe.send_text("hx\x7fi\x04\x1b[A\tthere")
    reader._readable()
    reader.flush()
    assert "".join(
        data for _key, data in keys(reader) if len(data) == 1 and data.isprintable()
    ) == ("hxithere")
    assert Keys.ControlD not in [k.key for k in reader.held_keys]
    assert Keys.Up not in [k.key for k in reader.held_keys]
    assert recorder.queued == []  # no Enter yet: becomes a draft in the next prompt


def test_lone_escape_cancels_but_alt_chords_do_not(pipe: PipeInput) -> None:
    recorder = Recorder()
    reader = recorder.reader(pipe)
    pipe.send_text("\x1bv")
    reader._readable()
    reader.flush()
    assert recorder.cancelled == 0
    assert reader.held_keys == []
    pipe.send_text("\x1b")
    reader._readable()
    assert recorder.cancelled == 0  # may still be the start of a sequence
    reader.flush()
    assert recorder.cancelled == 1


@pytest.mark.asyncio
async def test_pty_mode_is_held_during_turn_and_restored() -> None:
    master, slave = os.openpty()
    stdin = os.fdopen(os.dup(slave), "r")
    recorder = Recorder()
    reader = recorder.reader(create_input(stdin=stdin))
    try:
        original = termios.tcgetattr(slave)
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
        reader.stop()
        stdin.close()
        os.close(master)
        os.close(slave)


@pytest.mark.asyncio
async def test_held_keys_replay_into_successive_prompts(pipe: PipeInput) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input.typeahead import store_typeahead
    from prompt_toolkit.output import DummyOutput

    from fast_agent.ui.prompt.keybindings import create_keybindings

    recorder = Recorder()
    reader = recorder.reader(pipe)
    pipe.send_text("\x1b[200~first message\x1b[201~\r\x1b[200~second\x1b[201~\rdraft")
    reader._readable()
    reader.flush()

    store_typeahead(pipe, reader.stop())
    session = PromptSession(input=pipe, output=DummyOutput(), key_bindings=create_keybindings())
    assert await session.prompt_async() == "first message"
    assert await session.prompt_async() == "second"
    pipe.send_text("\r")
    assert await session.prompt_async() == "draft"


def test_unicode_is_decoded_incrementally(pipe: PipeInput) -> None:
    recorder = Recorder()
    reader = recorder.reader(pipe)
    # Deliver every byte separately, including inside a four-byte character.
    for byte in "café/📁.txt\r".encode():
        pipe.send_bytes(bytes([byte]))
        reader._readable()
    assert "".join(key.data for key in reader.stop()) == "café/📁.txt\r"
    assert recorder.queued == [1, 0]


_PASTE = "\x1b[200~first café\nsecond 📁 half\x1b[201~".encode()


@pytest.mark.asyncio
@pytest.mark.parametrize("split", range(1, len(_PASTE)))
async def test_partial_paste_survives_turn_to_prompt_handoff(monkeypatch, split: int) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.output import DummyOutput

    master, slave = os.openpty()
    try:
        with os.fdopen(os.dup(slave), "r") as stdin:
            input = create_input(stdin=stdin)
            monkeypatch.setattr(turn_input, "turn_input_enabled", lambda: True)
            monkeypatch.setattr(turn_input, "_write_stdout", lambda _s: None)
            with create_app_session(input=input, output=DummyOutput()):
                with turn_input.hold_turn_input() as reader:
                    assert reader is not None
                    os.write(master, b"draft " + _PASTE[:split])
                    async with asyncio.timeout(1):
                        while not reader.held_keys:
                            await asyncio.sleep(0.001)
                # The default input must be the one used during the turn.
                prompt = PromptSession()

                def finish_paste() -> None:
                    os.write(master, _PASTE[split:] + b"\r")

                result = await asyncio.wait_for(
                    prompt.prompt_async(pre_run=finish_paste),
                    timeout=1,
                )
                assert result == "draft first café\nsecond 📁 half"
    finally:
        os.close(master)
        os.close(slave)


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


def test_default_publisher_feeds_overlays_and_status(monkeypatch, pipe: PipeInput) -> None:
    from fast_agent.integrations import herdr_lifecycle

    reported: list[int] = []
    monkeypatch.setattr(herdr_lifecycle, "report_queued_input", reported.append)
    reader = TurnInputReader(input=pipe, write=lambda _s: None)
    pipe.send_text("hello\r")
    reader._readable()
    reader.flush()
    assert turn_input.queued_input_count() == 1
    reader.stop()
    assert turn_input.queued_input_count() == 0
    assert reported == [1, 0]


def test_take_messages_drains_ready_bytes_and_preserves_draft(pipe: PipeInput) -> None:
    recorder = Recorder()
    reader = recorder.reader(pipe)
    pipe.send_text("firsx\x7ft\r\x1b[200~second\r\nline café\x1b[201~\rdraft")
    assert reader.take_messages() == ["first", "second\nline café"]
    assert reader.take_messages() == []
    assert "".join(key.data for key in reader.held_keys) == "draft"
    assert recorder.queued[-1] == 0
    pipe.send_text("\r")
    assert reader.take_messages() == ["draft"]
    assert reader.stop() == []


@pytest.mark.parametrize("multiline", [False, True])
def test_take_messages_waits_for_active_submit_key(pipe: PipeInput, multiline: bool) -> None:
    recorder = Recorder()
    reader = recorder.reader(pipe, multiline=multiline)
    newline, submit = ("\r", "\n") if multiline else ("\n", "\r")
    draft = f"first line{newline}second line"
    pipe.send_text(draft)
    assert reader.take_messages() == []
    assert recorder.queued == []
    assert "".join(key.data for key in reader.held_keys) == draft

    pipe.send_text(submit + "next draft")
    assert reader.take_messages() == ["first line\nsecond line"]
    assert recorder.queued == [1, 0]
    assert "".join(key.data for key in reader.stop()) == "next draft"


@pytest.mark.asyncio
@pytest.mark.parametrize("multiline", [False, True])
async def test_unfinished_multiline_draft_replays_intact(
    monkeypatch, pipe: PipeInput, multiline: bool
) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input.typeahead import store_typeahead
    from prompt_toolkit.output import DummyOutput

    from fast_agent.ui.prompt import input as prompt_input
    from fast_agent.ui.prompt.keybindings import create_keybindings

    monkeypatch.setattr(prompt_input, "in_multiline_mode", multiline)
    reader = Recorder().reader(pipe, multiline=multiline)
    newline, submit = ("\r", "\n") if multiline else ("\n", "\r")
    pipe.send_text(f"first line{newline}second line")
    assert reader.take_messages() == []
    store_typeahead(pipe, reader.stop())
    prompt = PromptSession(input=pipe, output=DummyOutput(), key_bindings=create_keybindings())
    result = await asyncio.wait_for(
        prompt.prompt_async(pre_run=lambda: pipe.send_text(submit)), timeout=1
    )
    assert result == "first line\nsecond line"


@pytest.mark.parametrize(
    "deferred",
    [
        "/help",
        "!echo hi",
        "@other",
        "#other hello",
        "look ^file:photo.png",
        "STOP",
        " stop ",
        "exit",
        "ExIt",
        "HELP",
        "HELP:STATUS",
        "SHOW_USAGE",
        "SELECT_PROMPT:example",
        "SWITCH:other",
    ],
)
@pytest.mark.asyncio
async def test_command_barrier_preserves_order_and_replays_once(
    pipe: PipeInput, deferred: str
) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input.typeahead import store_typeahead
    from prompt_toolkit.output import DummyOutput

    from fast_agent.ui.prompt.keybindings import create_keybindings

    recorder = Recorder()
    reader = recorder.reader(pipe)
    pipe.send_text(f"report\r{deferred}\rlater\r")
    assert reader.take_messages() == ["report"]
    assert recorder.queued[-1] == 2
    assert reader.take_messages() == []
    store_typeahead(pipe, reader.stop())
    prompt = PromptSession(input=pipe, output=DummyOutput(), key_bindings=create_keybindings())
    assert await prompt.prompt_async() == deferred
    assert await prompt.prompt_async() == "later"


@pytest.mark.parametrize("text", ["STOP after this tool", "exit criteria", "HELP me debug", "help"])
def test_command_like_plain_text_is_consumed(pipe: PipeInput, text: str) -> None:
    reader = Recorder().reader(pipe)
    pipe.send_text(f"{text}\r")
    assert reader.take_messages() == [text]
    assert reader.held_keys == []


def test_pasted_image_stays_on_attachment_path(pipe: PipeInput, tmp_path) -> None:
    image = tmp_path / "photo.png"
    image.write_bytes(b"image")
    reader = Recorder().reader(pipe)
    pipe.send_text(f"look \x1b[200~{image}\x1b[201~\rlater\r")
    assert reader.take_messages() == []
    assert image.as_posix() in "".join(key.data for key in reader.stop())


def test_suspended_reader_does_not_consume_form_input(pipe: PipeInput) -> None:
    reader = Recorder().reader(pipe)
    pipe.send_text("report\r")
    with reader.suspended():
        assert reader.take_messages() == []
    assert reader.take_messages() == ["report"]


@pytest.mark.asyncio
async def test_replayed_count_survives_handoff_until_prompt_accepts(pipe: PipeInput) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input.typeahead import store_typeahead
    from prompt_toolkit.output import DummyOutput

    from fast_agent.ui.prompt.input_runtime import run_prompt_once
    from fast_agent.ui.prompt.keybindings import create_keybindings

    reader = TurnInputReader(input=pipe, write=lambda _s: None)
    pipe.send_text("/help\r\rlater\r")
    assert reader.take_messages() == []
    store_typeahead(pipe, reader.stop(replay=True))
    assert turn_input.queued_input_count() == 2
    prompt = PromptSession(input=pipe, output=DummyOutput(), key_bindings=create_keybindings())
    try:
        for expected, count in [("/help", 1), ("", 1), ("later", 0)]:
            result = await run_prompt_once(
                session=prompt,
                agent_name="test",
                default_agent_name="test",
                default_buffer="",
                resolve_prompt_text=lambda: "> ",
                parse_special_input=lambda text: text,
            )
            assert result == expected
            assert turn_input.queued_input_count() == count
    finally:
        turn_input.clear_replayed_input()


@pytest.mark.asyncio
@pytest.mark.parametrize("multiline", [False, True])
async def test_new_turn_adopts_prompt_typeahead_before_fresh_input(
    monkeypatch, multiline: bool
) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input.typeahead import store_typeahead
    from prompt_toolkit.key_binding import KeyPress
    from prompt_toolkit.output import DummyOutput

    from fast_agent.ui.prompt import input as prompt_input
    from fast_agent.ui.prompt.keybindings import create_keybindings

    monkeypatch.setattr(prompt_input, "in_multiline_mode", multiline)
    submit_key, submit = (Keys.ControlJ, "\n") if multiline else (Keys.ControlM, "\r")

    master, slave = os.openpty()
    try:
        with os.fdopen(os.dup(slave), "r") as stdin:
            input = create_input(stdin=stdin)
            monkeypatch.setattr(turn_input, "turn_input_enabled", lambda: True)
            monkeypatch.setattr(turn_input, "_write_stdout", lambda _s: None)
            with create_app_session(input=input, output=DummyOutput()):
                store_typeahead(
                    input,
                    [
                        KeyPress(Keys.BracketedPaste, "older"),
                        KeyPress(submit_key, submit),
                    ],
                )
                with turn_input.hold_turn_input() as reader:
                    assert reader is not None
                    os.write(master, f"newer{submit}/help{submit}".encode())
                    assert reader.take_messages() == ["older", "newer"]
                    assert turn_input.queued_input_count() == 1
                assert turn_input.queued_input_count() == 1
                prompt = PromptSession(key_bindings=create_keybindings())
                assert await prompt.prompt_async() == "/help"
                turn_input.acknowledge_replayed_input()
                assert turn_input.queued_input_count() == 0
    finally:
        turn_input.clear_replayed_input()
        os.close(master)
        os.close(slave)
