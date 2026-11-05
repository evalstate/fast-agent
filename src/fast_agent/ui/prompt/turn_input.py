"""Hold terminal input that arrives while an interactive turn is running.

Between prompts the terminal is otherwise left in cooked mode: typed or pasted
text is echoed into the streaming output, and the kernel translates Enter into
Ctrl+J, which the prompt treats as "insert newline". Messages typed ahead, or
sent by terminal managers such as herdr-core (``hc send``), were therefore left
unsubmitted in the next prompt.

While a turn runs this reader keeps the terminal in a non-echoing, untranslated
mode (signals stay enabled, so Ctrl+C is unchanged) and drains input as key
presses. Plain submitted text can be consumed at tool-loop continuation boundaries.
Commands, attachments, late messages and drafts are replayed into the next prompt
as prompt_toolkit typeahead. Submission order is preserved across both paths.
A lone Escape cancels the turn exactly like Ctrl+C, which is also what
``hc interrupt`` sends.

Nothing is printed while input is held: the queued count is shown in the live
streaming header and progress overlay (transient, never scrollback) and in the
terminal program status message.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fast_agent.utils.env import is_truthy_env_value

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from prompt_toolkit.input import Input
    from prompt_toolkit.key_binding import KeyPress

_DISABLE_ENV = "FAST_AGENT_DISABLE_TURN_INPUT"
_ESCAPE_TIMEOUT_SECONDS = 0.05
_BRACKETED_PASTE_ON = "\x1b[?2004h"
_BRACKETED_PASTE_OFF = "\x1b[?2004l"
_PREVIEW_CHARS = 72

_active: TurnInputReader | None = None
_queued_count = 0
_replayed_submissions: deque[bool] = deque()


@dataclass(frozen=True)
class _Submission:
    end: int
    text: str


def acknowledge_replayed_input() -> None:
    """A prompt accepted the next replayed submission (including empty Enter)."""
    if _replayed_submissions:
        _replayed_submissions.popleft()
        _publish_queued_count(0)


def clear_replayed_input() -> None:
    """Reset replay accounting when discarding an input runtime."""
    _replayed_submissions.clear()
    _publish_queued_count(0)


def turn_input_enabled() -> bool:
    """Interactive POSIX terminals only, unless disabled by environment."""
    if is_truthy_env_value(os.environ.get(_DISABLE_ENV)) or os.name == "nt":
        return False
    try:
        return sys.stdin is not None and sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def _default_cancel() -> None:
    # Identical to the person pressing Ctrl+C during a turn.
    signal.raise_signal(signal.SIGINT)


def _publish_queued_count(count: int) -> None:
    """Expose the count to live overlays and terminal status; never print it."""
    global _queued_count

    _queued_count = count + sum(_replayed_submissions)
    from fast_agent.integrations.herdr_lifecycle import report_queued_input

    report_queued_input(_queued_count)


def queued_input_count() -> int:
    """Submitted messages awaiting a continuation boundary or the next prompt."""
    return _queued_count


def _write_stdout(sequence: str) -> None:
    from rich.file_proxy import FileProxy

    stream: Any = sys.stdout
    # Streaming wraps stdout to render prints through Rich, which would show
    # control sequences as text. Write to the underlying terminal stream.
    while isinstance(stream, FileProxy):
        stream = stream.rich_proxied_file
    try:
        stream.write(sequence)
        stream.flush()
    except (OSError, ValueError):
        pass


class TurnInputReader:
    """Drain stdin during a turn; keep submitted text as prompt typeahead."""

    def __init__(
        self,
        *,
        input: Input,
        multiline: bool = False,
        on_cancel: Callable[[], None] = _default_cancel,
        on_queued: Callable[[int], None] = _publish_queued_count,
        write: Callable[[str], None] = _write_stdout,
    ) -> None:
        from prompt_toolkit.keys import Keys

        # Share the prompt's input so its incremental decoder and VT100 parser
        # keep partial characters, escape sequences and pastes across turns.
        self._input = input
        self._submit_key = Keys.ControlJ if multiline else Keys.ControlM
        self._fd = input.fileno()
        self._on_cancel = on_cancel
        self._on_queued = on_queued
        self._queued = 0
        self._write = write
        self._keys: list[KeyPress] = []
        self._submissions: list[_Submission] = []
        self._line = ""
        self._escape_pending = False
        self._escape_timer: asyncio.TimerHandle | None = None
        self._saved_attrs: list[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reading = False
        self._suspended = 0

    @property
    def held_keys(self) -> list[KeyPress]:
        return list(self._keys)

    def take_messages(self) -> list[str]:
        """Take a FIFO prefix of plain submitted text at a tool-loop boundary.

        Leave commands, attachments and unfinished drafts to the prompt. Stop
        at the first deferred submission so later messages cannot overtake it.
        """
        from prompt_toolkit.keys import Keys

        from fast_agent.ui.prompt.pasted_paths import pasted_image_paths

        if self._suspended or _replayed_submissions:
            # An earlier turn may still have submissions in prompt typeahead.
            # New terminal input must not overtake that backlog.
            return []
        # A tool may finish before the fd callback runs. Drain bytes already
        # available without waiting for a message that has not arrived.
        self._readable()
        messages: list[str] = []
        consumed_keys = 0
        consumed_submissions = 0
        for submission in self._submissions:
            end, text = submission.end, submission.text
            stripped = text.lstrip()
            if stripped.startswith(("/", "!", "@", "#")) or any(
                word.startswith("^") for word in text.split()
            ):
                break
            if any(
                key.key == Keys.BracketedPaste and pasted_image_paths(key.data)
                for key in self._keys[consumed_keys:end]
            ):
                break
            if stripped:
                messages.append(text)
            consumed_keys = end
            consumed_submissions += 1
        if consumed_submissions:
            self._keys = self._keys[consumed_keys:]
            self._submissions = [
                _Submission(submission.end - consumed_keys, submission.text)
                for submission in self._submissions[consumed_submissions:]
            ]
            self._queued -= len(messages)
            self._on_queued(self._queued)
        return messages

    def start(self) -> bool:
        import termios

        try:
            self._saved_attrs = termios.tcgetattr(self._fd)
        except (termios.error, OSError, ValueError):
            return False
        self._loop = asyncio.get_running_loop()
        if not self._resume():
            return False

        from prompt_toolkit.input.typeahead import get_typeahead

        # The prompt may already have read more than one submission. Take its
        # remaining keys before fresh stdin so they cannot be overtaken.
        typeahead = get_typeahead(self._input)
        _replayed_submissions.clear()
        for key_press in typeahead:
            self._on_key(key_press)
        return True

    def stop(self, *, replay: bool = False) -> list[KeyPress]:
        """Stop reading, restore the terminal and return held key presses."""
        self._pause()
        self._cancel_escape_timer()
        # Leave unfinished input in the shared decoder/parser for the prompt.
        self._escape_pending = False
        self._saved_attrs = None
        if replay:
            # Ownership changes, but the messages are still outstanding until
            # the prompt accepts them. Do not advertise an empty queue here.
            _replayed_submissions.extend(bool(item.text.strip()) for item in self._submissions)
        if self._queued:
            self._queued = 0
            self._on_queued(0)
        return self.held_keys

    @contextmanager
    def suspended(self) -> Generator[None]:
        """Hand the terminal to another reader (a form, an OAuth paste)."""
        self._suspended += 1
        if self._suspended == 1:
            self._call_on_loop(self._pause)
        try:
            yield
        finally:
            self._suspended -= 1
            if self._suspended == 0:
                self._call_on_loop(self._resume)

    def _call_on_loop(self, action: Callable[[], object]) -> None:
        loop = self._loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is None or running is loop or loop.is_closed():
            action()
            return
        # Event-loop readers may only be changed from their own loop thread.
        done = threading.Event()

        def run() -> None:
            try:
                action()
            finally:
                done.set()

        loop.call_soon_threadsafe(run)
        done.wait(1.0)

    def _resume(self) -> bool:
        import termios

        if self._reading or self._saved_attrs is None or self._loop is None:
            return self._reading
        try:
            attrs = termios.tcgetattr(self._fd)
            attrs[0] &= ~(termios.ICRNL | termios.INLCR | termios.IGNCR | termios.IXON)
            attrs[3] &= ~(termios.ICANON | termios.ECHO | termios.IEXTEN)
            attrs[3] |= termios.ISIG
            attrs[6][termios.VMIN] = 1
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
            self._loop.add_reader(self._fd, self._readable)
        except (termios.error, OSError, ValueError, NotImplementedError, RuntimeError):
            self._restore_attrs()
            return False
        self._reading = True
        self._write(_BRACKETED_PASTE_ON)
        return True

    def _pause(self) -> None:
        if not self._reading:
            return
        self._reading = False
        if self._loop is not None:
            try:
                self._loop.remove_reader(self._fd)
            except (OSError, ValueError, RuntimeError):
                pass
        self._write(_BRACKETED_PASTE_OFF)
        self._restore_attrs()

    def _restore_attrs(self) -> None:
        import termios

        if self._saved_attrs is None:
            return
        try:
            termios.tcsetattr(self._fd, termios.TCSANOW, self._saved_attrs)
        except (termios.error, OSError, ValueError):
            pass

    def _cancel_escape_timer(self) -> None:
        if self._escape_timer is not None:
            self._escape_timer.cancel()
            self._escape_timer = None

    def _readable(self) -> None:
        try:
            key_presses = self._input.read_keys()
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            self._pause()
            return
        for key_press in key_presses:
            self._on_key(key_press)
        if self._input.closed:
            self._pause()
            return
        self._cancel_escape_timer()
        if self._loop is not None:
            # The parser holds a lone Escape until no sequence can follow it.
            self._escape_timer = self._loop.call_later(_ESCAPE_TIMEOUT_SECONDS, self.flush)

    def flush(self) -> None:
        self._escape_timer = None
        for key_press in self._input.flush_keys():
            self._on_key(key_press)
        if self._escape_pending:
            self._escape_pending = False
            self._on_cancel()

    def _on_key(self, key_press: KeyPress) -> None:
        from prompt_toolkit.keys import Keys

        key = key_press.key
        if self._escape_pending:
            # Escape followed by a key is an Alt chord, not a cancellation.
            # Prompt shortcuts are not replayed across turns.
            self._escape_pending = False
            return
        if key == Keys.Escape:
            self._escape_pending = True
        elif key == Keys.BracketedPaste:
            self._keys.append(key_press)
            self._line += key_press.data.replace("\r\n", "\n").replace("\r", "\n")
        elif key == self._submit_key:
            self._keys.append(key_press)
            self._submissions.append(_Submission(len(self._keys), self._line))
            if self._line.strip():
                self._queued += 1
                self._on_queued(self._queued)
            self._line = ""
        elif key in (Keys.ControlM, Keys.ControlJ):
            self._keys.append(key_press)
            self._line += "\n"
        elif key == Keys.ControlH:
            self._keys.append(key_press)
            self._line = self._line[:-1]
        elif not isinstance(key, Keys) and key_press.data.isprintable():
            self._keys.append(key_press)
            self._line += key_press.data
        # Other control and navigation keys (Ctrl+D, arrows, Tab, F-keys) are
        # not replayed: they would act on a prompt nobody could see.


def _store_typeahead(input: Input, key_presses: list[KeyPress]) -> None:
    if not key_presses:
        return
    from prompt_toolkit.input.typeahead import store_typeahead

    store_typeahead(input, key_presses)


@contextmanager
def hold_turn_input() -> Generator[TurnInputReader | None]:
    """Queue terminal input for the next prompt while the body runs."""
    global _active

    reader: TurnInputReader | None = None
    if _active is None and turn_input_enabled():
        from prompt_toolkit.application.current import get_app_session

        from fast_agent.ui.prompt.input import in_multiline_mode

        input = get_app_session().input
        candidate = TurnInputReader(input=input, multiline=in_multiline_mode)
        try:
            started = candidate.start()
        except Exception:
            started = False
        if started:
            reader = candidate
            _active = reader
    try:
        yield reader
    finally:
        if reader is not None:
            _active = None
            try:
                _store_typeahead(input, reader.stop(replay=True))
            except Exception:
                pass


@contextmanager
def suspend_turn_input() -> Generator[None]:
    """Give the terminal back to code that reads it directly during a turn."""
    reader = _active
    if reader is None:
        yield
        return
    with reader.suspended():
        yield
