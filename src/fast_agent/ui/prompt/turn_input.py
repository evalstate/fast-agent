"""Hold terminal input that arrives while an interactive turn is running.

Between prompts the terminal is otherwise left in cooked mode: typed or pasted
text is echoed into the streaming output, and the kernel translates Enter into
Ctrl+J, which the prompt treats as "insert newline". Messages typed ahead, or
sent by terminal managers such as herdr-core (``hc send``), were therefore left
unsubmitted in the next prompt.

While a turn runs this reader keeps the terminal in a non-echoing, untranslated
mode (signals stay enabled, so Ctrl+C is unchanged) and drains input as key
presses. Text followed by Enter is queued and replayed into the next prompt as
prompt_toolkit typeahead, so queued messages are submitted in order through the
normal prompt path (commands, attachments, history). A lone Escape cancels the
turn exactly like Ctrl+C, which is also what ``hc interrupt`` sends.

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
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from fast_agent.utils.env import is_truthy_env_value

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from prompt_toolkit.key_binding import KeyPress

_DISABLE_ENV = "FAST_AGENT_DISABLE_TURN_INPUT"
_ESCAPE_TIMEOUT_SECONDS = 0.05
_BRACKETED_PASTE_ON = "\x1b[?2004h"
_BRACKETED_PASTE_OFF = "\x1b[?2004l"
_PREVIEW_CHARS = 72

_active: TurnInputReader | None = None
_queued_count = 0


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

    _queued_count = count
    from fast_agent.integrations.herdr_lifecycle import report_queued_input

    report_queued_input(count)


def queued_input_count() -> int:
    """Messages waiting to be submitted when the current turn ends."""
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
        fd: int,
        on_cancel: Callable[[], None] = _default_cancel,
        on_queued: Callable[[int], None] = _publish_queued_count,
        write: Callable[[str], None] = _write_stdout,
    ) -> None:
        from prompt_toolkit.input.vt100_parser import Vt100Parser

        self._fd = fd
        self._on_cancel = on_cancel
        self._on_queued = on_queued
        self._queued = 0
        self._write = write
        self._parser = Vt100Parser(self._on_key)
        self._keys: list[KeyPress] = []
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

    def start(self) -> bool:
        import termios

        try:
            self._saved_attrs = termios.tcgetattr(self._fd)
        except (termios.error, OSError, ValueError):
            return False
        self._loop = asyncio.get_running_loop()
        return self._resume()

    def stop(self) -> list[KeyPress]:
        """Stop reading, restore the terminal and return held key presses."""
        self._pause()
        self._cancel_escape_timer()
        # The turn is over: a trailing Escape no longer means "cancel".
        self._parser.flush()
        self._escape_pending = False
        self._saved_attrs = None
        if self._queued:
            # Held messages now belong to the next prompt, which shows them.
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
            data = os.read(self._fd, 4096)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            self._pause()
            return
        if not data:
            self._pause()
            return
        self.feed(data.decode("utf-8", errors="replace"))

    def feed(self, data: str) -> None:
        self._parser.feed(data)
        self._cancel_escape_timer()
        if self._loop is not None:
            # The parser holds a lone Escape until no sequence can follow it.
            self._escape_timer = self._loop.call_later(_ESCAPE_TIMEOUT_SECONDS, self.flush)

    def flush(self) -> None:
        self._escape_timer = None
        self._parser.flush()
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
            self._line += key_press.data
        elif key == Keys.ControlM:
            self._keys.append(key_press)
            if self._line.strip():
                self._queued += 1
                self._on_queued(self._queued)
            self._line = ""
        elif key == Keys.ControlJ:
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


def _store_typeahead(key_presses: list[KeyPress]) -> None:
    if not key_presses:
        return
    from prompt_toolkit.input import create_input
    from prompt_toolkit.input.typeahead import store_typeahead

    store_typeahead(create_input(), key_presses)


@contextmanager
def hold_turn_input() -> Generator[TurnInputReader | None]:
    """Queue terminal input for the next prompt while the body runs."""
    global _active

    reader: TurnInputReader | None = None
    if _active is None and turn_input_enabled():
        candidate = TurnInputReader(fd=sys.stdin.fileno())
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
                _store_typeahead(reader.stop())
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
