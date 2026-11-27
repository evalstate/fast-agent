"""Hold and edit terminal input that arrives while an interactive turn is running.

Between prompts the terminal is otherwise left in cooked mode: typed or pasted
text is echoed into the streaming output, and the kernel translates Enter into
Ctrl+J, which the prompt treats as "insert newline". Messages typed ahead, or
sent by terminal managers such as herdr-core (``hc send``), were therefore left
unsubmitted in the next prompt.

While a turn runs this reader keeps the terminal in a non-echoing, untranslated
mode (signals stay enabled, so Ctrl+C is unchanged) and drains input as key
presses into a live ``prompt_toolkit`` ``Buffer``. The live streaming and
progress renderers display the active draft and queued submissions in an inline
bottom dock. Plain submitted text can be consumed at tool-loop continuation
boundaries. Commands, attachments, late messages and drafts are replayed into
the next prompt as prompt_toolkit typeahead. Submission order is preserved
across both paths. A lone Escape cancels the turn exactly like Ctrl+C, which is
also what ``hc interrupt`` sends.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
from collections import deque
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fast_agent.core.logging.logger import get_logger
from fast_agent.ui.prompt.turn_input_dock import (
    DEFAULT_PROMPT_PREFIX,
    TurnInputCompletion,
    TurnInputSnapshot,
)
from fast_agent.utils.env import is_truthy_env_value

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from prompt_toolkit.completion import Completer
    from prompt_toolkit.input import Input
    from prompt_toolkit.key_binding import KeyPress

logger = get_logger(__name__)

_DISABLE_ENV = "FAST_AGENT_DISABLE_TURN_INPUT"
_ESCAPE_TIMEOUT_SECONDS = 0.05
_MAX_COMPLETION_ITEMS = 24
_BRACKETED_PASTE_ON = "\x1b[?2004h"
_BRACKETED_PASTE_OFF = "\x1b[?2004l"

_active: TurnInputReader | None = None
_queued_count = 0
_replayed_submissions: deque[bool] = deque()
_listeners: list[Callable[[], None]] = []
_listeners_lock = threading.Lock()


@dataclass(frozen=True)
class _Submission:
    text: str
    image_pastes: tuple[str, ...] = ()

    @property
    def has_pasted_image(self) -> bool:
        return bool(self.image_pastes)


@dataclass(frozen=True)
class _DraftState:
    text: str
    cursor_position: int
    image_pastes: tuple[str, ...] = ()


def add_turn_input_listener(callback: Callable[[], None]) -> Callable[[], None]:
    """Subscribe to live turn-input state changes; returns an unsubscribe callback."""
    with _listeners_lock:
        _listeners.append(callback)

    def _unsubscribe() -> None:
        with _listeners_lock, suppress(ValueError):
            _listeners.remove(callback)

    return _unsubscribe


def _notify_turn_input_listeners() -> None:
    with _listeners_lock:
        listeners = tuple(_listeners)
    for listener in listeners:
        try:
            listener()
        except Exception:
            logger.debug("Turn input listener failed", exc_info=True)


def active_turn_input_snapshot() -> TurnInputSnapshot | None:
    """Return a snapshot of the active, non-suspended turn input dock, if any."""
    reader = _active
    if reader is None or reader.is_suspended:
        return None
    return reader.snapshot()


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


def _paste_keys_preserving_images(
    text: str,
    image_pastes: tuple[str, ...],
) -> list[KeyPress]:
    """Emit BracketedPaste key(s), keeping image-path pastes isolated for attachment parsing."""
    from prompt_toolkit.key_binding import KeyPress
    from prompt_toolkit.keys import Keys

    if not text:
        return []
    if not image_pastes:
        return [KeyPress(Keys.BracketedPaste, text)]

    remaining = list(image_pastes)
    keys: list[KeyPress] = []
    pos = 0
    while remaining:
        best_idx = -1
        best_pos = -1
        for i, paste in enumerate(remaining):
            found = text.find(paste, pos)
            if found >= 0 and (best_pos < 0 or found < best_pos):
                best_pos = found
                best_idx = i
        if best_idx < 0:
            break
        paste = remaining.pop(best_idx)
        if best_pos > pos:
            keys.append(KeyPress(Keys.BracketedPaste, text[pos:best_pos]))
        keys.append(KeyPress(Keys.BracketedPaste, paste))
        pos = best_pos + len(paste)
    if pos < len(text):
        keys.append(KeyPress(Keys.BracketedPaste, text[pos:]))
    return keys or [KeyPress(Keys.BracketedPaste, text)]


def _cursor_replay_keys(text: str, cursor_position: int) -> list[KeyPress]:
    """Build row-aware navigation keys to restore cursor position after a BracketedPaste."""
    from prompt_toolkit.document import Document
    from prompt_toolkit.key_binding import KeyPress
    from prompt_toolkit.keys import Keys

    if not text or cursor_position >= len(text):
        return []
    doc = Document(text, cursor_position=max(0, cursor_position))
    rows_from_end = doc.line_count - 1 - doc.cursor_position_row
    if rows_from_end == 0:
        trailing = len(doc.current_line) - doc.cursor_position_col
        return [KeyPress(Keys.Left, "\x1b[D") for _ in range(trailing)]

    keys: list[KeyPress] = [KeyPress(Keys.Up, "\x1b[A") for _ in range(rows_from_end)]
    keys.append(KeyPress(Keys.ControlA, "\x01"))
    keys.extend(KeyPress(Keys.Right, "\x1b[C") for _ in range(doc.cursor_position_col))
    return keys


class TurnInputReader:
    """Drain stdin during a turn into a live buffer; queue submitted messages."""

    def __init__(
        self,
        *,
        input: Input,
        multiline: bool = False,
        prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
        completer: Completer | None = None,
        on_submit: Callable[[str], bool] | None = None,
        on_cancel: Callable[[], None] = _default_cancel,
        on_queued: Callable[[int], None] = _publish_queued_count,
        write: Callable[[str], None] = _write_stdout,
    ) -> None:
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.keys import Keys

        # Share the prompt's input so its incremental decoder and VT100 parser
        # keep partial characters, escape sequences and pastes across turns.
        self._input = input
        self._multiline = multiline
        self._prompt_prefix = prompt_prefix
        self._completer = completer
        self._on_submit = on_submit
        self._submit_key = Keys.ControlJ if multiline else Keys.ControlM
        self._submit_data = "\n" if multiline else "\r"
        self._fd = input.fileno()
        self._on_cancel = on_cancel
        self._on_queued = on_queued
        self._queued = 0
        self._write = write
        self._submissions: list[_Submission] = []
        self._buffer = Buffer(multiline=True)
        self._draft_image_pastes: list[str] = []
        # A draft set aside while an hc message pasted behind it is submitted.
        self._saved_draft: _DraftState | None = None
        self._completions: tuple[TurnInputCompletion, ...] = ()
        self._selected_completion = 0
        self._completion_seq = 0
        self._completion_task: asyncio.Task[None] | None = None
        self._escape_pending = False
        self._escape_timer: asyncio.TimerHandle | None = None
        self._saved_attrs: list[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reading = False
        self._suspended = 0

    @property
    def is_suspended(self) -> bool:
        return self._suspended > 0

    @property
    def draft_text(self) -> str:
        return self._buffer.text

    @property
    def cursor_position(self) -> int:
        return self._buffer.cursor_position

    @property
    def queued_messages(self) -> tuple[str, ...]:
        return tuple(item.text for item in self._submissions if item.text.strip())

    def snapshot(self) -> TurnInputSnapshot:
        return TurnInputSnapshot(
            prompt_prefix=self._prompt_prefix,
            draft_text=self._buffer.text,
            cursor_position=self._buffer.cursor_position,
            queued_messages=self.queued_messages,
            multiline=self._multiline,
            completions=self._completions,
            selected_completion=self._selected_completion,
        )

    def _state_signature(
        self,
    ) -> tuple[str, int, tuple[str, ...], tuple[TurnInputCompletion, ...], int]:
        return (
            self._buffer.text,
            self._buffer.cursor_position,
            self.queued_messages,
            self._completions,
            self._selected_completion,
        )

    @property
    def held_keys(self) -> list[KeyPress]:
        from prompt_toolkit.key_binding import KeyPress

        keys: list[KeyPress] = []
        for submission in self._submissions:
            keys.extend(_paste_keys_preserving_images(submission.text, submission.image_pastes))
            keys.append(KeyPress(self._submit_key, self._submit_data))

        # Replay a set-aside draft before an unsubmitted hc paste so the prompt's
        # BracketedPaste handler can split them apart again.
        if self._saved_draft is not None and self._saved_draft.text:
            keys.extend(
                _paste_keys_preserving_images(
                    self._saved_draft.text, self._saved_draft.image_pastes
                )
            )

        if self._buffer.text:
            keys.extend(
                _paste_keys_preserving_images(self._buffer.text, tuple(self._draft_image_pastes))
            )
            if self._saved_draft is None and not self._draft_image_pastes:
                keys.extend(_cursor_replay_keys(self._buffer.text, self._buffer.cursor_position))

        return keys

    def _set_draft_aside(self) -> None:
        self._saved_draft = _DraftState(
            text=self._buffer.text,
            cursor_position=self._buffer.cursor_position,
            image_pastes=tuple(self._draft_image_pastes),
        )
        self._buffer.reset()
        self._draft_image_pastes = []

    def _restore_draft(self) -> None:
        if self._saved_draft is None:
            return
        from prompt_toolkit.document import Document

        saved = self._saved_draft
        self._saved_draft = None
        self._buffer.set_document(
            Document(saved.text, cursor_position=min(saved.cursor_position, len(saved.text)))
        )
        self._draft_image_pastes = list(saved.image_pastes)

    def take_messages(self) -> list[str]:
        """Take a FIFO prefix of plain submitted text at a tool-loop boundary.

        Leave commands, attachments and unfinished drafts to the prompt. Stop
        at the first deferred submission so later messages cannot overtake it.
        """
        from fast_agent.integrations.hc_children import is_hc_message
        from fast_agent.ui.prompt.special_commands import is_bare_prompt_command

        if self._suspended or _replayed_submissions:
            # An earlier turn may still have submissions in prompt typeahead.
            # New terminal input must not overtake that backlog.
            return []
        # A tool may finish before the fd callback runs. Drain bytes already
        # available without waiting for a message that has not arrived.
        self._readable()
        messages: list[str] = []
        consumed_submissions = 0
        for submission in self._submissions:
            text = submission.text
            stripped = text.lstrip()
            if not is_hc_message(text) and (
                stripped.startswith(("/", "!", "@", "#"))
                or is_bare_prompt_command(text)
                or any(word.startswith("^") for word in text.split())
            ):
                break
            if submission.has_pasted_image:
                break
            if stripped:
                messages.append(text)
            consumed_submissions += 1
        if consumed_submissions:
            self._submissions = self._submissions[consumed_submissions:]
            self._queued -= len(messages)
            self._on_queued(self._queued)
            _notify_turn_input_listeners()
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
            _notify_turn_input_listeners()
        try:
            yield
        finally:
            self._suspended -= 1
            if self._suspended == 0:
                self._call_on_loop(self._resume)
                _notify_turn_input_listeners()

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

    def _clear_completions(self) -> None:
        self._completion_seq += 1
        if self._completion_task is not None:
            self._completion_task.cancel()
            self._completion_task = None
        self._completions = ()
        self._selected_completion = 0

    @staticmethod
    def _collect_completions(
        completer: Completer,
        document: Any,
        *,
        explicit: bool,
    ) -> tuple[TurnInputCompletion, ...]:
        from prompt_toolkit.completion import CompleteEvent

        event = CompleteEvent(text_inserted=not explicit, completion_requested=explicit)
        items: list[TurnInputCompletion] = []
        for comp in completer.get_completions(document, event):
            items.append(
                TurnInputCompletion(
                    text=comp.text,
                    display=comp.display_text or comp.text,
                    display_meta=comp.display_meta_text or "",
                    start_position=comp.start_position,
                )
            )
            if len(items) >= _MAX_COMPLETION_ITEMS:
                break
        return tuple(items)

    def _apply_completion(self, item: TurnInputCompletion) -> bool:
        before = self._buffer.text
        if item.start_position < 0:
            self._buffer.delete_before_cursor(count=-item.start_position)
        self._buffer.insert_text(item.text)
        self._clear_completions()
        return self._buffer.text != before

    def _accept_selected_completion(self) -> bool:
        if not self._completions:
            return False
        idx = max(0, min(self._selected_completion, len(self._completions) - 1))
        return self._apply_completion(self._completions[idx])

    def _apply_completion_results(
        self,
        seq: int,
        items: tuple[TurnInputCompletion, ...],
        *,
        explicit: bool,
    ) -> None:
        if seq != self._completion_seq:
            return
        if explicit and len(items) == 1:
            self._apply_completion(items[0])
            _notify_turn_input_listeners()
            return
        if items != self._completions or self._selected_completion != 0:
            self._completions = items
            self._selected_completion = 0
            _notify_turn_input_listeners()

    def _request_completions(self, *, explicit: bool) -> None:
        completer = self._completer
        if completer is None:
            return
        self._completion_seq += 1
        seq = self._completion_seq
        if self._completion_task is not None:
            self._completion_task.cancel()
            self._completion_task = None
        document = self._buffer.document
        loop = self._loop
        if loop is None or loop.is_closed():
            try:
                items = self._collect_completions(completer, document, explicit=explicit)
            except Exception:
                logger.debug("Turn input completion failed", exc_info=True)
                items = ()
            if explicit and len(items) == 1:
                self._apply_completion(items[0])
            else:
                self._completions = items
                self._selected_completion = 0
            return

        async def _run() -> None:
            try:
                items = await loop.run_in_executor(
                    None,
                    lambda: self._collect_completions(completer, document, explicit=explicit),
                )
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("Turn input completion failed", exc_info=True)
                items = ()
            self._apply_completion_results(seq, items, explicit=explicit)

        self._completion_task = loop.create_task(_run())

    def _maybe_refresh_completions(self) -> None:
        if self._completer is None:
            return
        text = self._buffer.text
        if not text:
            if self._completions:
                self._clear_completions()
            return
        if self._completions or text.startswith(("/", "@", "#")):
            self._request_completions(explicit=False)

    def _pause(self) -> None:
        self._clear_completions()
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
        before = self._state_signature()
        for key_press in key_presses:
            self._on_key(key_press)
        if self._state_signature() != before:
            _notify_turn_input_listeners()
        if self._input.closed:
            self._pause()
            return
        self._cancel_escape_timer()
        if self._loop is not None:
            # The parser holds a lone Escape until no sequence can follow it.
            self._escape_timer = self._loop.call_later(_ESCAPE_TIMEOUT_SECONDS, self.flush)

    def flush(self) -> None:
        self._escape_timer = None
        before = self._state_signature()
        for key_press in self._input.flush_keys():
            self._on_key(key_press)
        if self._state_signature() != before:
            _notify_turn_input_listeners()
        if self._escape_pending:
            self._escape_pending = False
            if self._completions:
                self._clear_completions()
                _notify_turn_input_listeners()
            else:
                self._on_cancel()

    def _move_cursor(self, delta: int | None) -> None:
        if delta:
            self._buffer.cursor_position += delta

    def _move_word_left(self) -> None:
        self._move_cursor(self._buffer.document.find_previous_word_beginning())

    def _move_word_right(self) -> None:
        self._move_cursor(self._buffer.document.find_next_word_ending())

    def _delete_word_before_cursor(self) -> None:
        pos = self._buffer.document.find_previous_word_beginning()
        if pos:
            self._buffer.delete_before_cursor(count=-pos)

    def _pop_last_queued_submission(self) -> None:
        for index in range(len(self._submissions) - 1, -1, -1):
            candidate = self._submissions[index]
            if candidate.text.strip():
                del self._submissions[index]
                self._queued = max(0, self._queued - 1)
                self._on_queued(self._queued)
                self._buffer.reset()
                self._buffer.insert_text(candidate.text)
                self._draft_image_pastes = list(candidate.image_pastes)
                return

    def _handle_alt_chord(self, key_press: KeyPress) -> None:
        from prompt_toolkit.keys import Keys

        key = key_press.key
        if key in (Keys.ControlM, Keys.ControlJ):
            self._buffer.insert_text("\n")
        elif key == Keys.ControlH:
            self._delete_word_before_cursor()
        elif key == Keys.Left or key_press.data == "b":
            self._move_word_left()
        elif key == Keys.Right or key_press.data == "f":
            self._move_word_right()

    def _on_key(self, key_press: KeyPress) -> None:
        from prompt_toolkit.keys import Keys

        key = key_press.key
        if self._escape_pending:
            # Escape followed by a key is an Alt chord, not a cancellation.
            self._escape_pending = False
            self._handle_alt_chord(key_press)
            return
        if key == Keys.Escape:
            self._escape_pending = True
        elif key == Keys.BracketedPaste:
            from fast_agent.integrations.hc_children import is_hc_message
            from fast_agent.ui.prompt.pasted_paths import pasted_image_paths

            if self._buffer.text and self._saved_draft is None and is_hc_message(key_press.data):
                # hc pastes behind whatever the person was typing. Submit its
                # message alone; the draft stays a draft.
                self._set_draft_aside()
            pasted = key_press.data.replace("\r\n", "\n").replace("\r", "\n")
            self._buffer.insert_text(pasted)
            if pasted_image_paths(key_press.data):
                self._draft_image_pastes.append(pasted)
        elif key == Keys.ControlI:
            if len(self._completions) == 1:
                self._accept_selected_completion()
            elif self._completions:
                self._selected_completion = (self._selected_completion + 1) % len(self._completions)
            elif self._completer is not None:
                self._request_completions(explicit=True)
        elif key == Keys.BackTab:
            if self._completions:
                self._selected_completion = (self._selected_completion - 1) % len(self._completions)
        elif key == self._submit_key:
            if self._completions:
                changed = self._accept_selected_completion()
                if changed:
                    return
            self._clear_completions()
            text = self._buffer.text
            active_image_pastes = tuple(p for p in self._draft_image_pastes if p in text)
            handled_concurrently = False
            if self._on_submit is not None and text.strip() and not active_image_pastes:
                from fast_agent.integrations.hc_children import is_hc_message

                if not is_hc_message(text):
                    handled_concurrently = bool(self._on_submit(text))
            if not handled_concurrently:
                self._submissions.append(_Submission(text=text, image_pastes=active_image_pastes))
                if text.strip():
                    self._queued += 1
                    self._on_queued(self._queued)
            self._buffer.reset()
            self._draft_image_pastes = []
            self._restore_draft()
        elif key in (Keys.ControlM, Keys.ControlJ):
            self._clear_completions()
            self._buffer.insert_text("\n")
        elif key == Keys.ControlH:
            self._buffer.delete_before_cursor(1)
            self._maybe_refresh_completions()
        elif key == Keys.Delete:
            self._buffer.delete(1)
            self._maybe_refresh_completions()
        elif key == Keys.Left:
            self._buffer.cursor_left(1)
            if self._completions:
                self._maybe_refresh_completions()
        elif key == Keys.Right:
            self._buffer.cursor_right(1)
            if self._completions:
                self._maybe_refresh_completions()
        elif key == Keys.ControlLeft:
            self._move_word_left()
            if self._completions:
                self._maybe_refresh_completions()
        elif key == Keys.ControlRight:
            self._move_word_right()
            if self._completions:
                self._maybe_refresh_completions()
        elif key in (Keys.Home, Keys.ControlA):
            self._move_cursor(self._buffer.document.get_start_of_line_position())
            if self._completions:
                self._clear_completions()
        elif key in (Keys.End, Keys.ControlE):
            self._move_cursor(self._buffer.document.get_end_of_line_position())
            if self._completions:
                self._maybe_refresh_completions()
        elif key == Keys.ControlW:
            self._delete_word_before_cursor()
            self._maybe_refresh_completions()
        elif key == Keys.ControlU:
            delta = self._buffer.document.get_start_of_line_position()
            if delta < 0:
                self._buffer.delete_before_cursor(count=-delta)
            self._maybe_refresh_completions()
        elif key == Keys.ControlK:
            delta = self._buffer.document.get_end_of_line_position()
            if delta > 0:
                self._buffer.delete(count=delta)
            self._maybe_refresh_completions()
        elif key == Keys.Up:
            if self._completions:
                self._selected_completion = (self._selected_completion - 1) % len(self._completions)
            elif self._buffer.document.cursor_position_row > 0:
                self._buffer.cursor_up(1)
            elif not self._buffer.text and self._submissions:
                self._pop_last_queued_submission()
        elif key == Keys.Down:
            if self._completions:
                self._selected_completion = (self._selected_completion + 1) % len(self._completions)
            elif self._buffer.document.cursor_position_row < self._buffer.document.line_count - 1:
                self._buffer.cursor_down(1)
        elif not isinstance(key, Keys) and key_press.data.isprintable():
            self._buffer.insert_text(key_press.data)
            self._maybe_refresh_completions()


def _store_typeahead(input: Input, key_presses: list[KeyPress]) -> None:
    if not key_presses:
        return
    from prompt_toolkit.input.typeahead import store_typeahead

    store_typeahead(input, key_presses)


def keep_draft_for_next_prompt(input: Input, draft: str) -> None:
    """Return an unfinished draft to the prompt after the current submission."""
    from prompt_toolkit.key_binding import KeyPress
    from prompt_toolkit.keys import Keys

    _store_typeahead(input, [KeyPress(Keys.BracketedPaste, draft)])


@contextmanager
def hold_turn_input(
    *,
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
    completer: Completer | None = None,
    on_submit: Callable[[str], bool] | None = None,
) -> Generator[TurnInputReader | None]:
    """Queue terminal input for the next prompt while the body runs."""
    global _active

    reader: TurnInputReader | None = None
    if _active is None and turn_input_enabled():
        from prompt_toolkit.application.current import get_app_session

        from fast_agent.ui.prompt.input import in_multiline_mode

        input = get_app_session().input
        candidate = TurnInputReader(
            input=input,
            multiline=in_multiline_mode,
            prompt_prefix=prompt_prefix,
            completer=completer,
            on_submit=on_submit,
        )
        try:
            started = candidate.start()
        except Exception:
            started = False
        if started:
            reader = candidate
            _active = reader
            _notify_turn_input_listeners()
    try:
        yield reader
    finally:
        if reader is not None:
            _active = None
            _notify_turn_input_listeners()
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
