"""Compatibility for textual-image's batched terminal probe.

Herdr can reply to primary DA before its Kitty graphics reply, and can emit
duplicate Kitty graphics replies (for example, from both the terminal core and
an attached viewer). textual-image 0.14.2 treats DA as an end marker, caching
a false negative and leaving the Kitty reply in the input stream; stopping
immediately after the first Kitty reply likewise leaves a duplicate reply to
echo into output and leak into the next prompt. Collect all replies and let the
stream settle briefly before restoring cooked mode. Keep the workaround scoped
to our initial probe, not a permanent library patch.
"""

from __future__ import annotations

import re
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING

from textual_image import _terminal as terminal

from fast_agent.ui.prompt.turn_input import suspend_turn_input

if TYPE_CHECKING:
    from collections.abc import Generator

_PRIMARY_DA = re.compile(r"\x1b\[\?[0-9;]*c")
_GRAPHICS_REPLY = re.compile(r"\x1b_G[^\x1b]*\x1b\\")
# Non-graphics terminals never answer the Kitty query. Give asynchronous
# replies a short grace period without imposing the full timeout on them.
_CELL_REPLY = re.compile(r"\x1b\[6;[0-9]+;[0-9]+t")
_REPLY_GRACE_SECONDS = 0.25
# Brief quiet window after expected replies arrive to absorb duplicate graphics
# replies before restoring echo on the TTY.
_REPLY_SETTLE_SECONDS = 0.08
_MAX_REPLY_CHARS = 8192


def _has_incomplete_graphics_reply(sequence: str) -> bool:
    if sequence.endswith(("\x1b", "\x1b_")):
        return True
    last_start = sequence.rfind("\x1b_G")
    if last_start == -1:
        return False
    return sequence.find("\x1b\\", last_start + 3) == -1


@contextmanager
def _capture_replies(timeout: float | None = None) -> Generator[SimpleNamespace]:
    if sys.__stdin__ is None:
        raise terminal.TerminalError("stdin is closed")
    fd = sys.__stdin__.buffer.fileno()
    response = SimpleNamespace(sequence="")
    try:
        rows, cols, width, height = terminal.get_tiocgwinsz()
        need_cell_reply = not (rows and cols and int(width / cols) and int(height / rows))
    except OSError:
        need_cell_reply = True
    with suspend_turn_input(), terminal.capture_mode():
        yield response
        deadline = time.monotonic() + (timeout if timeout is not None else 2.0)
        da_deadline: float | None = None
        last_rx: float | None = None
        while len(response.sequence) < _MAX_REPLY_CHARS:
            has_da = _PRIMARY_DA.search(response.sequence) is not None
            has_cell = not need_cell_reply or _CELL_REPLY.search(response.sequence) is not None
            has_graphics = _GRAPHICS_REPLY.search(response.sequence) is not None
            in_graphics_reply = _has_incomplete_graphics_reply(response.sequence)
            now = time.monotonic()
            if has_da and da_deadline is None:
                da_deadline = min(deadline, now + _REPLY_GRACE_SECONDS)
            if in_graphics_reply:
                base = last_rx if last_rx is not None else now
                end = min(deadline, base + _REPLY_GRACE_SECONDS)
            elif has_da and has_cell and has_graphics:
                base = last_rx if last_rx is not None else now
                end = min(deadline, base + _REPLY_SETTLE_SECONDS)
            else:
                end = da_deadline if da_deadline is not None else deadline
            remaining = end - now
            if remaining <= 0:
                break
            try:
                chunk = terminal.read(fd, 1, remaining)
            except TimeoutError:
                break
            if not chunk:
                break
            response.sequence += chunk
            last_rx = time.monotonic()


def prepare_terminal_image_probe() -> None:
    """Populate the library cache before renderable import selects a backend.

    Use textual-image's platform-specific IO and parser, preserving its tmux,
    Windows and cell-size handling. Never probe redirected/noninteractive IO.
    """
    if not (
        sys.__stdout__ is not None
        and sys.__stdout__.isatty()
        and sys.__stdin__ is not None
        and sys.__stdin__.isatty()
    ):
        return
    original_capture = terminal.capture_until_primary_da
    terminal.capture_until_primary_da = _capture_replies
    try:
        terminal.probe_terminal()
    finally:
        terminal.capture_until_primary_da = original_capture
