"""Compatibility for textual-image's batched terminal probe.

Herdr can reply to primary DA before its Kitty graphics reply. textual-image
0.14.2 treats DA as an end marker, caching a false negative and leaving the
Kitty reply in the input stream. Collect both replies before backend selection.
Keep the workaround scoped to our initial probe, not a permanent library patch.
"""

from __future__ import annotations

import re
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING

from textual_image import _terminal as terminal

if TYPE_CHECKING:
    from collections.abc import Generator

_PRIMARY_DA = re.compile(r"\x1b\[\?[0-9;]*c")
_GRAPHICS_REPLY = re.compile(r"\x1b_G[^\x1b]*\x1b\\")
# Non-graphics terminals never answer the Kitty query. Give asynchronous
# replies a short grace period without imposing the full timeout on them.
_CELL_REPLY = re.compile(r"\x1b\[6;[0-9]+;[0-9]+t")
_REPLY_GRACE_SECONDS = 0.25
_MAX_REPLY_CHARS = 8192


@contextmanager
def _capture_replies(timeout: float | None = None) -> Generator[SimpleNamespace]:
    if sys.__stdin__ is None:
        raise terminal.TerminalError("stdin is closed")
    fd = sys.__stdin__.buffer.fileno()
    response = SimpleNamespace(sequence="")
    try:
        rows, cols, width, height = terminal.get_tiocgwinsz()
        need_cell_reply = not (rows and cols and width and height)
    except OSError:
        need_cell_reply = True
    with terminal.capture_mode():
        yield response
        deadline = time.monotonic() + (timeout if timeout is not None else 2.0)
        da_deadline: float | None = None
        while len(response.sequence) < _MAX_REPLY_CHARS:
            has_da = _PRIMARY_DA.search(response.sequence) is not None
            has_cell = not need_cell_reply or _CELL_REPLY.search(response.sequence) is not None
            if has_da and has_cell and _GRAPHICS_REPLY.search(response.sequence):
                break
            now = time.monotonic()
            if has_da and da_deadline is None:
                da_deadline = min(deadline, now + _REPLY_GRACE_SECONDS)
            end = da_deadline if da_deadline is not None else deadline
            remaining = end - now
            if remaining <= 0:
                break
            try:
                response.sequence += terminal.read(fd, 1, remaining)
            except TimeoutError:
                break


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
