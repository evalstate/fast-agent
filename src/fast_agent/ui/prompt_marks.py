"""Helpers for emitting terminal prompt marks (OSC 133)."""

from __future__ import annotations

import sys
from typing import TextIO

from fast_agent.config import Settings, get_settings

_OSC = "\x1b]"
_ST = "\x07"


def emit_prompt_mark(
    code: str,
    *,
    settings: Settings | None = None,
    stream: TextIO | None = None,
) -> None:
    """Emit an OSC 133 prompt mark if enabled and stdout is a TTY."""
    sequence = prompt_mark_sequence(code, settings=settings, stream=stream)
    if not sequence:
        return

    target_stream = stream or sys.stdout
    target_stream.write(sequence)
    target_stream.flush()


def prompt_mark_sequence(
    code: str,
    *,
    settings: Settings | None = None,
    stream: TextIO | None = None,
) -> str:
    """Return an OSC 133 prompt mark sequence when enabled for a TTY."""
    if not code:
        return ""

    target_stream = stream or sys.stdout
    if not hasattr(target_stream, "isatty") or not target_stream.isatty():
        return ""

    resolved_settings = settings or get_settings()
    if not resolved_settings.logger.enable_prompt_marks:
        return ""

    return f"{_OSC}133;{code}{_ST}"
