"""Small helpers for terminal-like streams."""

from __future__ import annotations

from typing import Protocol, TypeGuard, runtime_checkable


@runtime_checkable
class TtyCapableStream(Protocol):
    def isatty(self) -> bool: ...

    def write(self, value: str) -> object: ...

    def flush(self) -> object: ...


def is_tty_stream(stream: object) -> TypeGuard[TtyCapableStream]:
    if not isinstance(stream, TtyCapableStream):
        return False
    try:
        return stream.isatty()
    except Exception:
        return False
