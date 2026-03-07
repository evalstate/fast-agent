"""Request-scoped helpers for suppressing interactive display output."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Iterator

InteractiveDisplayMode: TypeAlias = Literal["normal", "progress_only"]

_interactive_display_mode: ContextVar[InteractiveDisplayMode] = ContextVar(
    "interactive_display_mode",
    default="normal",
)


def interactive_display_mode() -> InteractiveDisplayMode:
    """Return the active request-scoped interactive display mode."""
    return _interactive_display_mode.get()


@contextmanager
def suppress_interactive_display(
    mode: InteractiveDisplayMode = "progress_only",
) -> Iterator[None]:
    """Temporarily suppress interactive transcript-oriented rendering."""
    token = _interactive_display_mode.set(mode)
    try:
        yield
    finally:
        _interactive_display_mode.reset(token)


def display_chat_enabled() -> bool:
    """Return True when chat-style interactive rendering is enabled."""
    return interactive_display_mode() == "normal"


def display_tools_enabled() -> bool:
    """Return True when tool call/result rendering is enabled."""
    return interactive_display_mode() == "normal"


def display_status_enabled() -> bool:
    """Return True when transient status-line rendering is enabled."""
    return interactive_display_mode() == "normal"


def display_usage_enabled() -> bool:
    """Return True when post-turn usage rendering is enabled."""
    return interactive_display_mode() == "normal"
