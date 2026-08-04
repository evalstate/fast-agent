"""
Centralized console configuration for MCP Agent.

This module provides shared console instances for consistent output handling:
- console: Main console for general output
- error_console: Error console for application errors (writes to stderr)
- server_console: Special console for MCP server output
"""

from __future__ import annotations

import io
import os
import re
import sys
from contextlib import suppress
from importlib.resources import files
from pathlib import Path
from typing import IO, Literal, TextIO, cast

from rich.console import Console
from rich.theme import Theme

from fast_agent.utils.env import is_truthy_env_value

_DEFAULT_THEME_RELATIVE_PATH = Path("examples") / "markdown" / "fast-agent-theme.ini"
_SURROGATE_PATTERN = re.compile(r"[\ud800-\udbff][\udc00-\udfff]|[\ud800-\udfff]")


def _normalize_surrogate_code_points(text: str) -> str:
    """Recombine valid UTF-16 pairs and escape isolated surrogate units."""

    def replace(match: re.Match[str]) -> str:
        value = match.group()
        if len(value) == 2:
            high, low = map(ord, value)
            return chr(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
        return f"\\u{ord(value):04x}"

    return _SURROGATE_PATTERN.sub(replace, text)


class _SurrogateSafeTextIO:
    """Terminal stream adapter that leaves valid Unicode untouched."""

    def __init__(self, stream: IO[str]) -> None:
        self._stream = cast("TextIO", stream)

    @property
    def encoding(self) -> str | None:
        return self._stream.encoding

    @property
    def errors(self) -> str | None:
        return self._stream.errors

    @property
    def closed(self) -> bool:
        return self._stream.closed

    def fileno(self) -> int:
        return self._stream.fileno()

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def writable(self) -> bool:
        return self._stream.writable()

    def write(self, text: str) -> int:
        self._stream.write(_normalize_surrogate_code_points(text))
        return len(text)


class SurrogateSafeConsole(Console):
    """Rich console that cannot crash while encoding malformed surrogates."""

    _surrogate_safe_source: IO[str] | None = None
    _surrogate_safe_file: IO[str] | None = None

    @property
    def file(self) -> IO[str]:
        source = super().file
        if source is not self._surrogate_safe_source:
            self._surrogate_safe_source = source
            self._surrogate_safe_file = cast("IO[str]", _SurrogateSafeTextIO(source))
        safe_file = self._surrogate_safe_file
        assert safe_file is not None
        return safe_file

    @file.setter
    def file(self, new_file: IO[str]) -> None:
        self._file = new_file
        self._surrogate_safe_source = None
        self._surrogate_safe_file = None


def _load_default_theme() -> Theme:
    source_theme = Path(__file__).resolve().parents[3] / _DEFAULT_THEME_RELATIVE_PATH
    if source_theme.is_file():
        return Theme.read(str(source_theme))

    packaged_theme = (
        files("fast_agent")
        .joinpath("resources")
        .joinpath("examples")
        .joinpath("markdown")
        .joinpath("fast-agent-theme.ini")
    )
    if packaged_theme.is_file():
        return Theme.from_file(
            io.StringIO(packaged_theme.read_text(encoding="utf-8")),
            source=str(_DEFAULT_THEME_RELATIVE_PATH),
        )

    return Theme()


_DEFAULT_THEME = _load_default_theme()


# When uvloop registers a reader, it makes the file description non-blocking
# and doesn't restore it. If stdin/stdout/stderr share the same TTY, writes
# can raise BlockingIOError. Use a dedicated blocking TTY stream when needed.
_blocking_console_file: IO[str] | None = None
_theme_applied = False
_theme_path: Path | None = None


def _open_blocking_tty(stream: IO[str]) -> IO[str] | None:
    try:
        fd = stream.fileno()
    except Exception:
        return None
    if not os.isatty(fd):
        return None
    try:
        tty_path = os.ttyname(fd)
    except OSError:
        tty_path = "/dev/tty"
    try:
        tty_fd = os.open(tty_path, os.O_WRONLY | os.O_NOCTTY)
    except OSError:
        return None
    with suppress(Exception):
        os.set_blocking(tty_fd, True)
    return os.fdopen(tty_fd, "w", buffering=1, encoding="utf-8", errors="replace")


def _redirect_standard_stream_to_blocking_tty(
    source_fd: int,
    blocking_stream: IO[str],
) -> None:
    for stream in (sys.__stdout__, sys.__stderr__):
        if stream is None:
            continue
        try:
            if stream.fileno() == source_fd:
                os.dup2(blocking_stream.fileno(), source_fd)
                return
        except (OSError, ValueError):
            continue


def ensure_blocking_console() -> None:
    """
    Ensure the shared console writes to a blocking TTY stream when stdout/stderr
    has been made non-blocking by the event loop.
    """
    global _blocking_console_file

    current_file = console.file
    try:
        if os.get_blocking(current_file.fileno()):
            return
    except Exception:
        return

    if _blocking_console_file is None or _blocking_console_file.closed:
        _blocking_console_file = _open_blocking_tty(current_file)
    if _blocking_console_file is not None:
        _redirect_standard_stream_to_blocking_tty(current_file.fileno(), _blocking_console_file)
        console.file = _blocking_console_file


def _shared_consoles() -> tuple[Console, ...]:
    return (console, error_console, server_console)


def configure_console_theme(
    theme_file: str | os.PathLike[str] | None,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Apply or clear a shared Rich theme for all app consoles."""
    global _theme_applied, _theme_path

    target_path: Path | None = None
    if theme_file:
        target_path = Path(theme_file).expanduser()
        if not target_path.is_absolute() and base_dir is not None:
            target_path = Path(base_dir).expanduser() / target_path
        target_path = target_path.resolve()

    if _theme_applied and target_path == _theme_path:
        return _theme_path

    if _theme_applied:
        for shared_console in _shared_consoles():
            shared_console.pop_theme()
        _theme_applied = False
        _theme_path = None

    if target_path is None:
        return None

    theme = Theme.read(str(target_path))
    for shared_console in _shared_consoles():
        shared_console.push_theme(theme)

    _theme_applied = True
    _theme_path = target_path
    return target_path


# Allow forcing stderr via env (useful for ACP/stdio wrappers that import fast_agent early)
_default_stderr = is_truthy_env_value(os.environ.get("FAST_AGENT_FORCE_STDERR"))

# Main console for general output (stdout by default, can be toggled at runtime)
console: Console = SurrogateSafeConsole(
    stderr=_default_stderr,
    color_system="auto",
    theme=_DEFAULT_THEME,
)


def configure_console_stream(stream: Literal["stdout", "stderr"]) -> None:
    """
    Route the shared console to stdout (default) or stderr (required for stdio/ACP servers).
    """
    target_is_stderr = stream == "stderr"
    if console.stderr == target_is_stderr:
        return

    # Reset the underlying stream selection so Console.file uses the new stderr flag
    console._file = None
    console.stderr = target_is_stderr
    ensure_blocking_console()


# Error console for application errors
error_console: Console = SurrogateSafeConsole(
    stderr=True,
    style="bold red",
    theme=_DEFAULT_THEME,
)

# Special console for MCP server output
# This could have custom styling to distinguish server messages
server_console: Console = SurrogateSafeConsole(
    # Not stderr since we want to maintain output ordering with other messages
    style="dim blue",  # Or whatever style makes server output distinct
    theme=_DEFAULT_THEME,
)

# Drop-in replacement for Rich's module-level print that follows shared console routing.
rich_print = console.print
