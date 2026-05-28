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
from importlib.resources import files
from pathlib import Path
from typing import IO, Literal

from rich.console import Console
from rich.theme import Theme

_DEFAULT_THEME_RELATIVE_PATH = Path("examples") / "markdown" / "fast-agent-theme.ini"


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


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


def _create_console(stderr: bool) -> Console:
    return Console(stderr=stderr, color_system="auto", theme=_DEFAULT_THEME)


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
    try:
        os.set_blocking(tty_fd, True)
    except Exception:
        pass
    return os.fdopen(tty_fd, "w", buffering=1, encoding="utf-8", errors="replace")


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
_default_stderr = _env_truthy(os.environ.get("FAST_AGENT_FORCE_STDERR"))

# Main console for general output (stdout by default, can be toggled at runtime)
console = _create_console(stderr=_default_stderr)


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
error_console = Console(
    stderr=True,
    style="bold red",
    theme=_DEFAULT_THEME,
)

# Special console for MCP server output
# This could have custom styling to distinguish server messages
server_console = Console(
    # Not stderr since we want to maintain output ordering with other messages
    style="dim blue",  # Or whatever style makes server output distinct
    theme=_DEFAULT_THEME,
)
