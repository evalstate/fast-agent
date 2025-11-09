"""
Centralized console configuration for MCP Agent.

This module provides shared console instances for consistent output handling:
- console: Main console for general output
- error_console: Error console for application errors (writes to stderr)
- server_console: Special console for MCP server output
"""

from __future__ import annotations

import os
from typing import Literal

from rich.console import Console


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _create_console(stderr: bool) -> Console:
    return Console(stderr=stderr, color_system="auto")


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
    console._file = None  # type: ignore[attr-defined]
    console.stderr = target_is_stderr


# Error console for application errors
error_console = Console(
    stderr=True,
    style="bold red",
)

# Special console for MCP server output
# This could have custom styling to distinguish server messages
server_console = Console(
    # Not stderr since we want to maintain output ordering with other messages
    style="dim blue",  # Or whatever style makes server output distinct
)
