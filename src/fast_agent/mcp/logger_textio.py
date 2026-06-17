"""
Utilities for MCP stdio client integration with our logging system.
"""

import io
import os
from collections.abc import Callable
from contextlib import suppress
from typing import TextIO

from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


class LoggerTextIO(io.StringIO):
    """
    A TextIO implementation that logs to our application logger.
    This implements the full TextIO interface as specified by Python.

    Args:
        server_name: The name of the server to include in logs
    """

    def __init__(
        self,
        server_name: str,
        *,
        on_line: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.server_name = server_name
        self._on_line = on_line
        # Keep track of complete and partial lines
        self._line_buffer = ""
        self._devnull_fd: int | None = None

    def _emit_line(self, line: str) -> None:
        if not line.strip():
            return

        logger.debug(f"{self.server_name} (stderr): {line}")

        if self._on_line is None:
            return
        try:
            self._on_line(line)
        except Exception:
            logger.debug("%s: stderr line hook failed", self.server_name, exc_info=True)

    def write(self, s: str) -> int:
        """
        Write data to our buffer and log any complete lines.
        """
        if not s:
            return 0

        # Handle line buffering for clean log output
        text = self._line_buffer + s
        lines = text.split("\n")

        # If the text ends with a newline, the last line is complete
        if text.endswith("\n"):
            complete_lines = lines
            self._line_buffer = ""
        else:
            # Otherwise, the last line is incomplete
            complete_lines = lines[:-1]
            self._line_buffer = lines[-1]

        for line in complete_lines:
            self._emit_line(line)

        # Always write to the underlying buffer
        return super().write(s)

    def flush(self) -> None:
        """Flush the internal buffer."""
        super().flush()

    def close(self) -> None:
        """Close the stream."""
        # Log any remaining content in the line buffer
        if self._line_buffer:
            self._emit_line(self._line_buffer)
        self._close_devnull_fd()
        super().close()

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def fileno(self) -> int:
        """
        Return a file descriptor for /dev/null.
        This prevents output from showing on the terminal
        while still allowing our write() method to capture it for logging.
        """
        if self._devnull_fd is None:
            self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        return self._devnull_fd

    def _close_devnull_fd(self) -> None:
        if self._devnull_fd is None:
            return
        fd = self._devnull_fd
        self._devnull_fd = None
        with suppress(OSError):
            os.close(fd)

    def __del__(self) -> None:
        """Clean up the devnull file descriptor."""
        with suppress(AttributeError):
            self._close_devnull_fd()


def get_stderr_handler(
    server_name: str,
    *,
    on_line: Callable[[str], None] | None = None,
) -> TextIO:
    """
    Get a stderr handler that routes MCP server errors to our logger.

    Args:
        server_name: The name of the server to include in logs

    Returns:
        A file-like object that can be used as stderr by MCP
    """
    return LoggerTextIO(server_name, on_line=on_line)
