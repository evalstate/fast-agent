"""
Cancellation support for LLM calls.

This module provides infrastructure for cancelling ongoing LLM operations
via ESC key press (CLI) or session/cancel events (ACP).
"""

import asyncio
import sys
from typing import Callable, Optional

from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


class CancellationToken:
    """
    A token that can be used to signal cancellation of an operation.

    This is shared between the operation being performed and the
    cancellation monitor (ESC key listener or ACP cancel handler).
    """

    def __init__(self) -> None:
        self._cancelled = False
        self._event = asyncio.Event()
        self._callbacks: list[Callable[[], None]] = []

    def cancel(self) -> None:
        """Signal cancellation."""
        if not self._cancelled:
            self._cancelled = True
            self._event.set()
            # Invoke callbacks
            for callback in self._callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"Cancellation callback error: {e}")

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    async def wait(self) -> None:
        """Wait until cancellation is requested."""
        await self._event.wait()

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Register a callback to be invoked on cancellation."""
        self._callbacks.append(callback)
        # If already cancelled, invoke immediately
        if self._cancelled:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Cancellation callback error: {e}")


class CancellationError(Exception):
    """Raised when an operation is cancelled."""
    pass


async def monitor_esc_key(token: CancellationToken, check_interval: float = 0.1) -> None:
    """
    Monitor for ESC key press and signal cancellation.

    This runs in parallel with the main operation and checks for
    ESC key input periodically.

    Args:
        token: The cancellation token to signal
        check_interval: How often to check for input (seconds)
    """
    import select
    import termios
    import tty

    # Only works on Unix-like systems with a TTY
    if not sys.stdin.isatty():
        return

    # Save terminal settings
    try:
        old_settings = termios.tcgetattr(sys.stdin)
    except termios.error:
        return

    try:
        # Set terminal to raw mode (non-blocking input)
        tty.setcbreak(sys.stdin.fileno())

        while not token.is_cancelled:
            # Check if input is available (non-blocking)
            if select.select([sys.stdin], [], [], check_interval)[0]:
                ch = sys.stdin.read(1)
                if ch == '\x1b':  # ESC character
                    logger.info("ESC key pressed - cancelling operation")
                    token.cancel()
                    break
            else:
                # No input, just wait a bit
                await asyncio.sleep(check_interval)
    except Exception as e:
        logger.debug(f"ESC monitor error: {e}")
    finally:
        # Restore terminal settings
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except termios.error:
            pass


async def run_with_cancellation(
    operation: asyncio.coroutine,
    token: Optional[CancellationToken] = None,
    enable_esc_monitor: bool = True,
) -> tuple[any, bool]:
    """
    Run an async operation with cancellation support.

    Args:
        operation: The async operation to run
        token: Optional cancellation token (created if not provided)
        enable_esc_monitor: Whether to enable ESC key monitoring (CLI only)

    Returns:
        Tuple of (result, was_cancelled)
        - result is the operation result or None if cancelled
        - was_cancelled is True if the operation was cancelled
    """
    if token is None:
        token = CancellationToken()

    # Create the operation task
    operation_task = asyncio.create_task(operation)

    # Set up cancellation callback to cancel the task
    def cancel_task():
        if not operation_task.done():
            operation_task.cancel()

    token.on_cancel(cancel_task)

    tasks = [operation_task]

    # Add ESC monitor if enabled
    if enable_esc_monitor:
        esc_task = asyncio.create_task(monitor_esc_key(token))
        tasks.append(esc_task)

    try:
        # Wait for the first to complete
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Check if operation completed
        if operation_task in done:
            try:
                result = operation_task.result()
                return result, token.is_cancelled
            except asyncio.CancelledError:
                return None, True
        else:
            # ESC was pressed, operation still running
            return None, True

    except asyncio.CancelledError:
        return None, True


# Global registry for active cancellation tokens (for ACP integration)
_active_tokens: dict[str, CancellationToken] = {}
_tokens_lock = asyncio.Lock()


async def register_cancellation_token(session_id: str, token: CancellationToken) -> None:
    """Register a cancellation token for an ACP session."""
    async with _tokens_lock:
        _active_tokens[session_id] = token


async def unregister_cancellation_token(session_id: str) -> None:
    """Unregister a cancellation token for an ACP session."""
    async with _tokens_lock:
        _active_tokens.pop(session_id, None)


async def cancel_session(session_id: str) -> bool:
    """
    Cancel an active operation for a session.

    Returns True if a token was found and cancelled.
    """
    async with _tokens_lock:
        token = _active_tokens.get(session_id)
        if token:
            token.cancel()
            return True
        return False
