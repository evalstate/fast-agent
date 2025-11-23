"""
Keyboard interrupt handling for streaming cancellation.

This module provides functionality to detect ESC key presses during
LLM streaming to allow users to cancel ongoing requests.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Coroutine, TypeVar
    T = TypeVar('T')

# ESC key code
ESC_KEY = b'\x1b'

# Check interval in seconds
CHECK_INTERVAL = 0.05


async def _check_for_esc_unix() -> bool:
    """Check for ESC key press on Unix systems."""
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        # Set terminal to raw mode to read individual keystrokes
        tty.setraw(fd)

        # Check if input is available
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        if rlist:
            char = sys.stdin.read(1)
            if char == '\x1b':  # ESC
                return True
        return False
    finally:
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


async def _check_for_esc_windows() -> bool:
    """Check for ESC key press on Windows systems."""
    import msvcrt

    if msvcrt.kbhit():
        char = msvcrt.getch()
        if char == ESC_KEY:
            return True
    return False


async def listen_for_esc(task: asyncio.Task, check_interval: float = CHECK_INTERVAL) -> None:
    """
    Listen for ESC key press and cancel the given task when detected.

    Args:
        task: The asyncio task to cancel when ESC is pressed
        check_interval: How often to check for key presses (seconds)
    """
    # Select the appropriate checker for the platform
    if sys.platform == 'win32':
        check_func = _check_for_esc_windows
    else:
        check_func = _check_for_esc_unix

    try:
        while not task.done():
            try:
                if await check_func():
                    task.cancel()
                    return
            except Exception:
                # If we can't read keyboard (e.g., no TTY), just wait
                pass
            await asyncio.sleep(check_interval)
    except asyncio.CancelledError:
        # Listener was cancelled (normal when task completes)
        pass


async def run_with_esc_cancel(
    coro: "Coroutine[Any, Any, T]",
    on_cancel: callable = None,
) -> "T":
    """
    Run a coroutine with ESC key cancellation support.

    Args:
        coro: The coroutine to run
        on_cancel: Optional callback to run when cancelled

    Returns:
        The result of the coroutine

    Raises:
        asyncio.CancelledError: If ESC was pressed to cancel
    """
    task = asyncio.create_task(coro)
    listener = asyncio.create_task(listen_for_esc(task))

    try:
        result = await task
        return result
    except asyncio.CancelledError:
        if on_cancel:
            on_cancel()
        raise
    finally:
        listener.cancel()
        try:
            await listener
        except asyncio.CancelledError:
            pass
