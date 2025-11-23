"""
Keyboard interrupt handling for streaming cancellation.

This module provides functionality to detect ESC key presses during
LLM streaming to allow users to cancel ongoing requests.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Coroutine, TypeVar

    T = TypeVar("T")

# Check interval in seconds
CHECK_INTERVAL = 0.05


def _read_key_unix(stop_event: threading.Event) -> str | None:
    """
    Read a single key from stdin on Unix systems.
    Returns the key pressed or None if stop_event is set.
    """
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        # Set terminal to cbreak mode (less intrusive than raw)
        tty.setcbreak(fd)

        while not stop_event.is_set():
            # Check if input is available with a short timeout
            rlist, _, _ = select.select([sys.stdin], [], [], CHECK_INTERVAL)
            if rlist:
                char = sys.stdin.read(1)
                return char
        return None
    except Exception:
        return None
    finally:
        # Restore terminal settings
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


def _read_key_windows(stop_event: threading.Event) -> str | None:
    """
    Read a single key from stdin on Windows systems.
    Returns the key pressed or None if stop_event is set.
    """
    import msvcrt
    import time

    while not stop_event.is_set():
        if msvcrt.kbhit():
            char = msvcrt.getch()
            return char.decode("utf-8", errors="ignore")
        time.sleep(CHECK_INTERVAL)
    return None


def _keyboard_listener_thread(
    task_to_cancel: asyncio.Task,
    loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
) -> None:
    """
    Thread function that listens for ESC key and cancels the task.
    """
    # Select the appropriate key reader for the platform
    if sys.platform == "win32":
        read_func = _read_key_windows
    else:
        read_func = _read_key_unix

    try:
        while not stop_event.is_set() and not task_to_cancel.done():
            key = read_func(stop_event)
            if key == "\x1b":  # ESC
                # Cancel the task from the event loop's thread
                loop.call_soon_threadsafe(task_to_cancel.cancel)
                return
    except Exception:
        # If we can't read keyboard (e.g., no TTY), just exit
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
    loop = asyncio.get_running_loop()
    stop_event = threading.Event()

    # Start the keyboard listener thread
    listener_thread = threading.Thread(
        target=_keyboard_listener_thread,
        args=(task, loop, stop_event),
        daemon=True,
    )
    listener_thread.start()

    try:
        result = await task
        return result
    except asyncio.CancelledError:
        if on_cancel:
            on_cancel()
        raise
    finally:
        # Signal the listener thread to stop
        stop_event.set()
        # Give the thread a moment to clean up
        listener_thread.join(timeout=0.1)
