"""Pause Signal Handler — Unix signal-based pause/resume for subprocess agents.

Installs SIGUSR1 (pause) and SIGUSR2 (resume) handlers on the asyncio event loop.
The `pause_checkpoint` coroutine is used as a ToolRunnerHooks.before_llm_call to
block execution when the agent is paused.

Architecture:
  Main process sends os.kill(pid, SIGUSR1) to pause.
  Signal handler clears _pause_event → hook blocks at next checkpoint.
  Main process sends os.kill(pid, SIGUSR2) to resume.
  Signal handler sets _pause_event → hook unblocks.

Usage::

    # In isolated_runner.py main():
    install_pause_signal_handlers(asyncio.get_event_loop())

    # In hook merge chain:
    before_llm_call=pause_checkpoint
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Any

logger = logging.getLogger(__name__)

# Module-level pause event — shared across all hooks in this subprocess.
# Default: set (not paused). Signal handlers toggle this atomically.
_pause_event: asyncio.Event | None = None
_installed = False


def _ensure_event() -> asyncio.Event:
    """Lazily create the pause event."""
    global _pause_event
    if _pause_event is None:
        _pause_event = asyncio.Event()
        _pause_event.set()  # Not paused by default
    return _pause_event


def install_pause_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Install SIGUSR1/SIGUSR2 signal handlers for pause/resume.

    Must be called from the main thread before the agent starts.
    Safe to call multiple times (idempotent).
    """
    global _installed
    if _installed:
        return

    event = _ensure_event()

    agent_name = os.environ.get("TEAM_MY_NAME", "unknown")
    run_id = os.environ.get("SPAWN_RUN_ID", "")

    def _on_pause() -> None:
        event.clear()
        logger.info("[PAUSE-SIGNAL] SIGUSR1 received — agent %s paused", agent_name)
        # Emit event to stderr → spawn_events.jsonl → bridge
        try:
            from fast_agent.spawn.spawn_events import emit_event
            emit_event("agent_paused", run_id, agent_name, status="paused")
        except Exception:
            pass

    def _on_resume() -> None:
        event.set()
        logger.info("[PAUSE-SIGNAL] SIGUSR2 received — agent %s resumed", agent_name)
        try:
            from fast_agent.spawn.spawn_events import emit_event
            emit_event("agent_resumed", run_id, agent_name, status="running")
        except Exception:
            pass

    try:
        loop.add_signal_handler(signal.SIGUSR1, _on_pause)
        loop.add_signal_handler(signal.SIGUSR2, _on_resume)
        _installed = True
        logger.info(
            "[PAUSE-SIGNAL] Handlers installed for %s (pid=%d)",
            agent_name, os.getpid(),
        )
    except (OSError, RuntimeError) as e:
        # Windows or non-main thread — signals not supported
        logger.warning("[PAUSE-SIGNAL] Cannot install signal handlers: %s", e)


async def pause_checkpoint(runner: Any, messages: Any) -> None:
    """ToolRunnerHooks.before_llm_call — blocks if agent is paused.

    Returns immediately if not paused.  Blocks until SIGUSR2 is received
    if the agent has been paused via SIGUSR1.
    """
    event = _ensure_event()
    if not event.is_set():
        agent_name = os.environ.get("TEAM_MY_NAME", "unknown")
        logger.info("[PAUSE-SIGNAL] Agent %s blocked at checkpoint", agent_name)
        await event.wait()
        logger.info("[PAUSE-SIGNAL] Agent %s unblocked, continuing", agent_name)
