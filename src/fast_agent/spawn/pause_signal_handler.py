"""Pause Signal Handler — Unix signal-based pause/resume for subprocess agents.

State machine (mirrors ``services.pause_controller.PauseController`` in the
parent process)::

    running ──SIGUSR1──► pausing ──[checkpoint hook]──► paused
       ▲                                                  │
       │                                                  │
       └─[hook resumes from await]─ resuming ◄──SIGUSR2───┘

Four SSE events flow back out via ``emit_event`` → ``spawn_events.jsonl`` →
``SpawnProgressBridge`` → ``activity_stream_manager`` so the dashboard sees
the same lifecycle for spawned agents as for the built-in Jarvis agent.

Idle-state edge case: if the signal arrives while the agent has no
in-flight turn, the checkpoint hook never fires and ``agent_paused``
would never be emitted — the UI would stick on the "Pausing…"
spinner. We track activity via the module-level ``_active`` flag
(flipped True in ``pause_checkpoint``, flipped False in
``pause_turn_complete``); when ``_active`` is False, the signal
handler emits the terminal event itself.

Architecture:
  Main process sends os.kill(pid, SIGUSR1) → handler emits agent_pausing,
    clears _pause_event. For idle agents, handler also emits agent_paused.
  ``pause_checkpoint`` hook on before_llm_call/before_tool_call blocks
    at ``await event.wait()`` and brackets it with paused/resumed emits.
  Main process sends os.kill(pid, SIGUSR2) → handler emits agent_resuming,
    sets _pause_event. For idle agents, handler also emits agent_resumed.

Usage::

    # In isolated_runner.py main():
    install_pause_signal_handlers(asyncio.get_event_loop())

    # In hook merge chain:
    before_llm_call=pause_checkpoint     # blocks if paused
    after_turn_complete=pause_turn_complete  # marks idle
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
# True between ``before_llm_call`` and ``after_turn_complete``. Used by the
# signal handler to know whether to emit the terminal paused/resumed event
# itself (idle agent) or defer to the checkpoint hook (active agent).
_active = False


def _ensure_event() -> asyncio.Event:
    """Lazily create the pause event."""
    global _pause_event
    if _pause_event is None:
        _pause_event = asyncio.Event()
        _pause_event.set()  # Not paused by default
    return _pause_event


def _emit(event_type: str, agent_name: str, run_id: str, status: str) -> None:
    """Best-effort emit to spawn_events.jsonl. Never raise."""
    try:
        from fast_agent.spawn.spawn_events import emit_event
        emit_event(event_type, run_id, agent_name, status=status)
    except Exception:
        pass


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
        logger.info("[PAUSE-SIGNAL] SIGUSR1 received — agent %s pausing", agent_name)
        _emit("agent_pausing", agent_name, run_id, "pausing")
        # Idle case: no in-flight turn means the checkpoint hook will
        # never fire to emit ``agent_paused`` — handle it here.
        if not _active:
            _emit("agent_paused", agent_name, run_id, "paused")

    def _on_resume() -> None:
        event.set()
        logger.info("[PAUSE-SIGNAL] SIGUSR2 received — agent %s resuming", agent_name)
        _emit("agent_resuming", agent_name, run_id, "resuming")
        # Idle case: no hook to wake from ``await event.wait()``, so the
        # terminal ``agent_resumed`` event must come from here.
        if not _active:
            _emit("agent_resumed", agent_name, run_id, "running")

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
    """``before_llm_call`` / ``before_tool_call`` hook — blocks if paused.

    Returns immediately if not paused. Otherwise emits ``agent_paused``,
    awaits the resume event, then emits ``agent_resumed`` so the UI
    knows the agent has actually resumed (not just received the request).
    Also flips ``_active`` to True so the signal handler knows to defer
    terminal-state emits to this hook.
    """
    global _active
    _active = True

    event = _ensure_event()
    if not event.is_set():
        agent_name = os.environ.get("TEAM_MY_NAME", "unknown")
        run_id = os.environ.get("SPAWN_RUN_ID", "")
        logger.info("[PAUSE-SIGNAL] Agent %s blocked at checkpoint", agent_name)
        _emit("agent_paused", agent_name, run_id, "paused")
        await event.wait()
        _emit("agent_resumed", agent_name, run_id, "running")
        logger.info("[PAUSE-SIGNAL] Agent %s unblocked, continuing", agent_name)


async def pause_turn_complete(runner: Any, message: Any) -> None:
    """``after_turn_complete`` hook — marks the agent idle.

    Once a turn finishes there are no more in-flight LLM/tool calls,
    so the next pause request must be terminated by the signal handler
    itself (the checkpoint hook won't fire until a new turn starts).
    """
    global _active
    _active = False
