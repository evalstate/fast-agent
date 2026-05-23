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
# Ref to the asyncio.Task that's mid-LLM-call. Set by ``pause_checkpoint``
# (before_llm_call), cleared by ``pause_after_llm`` (after_llm_call) so that
# the signal handler only cancels during the LLM phase. Tool calls are not
# cancelled (strategy B — let side-effects complete; pause cooperatively
# at the next checkpoint).
_current_llm_task: asyncio.Task | None = None


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
        # Idempotent: a repeat SIGUSR1 while already paused must not
        # re-emit ``agent_pausing`` or re-cancel anything — the parent
        # ``PauseController._pause_one`` already returns False for
        # already-paused agents, but external senders (operator script,
        # k8s preStop hook, etc.) could still double-fire. Cheap guard.
        if not event.is_set():
            return
        event.clear()
        logger.info("[PAUSE-SIGNAL] SIGUSR1 received — agent %s pausing", agent_name)
        _emit("agent_pausing", agent_name, run_id, "pausing")
        # Instant-pause: cancel the in-flight LLM task. ``_current_llm_task``
        # is set ONLY by ``pause_before_llm`` (cleared in ``pause_after_llm``)
        # — strategy B leaves tool calls running to completion, so during
        # the tool phase this ref is None and no cancel fires. Cooperative
        # block at the next ``pause_before_tool`` checkpoint handles the
        # tool-phase pause without tearing down the in-flight tool.
        if _current_llm_task is not None and not _current_llm_task.done():
            _current_llm_task.cancel()
            logger.info("[PAUSE-SIGNAL] Cancelled in-flight LLM task for %s", agent_name)
        # Idle case: no in-flight turn means the checkpoint hook will
        # never fire to emit ``agent_paused`` — handle it here.
        if not _active:
            _emit("agent_paused", agent_name, run_id, "paused")

    def _on_resume() -> None:
        # Mirror idempotency: ignore SIGUSR2 if already running.
        if event.is_set():
            return
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


async def _block_if_paused() -> None:
    """Shared cooperative-block helper. If the pause event is cleared,
    emit ``agent_paused`` (terminal), await the resume signal, then
    emit ``agent_resumed`` so the UI flips out of the spinner state.
    No state-flag side effects — pure block.
    """
    event = _ensure_event()
    if not event.is_set():
        agent_name = os.environ.get("TEAM_MY_NAME", "unknown")
        run_id = os.environ.get("SPAWN_RUN_ID", "")
        logger.info("[PAUSE-SIGNAL] Agent %s blocked at checkpoint", agent_name)
        _emit("agent_paused", agent_name, run_id, "paused")
        await event.wait()
        _emit("agent_resumed", agent_name, run_id, "running")
        logger.info("[PAUSE-SIGNAL] Agent %s unblocked, continuing", agent_name)


async def pause_before_llm(runner: Any, messages: Any) -> None:
    """``before_llm_call`` hook.

    Registers the current asyncio Task as the cancel target — this is
    the strategy-B-correct entry point: ``_current_llm_task`` is set
    ONLY here, so the SIGUSR1 handler can cancel an in-flight LLM call
    but won't touch a task running an unrelated tool. Cleared in
    ``pause_after_llm`` to bound the cancel-eligible window strictly
    to the LLM phase.

    Also flips ``_active`` so the signal handler knows the agent has
    an in-flight turn and defers the terminal ``agent_paused`` event
    to ``_block_if_paused`` below.
    """
    global _active, _current_llm_task
    _active = True
    try:
        _current_llm_task = asyncio.current_task()
    except RuntimeError:
        _current_llm_task = None  # not in a task — cancel disabled
    await _block_if_paused()


async def pause_before_tool(runner: Any, request: Any) -> None:
    """``before_tool_call`` hook.

    Cooperative block only — does NOT register the current task. Tool
    calls have side effects (file writes, network calls, MCP calls)
    that cannot be cleanly unrolled, so strategy B leaves them
    running to completion. The pause request lands at the *next* LLM
    checkpoint (where ``pause_before_llm`` fires and cancels safely).

    Splitting this from ``pause_before_llm`` is load-bearing: if the
    same hook were wired to both ``before_llm_call`` AND
    ``before_tool_call``, ``_current_llm_task`` would be set during
    the tool phase too, and SIGUSR1 would cancel the in-flight tool
    task → CancelledError propagates out of ``run_tools`` → the chat
    request task dies (no on_pause_cancel retry contract for tools).
    Regression-tested in test_pause_strategy_b_tool_phase.py.
    """
    await _block_if_paused()


# Backward-compat alias — keep until external callers migrate.
# Defaults to LLM behavior because that's the historical use site;
# tool-phase callers must explicitly use ``pause_before_tool``.
pause_checkpoint = pause_before_llm


async def pause_after_llm(runner: Any, message: Any) -> None:
    """``after_llm_call`` hook — LLM call finished, clear the task ref so
    a subsequent SIGUSR1 doesn't cancel something past the LLM phase
    (e.g. the upcoming tool call, which we leave running per strategy B).
    """
    global _current_llm_task
    _current_llm_task = None


async def pause_turn_complete(runner: Any, message: Any) -> None:
    """``after_turn_complete`` hook — marks the agent idle.

    Once a turn finishes there are no more in-flight LLM/tool calls,
    so the next pause request must be terminated by the signal handler
    itself (the checkpoint hook won't fire until a new turn starts).
    """
    global _active, _current_llm_task
    _active = False
    _current_llm_task = None


async def pause_cancel_filter(runner: Any) -> bool:
    """``on_pause_cancel`` hook — the LLM call inside tool_runner just
    took CancelledError. If we initiated the cancel via SIGUSR1 (event
    is cleared), await resume and return True (the runner retries the
    LLM call with the same delta_messages). Otherwise return False
    (genuine cancel — propagate).
    """
    event = _ensure_event()
    if not event.is_set():
        agent_name = os.environ.get("TEAM_MY_NAME", "unknown")
        run_id = os.environ.get("SPAWN_RUN_ID", "")
        _emit("agent_paused", agent_name, run_id, "paused")
        logger.info("[PAUSE-SIGNAL] Agent %s blocked after LLM-cancel", agent_name)
        await event.wait()
        _emit("agent_resumed", agent_name, run_id, "running")
        logger.info("[PAUSE-SIGNAL] Agent %s unblocked, retrying LLM call", agent_name)
        return True
    return False
