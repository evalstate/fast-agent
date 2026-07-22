from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING, Any

from fast_agent.core.logging.progress_payloads import build_progress_payload
from fast_agent.event_progress import ProgressAction
from fast_agent.utils.text import summarize_command
from fast_agent.utils.tool_names import EXECUTE_TOOL_NAME, POLL_PROCESS_TOOL_NAME

if TYPE_CHECKING:
    from fast_agent.tools.shell_process import ActiveProcessPoll, ManagedShellProcess

_PROCESS_PROGRESS_EMIT_INTERVAL_SECONDS = 1.0


class ShellProgressReporter:
    def __init__(self, logger: Any, agent_name: str | None) -> None:
        self._logger = logger
        self._agent_name = agent_name

    @staticmethod
    def process_details(metadata: dict[str, Any]) -> str:
        return str(metadata.get("process_id") or "process")

    def emit_process_output(self, process: ManagedShellProcess) -> None:
        """Refresh an active poll row, coalescing chatty output bursts."""
        active_poll = process.active_poll
        if active_poll is None:
            return

        now = time.monotonic()
        remaining = _PROCESS_PROGRESS_EMIT_INTERVAL_SECONDS - (
            now - active_poll.last_progress_emitted_at
        )
        if remaining > 0:
            pending = active_poll.pending_progress_task
            if pending is None or pending.done():
                active_poll.pending_progress_task = asyncio.create_task(
                    self._emit_deferred_process_output(
                        process,
                        active_poll,
                        delay=remaining,
                    ),
                    name=f"fast-agent-{process.process_id}-progress",
                )
            return
        self._emit_process_output_now(process, active_poll)

    async def _emit_deferred_process_output(
        self,
        process: ManagedShellProcess,
        active_poll: ActiveProcessPoll,
        *,
        delay: float,
    ) -> None:
        await asyncio.sleep(delay)
        active_poll.pending_progress_task = None
        self._emit_process_output_now(process, active_poll)

    def _emit_process_output_now(
        self,
        process: ManagedShellProcess,
        active_poll: ActiveProcessPoll,
    ) -> None:
        self._emit_process_poll_now(
            process,
            active_poll,
            has_fresh_output=True,
            log_message="Process output progress",
        )

    def _emit_process_poll_now(
        self,
        process: ManagedShellProcess,
        active_poll: ActiveProcessPoll,
        *,
        has_fresh_output: bool,
        log_message: str,
    ) -> None:
        if process.active_poll is not active_poll:
            return

        now = time.monotonic()
        active_poll.last_progress_emitted_at = now
        seconds_since_last_output = max(now - process.callbacks.last_output_time, 0.0)
        self.emit(
            action=ProgressAction.CALLING_TOOL,
            tool_use_id=active_poll.tool_use_id,
            tool_name=POLL_PROCESS_TOOL_NAME,
            tool_event="progress",
            details=process.process_id,
            process_elapsed_seconds=max(now - process.started_at, 0.0),
            process_command=summarize_command(process.command),
            process_id=process.process_id,
            # Keep the original poll budget stable so the UI countdown track can
            # drain against task elapsed time instead of a shrinking remainder.
            process_wait_seconds=max(
                math.ceil(active_poll.deadline_at - active_poll.started_at),
                0,
            ),
            process_has_observed_output=(
                True if has_fresh_output else process.output_state.had_stream_output
            ),
            process_seconds_since_last_output=(
                0.0 if has_fresh_output else seconds_since_last_output
            ),
            process_total_output_bytes=process.output_state.lifetime_output_bytes,
            log_message=log_message,
        )

    async def poll_heartbeat(
        self,
        process: ManagedShellProcess,
        active_poll: ActiveProcessPoll,
    ) -> None:
        """Keep the monitoring row alive during quiet waits (e.g. sleep)."""
        try:
            while process.active_poll is active_poll and not process.task.done():
                await asyncio.sleep(_PROCESS_PROGRESS_EMIT_INTERVAL_SECONDS)
                if process.active_poll is not active_poll:
                    return
                self._emit_process_poll_now(
                    process,
                    active_poll,
                    has_fresh_output=False,
                    log_message="Process poll heartbeat",
                )
        except asyncio.CancelledError:
            return

    def emit(
        self,
        *,
        action: ProgressAction,
        tool_use_id: str | None,
        tool_name: str = EXECUTE_TOOL_NAME,
        tool_event: str | None = None,
        progress: float | None = None,
        total: float | None = None,
        details: str | None = None,
        tool_state: str | None = None,
        tool_terminal: bool | None = None,
        process_elapsed_seconds: float | None = None,
        process_command: str | None = None,
        process_id: str | None = None,
        process_wait_seconds: int | None = None,
        process_yield_reason: str | None = None,
        process_has_observed_output: bool | None = None,
        process_seconds_since_last_output: float | None = None,
        process_total_output_bytes: int | None = None,
        log_message: str = "Shell tool lifecycle",
    ) -> None:
        """Emit shell tool lifecycle events for progress display when supported."""
        info = getattr(self._logger, "info", None)
        if not callable(info):
            return

        payload: dict[str, Any] = build_progress_payload(
            action=action,
            tool_name=tool_name,
            server_name="local",
            agent_name=self._agent_name,
            tool_use_id=tool_use_id,
            tool_call_id=tool_use_id,
            tool_event=tool_event,
            tool_state=tool_state,
            tool_terminal=tool_terminal,
            progress=progress,
            total=total,
            details=details,
            extra={
                key: value
                for key, value in {
                    "process_elapsed_seconds": process_elapsed_seconds,
                    "process_command": process_command,
                    "process_id": process_id,
                    "process_wait_seconds": process_wait_seconds,
                    "process_yield_reason": process_yield_reason,
                    "process_has_observed_output": process_has_observed_output,
                    "process_seconds_since_last_output": process_seconds_since_last_output,
                    "process_total_output_bytes": process_total_output_bytes,
                }.items()
                if value is not None
            },
        )

        try:
            info(log_message, data=payload)
        except TypeError:
            # Standard library loggers reject custom keyword arguments.
            return
        except Exception:
            return
