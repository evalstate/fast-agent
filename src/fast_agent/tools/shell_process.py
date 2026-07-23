from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from mcp.types import CallToolResult, TextContent

from fast_agent.constants import FAST_AGENT_SHELL_PROCESS_METADATA
from fast_agent.tools.process_resources import (
    ProcessResourceObservationState,
    ProcessResourceSnapshotMetadata,
)
from fast_agent.ui import console
from fast_agent.utils.tool_names import POLL_PROCESS_TOOL_NAME, TERMINATE_PROCESS_TOOL_NAME

if TYPE_CHECKING:
    from fast_agent.tools.execution_environment import (
        ShellExecution,
        ShellExecutionRequest,
    )
    from fast_agent.tools.shell_output import ShellOutputBuffer
    from fast_agent.tools.shell_progress import ShellProgressReporter
    from fast_agent.tools.shell_runtime import ShellRuntime


class ProcessResultMetadata(TypedDict, total=False):
    """Durable metadata emitted by managed-process lifecycle tools."""

    process_id: str
    lifecycle: Literal["session", "persistent"]
    process_status: str
    process_yield_reason: str | None
    process_elapsed_seconds: float
    os_process_id: int | None
    exit_code: int
    output_line_count: int
    output_bytes_since_last_poll: int
    seconds_since_last_output: float
    has_observed_output: bool
    total_output_bytes: int
    poll_wait_sec: int
    poll_wake_on_output: bool
    poll_elapsed_seconds: float
    poll_deadline_overshoot_seconds: float
    resource_snapshot: ProcessResourceSnapshotMetadata
    resource_observation: str


def process_result_metadata(result: CallToolResult) -> ProcessResultMetadata | None:
    """Return the canonical managed-process metadata attached to a result."""
    metadata = (result.meta or {}).get(FAST_AGENT_SHELL_PROCESS_METADATA)
    if not isinstance(metadata, dict):
        return None
    return cast("ProcessResultMetadata", metadata)


def process_result(
    message: str,
    *,
    is_error: bool,
    metadata: ProcessResultMetadata,
) -> CallToolResult:
    result = CallToolResult(
        isError=is_error,
        content=[TextContent(type="text", text=message)],
    )
    result.meta = {FAST_AGENT_SHELL_PROCESS_METADATA: metadata}
    # Shell result rendering consumes this transient projection. Process lifecycle
    # consumers read the canonical durable metadata above.
    if "output_line_count" in metadata:
        cast("Any", result).output_line_count = metadata["output_line_count"]
    return result


@dataclass(slots=True)
class ShellDisplayState:
    use_live_shell_display: bool
    display_line_limit: int | None
    display_head_limit: int = 0
    display_tail_limit: int = 0
    displayed_head_count: int = 0
    display_total_line_count: int = 0
    display_overflowed: bool = False
    display_ellipsis_printed: bool = False
    timeout_notice_printed: bool = False
    display_tail_buffer: deque[tuple[int, str, str | None]] = field(
        default_factory=lambda: deque(maxlen=1)
    )


@dataclass(slots=True)
class ShellRuntimeCallbacks:
    runtime: ShellRuntime
    progress: ShellProgressReporter
    output_state: ShellOutputBuffer
    display_state: ShellDisplayState
    activity_event: asyncio.Event = field(default_factory=asyncio.Event)
    started_event: asyncio.Event = field(default_factory=asyncio.Event)
    os_process_id: int | None = None
    last_output_time: float = field(default_factory=time.monotonic)
    process: ManagedShellProcess | None = None

    async def on_started(self, process_id: int | None) -> None:
        self.os_process_id = process_id
        try:
            if self.process is not None:
                await self.runtime._capture_process_resource_baseline(self.process)
        finally:
            self.started_event.set()

    async def on_stdout(self, text: str) -> None:
        self.runtime._record_stream_output(
            text,
            style=None,
            output_state=self.output_state,
            display_state=self.display_state,
            is_stderr=False,
        )
        self.last_output_time = time.monotonic()
        self.activity_event.set()
        if self.process is not None:
            self.progress.emit_process_output(self.process)

    async def on_stderr(self, text: str) -> None:
        self.runtime._record_stream_output(
            text,
            style="red",
            output_state=self.output_state,
            display_state=self.display_state,
            is_stderr=True,
        )
        self.last_output_time = time.monotonic()
        self.activity_event.set()
        if self.process is not None:
            self.progress.emit_process_output(self.process)

    async def on_idle_warning(self, elapsed: float, remaining: float) -> None:
        if self.display_state.use_live_shell_display:
            console.console.print(
                f"▶ No output detected - terminating in {int(remaining)}s",
                style="black on red",
            )
        await self.runtime._emit_watchdog_progress(elapsed)

    async def on_timeout(self) -> None:
        self.runtime._print_timeout_notice(self.display_state)


@dataclass(slots=True)
class ActiveProcessPoll:
    tool_use_id: str
    deadline_at: float
    started_at: float
    last_progress_emitted_at: float = 0.0
    pending_progress_task: asyncio.Task[None] | None = None
    heartbeat_task: asyncio.Task[None] | None = None


@dataclass(slots=True)
class ManagedShellProcess:
    process_id: str
    command: str
    working_directory: str
    started_at: float
    task: asyncio.Task[ShellExecution]
    request: ShellExecutionRequest
    lifecycle: Literal["session", "persistent"]
    callbacks: ShellRuntimeCallbacks
    output_state: ShellOutputBuffer
    display_state: ShellDisplayState
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    poll_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    completed_at: float | None = None
    terminated: bool = False
    buffered_result_recorded: bool = False
    active_poll: ActiveProcessPoll | None = None
    resource_observations: ProcessResourceObservationState = field(
        default_factory=ProcessResourceObservationState
    )


@dataclass(frozen=True, slots=True)
class ManagedProcessSnapshot:
    """Immutable user-facing state for one retained managed process."""

    process_id: str
    command: str
    working_directory: str
    status: str
    elapsed_seconds: float
    os_process_id: int | None
    total_output_bytes: int
    exit_code: int | None
    output_spool_path: str | None = None


def build_managed_process_result(
    process: ManagedShellProcess,
    *,
    yielded_reason: str | None,
    minimal_process_profile: bool,
    io_drain_timeout_seconds: float,
) -> CallToolResult:
    unread_output_line_count = process.output_state.unread_output_line_count
    output = process.output_state.consume()
    sections: list[str] = []
    if output:
        sections.append(output.rstrip("\n"))

    elapsed = time.monotonic() - process.started_at
    if yielded_reason == "background":
        sections.append(f"effective_lifecycle: {process.lifecycle}")
    if not process.task.done():
        if yielded_reason == "background":
            reason = "started in the background"
        elif yielded_reason == "idle":
            reason = "reached the no-output yield threshold"
        elif yielded_reason == "foreground":
            reason = "reached the foreground yield threshold"
        else:
            reason = "is still running"
        status_message = (
            "Process is still running."
            if yielded_reason is None
            else f"Process is still running because it {reason}."
        )
        sections.extend(
            [
                status_message,
                f"process_id: {process.process_id}",
                f"elapsed_seconds: {elapsed:.1f}",
                f"total_output_bytes: {process.output_state.lifetime_output_bytes}",
                (
                    "Use Process with action='status' or 'wait' to monitor it, "
                    "or action='stop' to stop it."
                    if minimal_process_profile
                    else (
                        f"Use {POLL_PROCESS_TOOL_NAME} to monitor it or "
                        f"{TERMINATE_PROCESS_TOOL_NAME} to stop it."
                    )
                ),
            ]
        )
        if (
            minimal_process_profile
            and process.lifecycle == "session"
            and yielded_reason in {"idle", "foreground"}
        ):
            sections.append(
                "This process is session-scoped and will be stopped when the agent finishes. "
                "If it must remain running, stop it and relaunch with "
                "run_in_background=true."
            )
        if process.callbacks.os_process_id is not None and not minimal_process_profile:
            sections.insert(-3, f"os_pid: {process.callbacks.os_process_id}")
        result = process_result(
            "\n".join(sections),
            is_error=False,
            metadata={
                "process_id": process.process_id,
                "lifecycle": process.lifecycle,
                "process_status": "running",
                "process_yield_reason": yielded_reason,
                "process_elapsed_seconds": elapsed,
                "os_process_id": process.callbacks.os_process_id,
                "output_line_count": unread_output_line_count,
                "total_output_bytes": process.output_state.lifetime_output_bytes,
            },
        )
        cast("Any", result)._suppress_display = yielded_reason is not None or not output
        return result

    if process.task.cancelled():
        status = "terminated" if process.terminated else "cancelled"
        sections.extend(
            [
                f"process_id: {process.process_id}",
                f"process status: {status}",
            ]
        )
        return process_result(
            "\n".join(sections),
            is_error=False,
            metadata={
                "process_id": process.process_id,
                "lifecycle": process.lifecycle,
                "process_status": status,
                "process_elapsed_seconds": elapsed,
                "os_process_id": process.callbacks.os_process_id,
                "output_line_count": unread_output_line_count,
                "total_output_bytes": process.output_state.lifetime_output_bytes,
            },
        )

    exception = process.task.exception()
    if exception is not None:
        sections.extend(
            [
                f"process_id: {process.process_id}",
                f"Command execution failed: {exception}",
            ]
        )
        return process_result(
            "\n".join(sections),
            is_error=True,
            metadata={
                "process_id": process.process_id,
                "lifecycle": process.lifecycle,
                "process_status": "failed",
                "process_elapsed_seconds": elapsed,
                "os_process_id": process.callbacks.os_process_id,
                "output_line_count": unread_output_line_count,
                "total_output_bytes": process.output_state.lifetime_output_bytes,
            },
        )

    execution = process.task.result()
    if execution.io_drain_timed_out:
        sections.append(
            f"Output collection stopped after {io_drain_timeout_seconds:.1f}s because "
            "stdout/stderr pipes remained open."
        )
    sections.extend(
        [
            f"process_id: {process.process_id}",
            f"process exit code was {execution.result.exit_code}",
        ]
    )
    return process_result(
        "\n".join(sections),
        is_error=execution.result.exit_code != 0,
        metadata={
            "process_id": process.process_id,
            "lifecycle": process.lifecycle,
            "process_status": ("completed" if execution.result.exit_code == 0 else "failed"),
            "process_elapsed_seconds": elapsed,
            "os_process_id": process.callbacks.os_process_id,
            "exit_code": execution.result.exit_code,
            "output_line_count": unread_output_line_count,
            "total_output_bytes": process.output_state.lifetime_output_bytes,
        },
    )
