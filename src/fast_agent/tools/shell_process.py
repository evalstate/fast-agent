from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from mcp_types import CallToolResult, TextContent

from fast_agent.constants import FAST_AGENT_SHELL_PROCESS_METADATA
from fast_agent.mcp.tool_result_metadata import update_tool_result_display_metadata
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


class ForegroundAutoAwaitMetadata(TypedDict):
    """Provenance for a runtime-owned wait after the initial foreground yield."""

    initial_yield_reason: Literal["idle", "foreground"]
    max_total_seconds: float
    initial_yield_elapsed_seconds: float
    awaited_seconds: float
    total_elapsed_seconds: float
    outcome: Literal[
        "process_finished",
        "cap_reached",
        "terminated",
        "cancelled",
        "disabled",
    ]


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
    retained_output_bytes_since_last_poll: int
    dropped_output_bytes_since_last_poll: int
    seconds_since_last_output: float
    seconds_since_last_stdout: float
    seconds_since_last_stderr: float
    has_observed_output: bool
    total_output_bytes: int
    stdout_bytes: int
    stderr_bytes: int
    output_spool_path: str
    retained_output_bytes: int
    retained_output_complete: bool
    dropped_output_bytes: int
    output_truncated: bool
    output_read_offset: int
    output_read_bytes: int
    output_read_has_more: bool
    output_query: str
    output_match_count: int
    poll_wait_sec: int
    poll_wake_on_output: bool
    poll_elapsed_seconds: float
    poll_deadline_overshoot_seconds: float
    resource_snapshot: ProcessResourceSnapshotMetadata
    resource_observation: str
    foreground_auto_await: ForegroundAutoAwaitMetadata


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
        is_error=is_error,
        content=[TextContent(type="text", text=message)],
    )
    result.meta = {FAST_AGENT_SHELL_PROCESS_METADATA: metadata}
    if "output_line_count" in metadata:
        update_tool_result_display_metadata(
            result,
            {"output_line_count": metadata["output_line_count"]},
        )
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
    last_stdout_time: float | None = None
    last_stderr_time: float | None = None
    raw_stdout_bytes_recorded: bool = False
    raw_stderr_bytes_recorded: bool = False
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
            count_bytes=not self.raw_stdout_bytes_recorded,
        )
        if self.raw_stdout_bytes_recorded:
            return
        now = time.monotonic()
        self.last_output_time = now
        self.last_stdout_time = now
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
            count_bytes=not self.raw_stderr_bytes_recorded,
        )
        if self.raw_stderr_bytes_recorded:
            return
        now = time.monotonic()
        self.last_output_time = now
        self.last_stderr_time = now
        self.activity_event.set()
        if self.process is not None:
            self.progress.emit_process_output(self.process)

    async def on_output_activity(self, *, is_stderr: bool, byte_count: int) -> None:
        self.output_state.had_stream_output = True
        self.output_state.unread_output_activity = True
        self.output_state.record_stream_bytes(byte_count, is_stderr=is_stderr)
        now = time.monotonic()
        self.last_output_time = now
        if is_stderr:
            self.raw_stderr_bytes_recorded = True
            self.last_stderr_time = now
        else:
            self.raw_stdout_bytes_recorded = True
            self.last_stdout_time = now
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
    intentional_persistent_background: bool
    callbacks: ShellRuntimeCallbacks
    output_state: ShellOutputBuffer
    display_state: ShellDisplayState
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    poll_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    completed_at: float | None = None
    terminated: bool = False
    buffered_result_recorded: bool = False
    active_poll: ActiveProcessPoll | None = None
    foreground_auto_await: ForegroundAutoAwaitMetadata | None = None
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
    lifecycle: Literal["session", "persistent"]
    output_spool_path: str | None = None


def _process_stream_metadata(process: ManagedShellProcess) -> ProcessResultMetadata:
    now = time.monotonic()
    metadata = ProcessResultMetadata(
        has_observed_output=process.output_state.had_stream_output,
        stdout_bytes=process.output_state.lifetime_stdout_bytes,
        stderr_bytes=process.output_state.lifetime_stderr_bytes,
    )
    if process.callbacks.last_stdout_time is not None:
        metadata["seconds_since_last_stdout"] = max(
            now - process.callbacks.last_stdout_time,
            0.0,
        )
    if process.callbacks.last_stderr_time is not None:
        metadata["seconds_since_last_stderr"] = max(
            now - process.callbacks.last_stderr_time,
            0.0,
        )
    return metadata


def build_managed_process_result(
    process: ManagedShellProcess,
    *,
    yielded_reason: str | None,
    minimal_process_profile: bool,
    aligned_shell_tool_name: str | None,
    io_drain_timeout_seconds: float,
    output_preview_limit: int | None = None,
) -> CallToolResult:
    unread_output_line_count = process.output_state.unread_output_line_count
    output = process.output_state.consume(output_preview_limit)
    sections: list[str] = []
    if output:
        sections.append(output.rstrip("\n"))

    elapsed = time.monotonic() - process.started_at
    if yielded_reason == "background":
        sections.append(f"effective_lifecycle: {process.lifecycle}")
    output_spool_path = process.request.output_spool_path
    if output_spool_path is not None:
        sections.append(f"output_spool_path: {output_spool_path}")
    if not process.task.done():
        persistent_background = process.intentional_persistent_background
        if persistent_background:
            status_message = "Managed background process is still running."
        elif yielded_reason == "idle":
            status_message = (
                "Command is still running; no completion result is available yet "
                "because it reached the no-output yield threshold."
            )
        elif yielded_reason == "foreground":
            status_message = (
                "Command is still running; no completion result is available yet "
                "because it reached the foreground yield threshold."
            )
        elif yielded_reason == "auto_await_cap":
            status_message = (
                "Command is still running after the bounded foreground auto-await "
                "total-runtime cap was reached. The command was not stopped."
            )
        else:
            status_message = "Command is still running; no completion result is available yet."
        if aligned_shell_tool_name is not None and persistent_background:
            next_action = (
                "This command was intentionally started with background=true. "
                "Do not wait for it to exit; use `process` with action='status' "
                "to inspect it or action='stop' to terminate it, and run readiness "
                f"checks in a separate `{aligned_shell_tool_name}` call."
            )
        elif minimal_process_profile and persistent_background:
            next_action = (
                "This command was intentionally started with "
                "run_in_background=true. Do not wait for it to exit; use `process` "
                "with action='status' to inspect it or action='stop' to terminate it, "
                "and run readiness checks in a separate `bash` call."
            )
        elif minimal_process_profile or aligned_shell_tool_name is not None:
            next_action = (
                "Next: call `process` with action='wait' or 'status'. Do not rely "
                "on partial output or end the task until the command completes."
            )
        else:
            next_action = (
                f"Use {POLL_PROCESS_TOOL_NAME} to monitor it or "
                f"{TERMINATE_PROCESS_TOOL_NAME} to stop it."
            )
        sections.extend(
            [
                status_message,
                f"process_id: {process.process_id}",
                f"elapsed_seconds: {elapsed:.1f}",
                f"total_output_bytes: {process.output_state.lifetime_output_bytes}",
                next_action,
            ]
        )
        if (
            (minimal_process_profile or aligned_shell_tool_name is not None)
            and process.lifecycle == "session"
            and yielded_reason in {"idle", "foreground", "auto_await_cap"}
        ):
            sections.append(
                "This process is session-scoped and will be stopped when the agent finishes. "
                "If it must remain running, stop it and relaunch with "
                f"{'background' if aligned_shell_tool_name is not None else 'run_in_background'}"
                "=true."
            )
        if (
            process.callbacks.os_process_id is not None
            and not minimal_process_profile
            and aligned_shell_tool_name is None
        ):
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
                **_process_stream_metadata(process),
                **(
                    {"output_spool_path": output_spool_path}
                    if output_spool_path is not None
                    else {}
                ),
            },
        )
        update_tool_result_display_metadata(
            result,
            {"suppress_display": yielded_reason is not None or not output},
        )
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
                **_process_stream_metadata(process),
                **(
                    {"output_spool_path": output_spool_path}
                    if output_spool_path is not None
                    else {}
                ),
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
                **_process_stream_metadata(process),
                **(
                    {"output_spool_path": output_spool_path}
                    if output_spool_path is not None
                    else {}
                ),
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
            **_process_stream_metadata(process),
            **({"output_spool_path": output_spool_path} if output_spool_path is not None else {}),
        },
    )
