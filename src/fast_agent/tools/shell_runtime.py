from __future__ import annotations

import asyncio
import math
import posixpath
import time
from collections import deque
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from mcp.types import CallToolResult, TextContent, Tool
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.config import Settings
    from fast_agent.tools.execution_environment import ShellEnvironment, ShellExecutionResult

# Import tool progress context for reporting shell execution progress
from fast_agent.agents.tool_agent import _tool_progress_context
from fast_agent.constants import (
    DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
    MAX_MANAGED_SHELL_PROCESSES,
    MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
    TERMINAL_BYTES_PER_TOKEN,
)
from fast_agent.core.logging.progress_payloads import build_progress_payload
from fast_agent.event_progress import ProgressAction
from fast_agent.tools.execution_environment import (
    ShellExecution,
    ShellExecutionRequest,
    ShellRuntimeInfo,
    execute_shell,
)
from fast_agent.tools.local_shell_executor import LocalShellExecutor
from fast_agent.tools.process_resources import (
    ProcessResourceSnapshot,
    observe_resource_changes,
    sample_process_resources,
)
from fast_agent.tools.shell_output import ShellOutputBuffer
from fast_agent.tools.shell_process import (
    ActiveProcessPoll,
    ManagedProcessSnapshot,
    ManagedShellProcess,
    ProcessResultMetadata,
    ShellDisplayState,
    ShellRuntimeCallbacks,
    build_managed_process_result,
    process_result,
    process_result_metadata,
)
from fast_agent.tools.shell_tool_definitions import (
    PROCESS_OUTPUT_DEBOUNCE_SECONDS,
    ShellExecuteArguments,
    build_execute_tool,
    build_minimal_bash_tool,
    build_minimal_process_tool,
    build_poll_process_tool,
    build_terminate_process_tool,
    parse_execute_arguments,
    parse_minimal_bash_arguments,
    parse_minimal_process_arguments,
    parse_poll_process_arguments,
    parse_terminate_process_arguments,
    set_poll_process_tool_default_wait_seconds,
)
from fast_agent.tools.tool_sources import SHELL_TOOL_SOURCE, set_tool_source
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.display_suppression import display_tools_enabled
from fast_agent.ui.progress_display import progress_display
from fast_agent.ui.shell_output_truncation import (
    SHELL_OUTPUT_TRUNCATION_MARKER,
    split_shell_output_line_limit,
)
from fast_agent.utils.path_display import format_relative_path
from fast_agent.utils.text import summarize_command
from fast_agent.utils.tool_names import (
    BASH_TOOL_NAME,
    EXECUTE_TOOL_NAME,
    POLL_PROCESS_TOOL_NAME,
    PROCESS_TOOL_NAME,
    TERMINATE_PROCESS_TOOL_NAME,
)

_IO_DRAIN_TIMEOUT_SECONDS = 2.0
_DEFAULT_IDLE_YIELD_SECONDS = 10
_DEFAULT_FOREGROUND_YIELD_SECONDS = 30
_PROCESS_OUTPUT_DEBOUNCE_SECONDS = PROCESS_OUTPUT_DEBOUNCE_SECONDS


def _default_max_process_poll_seconds() -> int:
    from fast_agent.config import ShellSettings

    return ShellSettings().process_poll_max_wait_seconds


def _default_minimal_process_profile() -> bool:
    from fast_agent.config import ShellSettings

    return ShellSettings().tool_profile == "minimal_process"


_PROCESS_PROGRESS_EMIT_INTERVAL_SECONDS = 1.0
_RESOURCE_OBSERVATION_TIMEOUT_SECONDS = 0.075


def _text_result(message: str, *, is_error: bool) -> CallToolResult:
    return CallToolResult(
        isError=is_error,
        content=[TextContent(type="text", text=message)],
    )


@dataclass(slots=True)
class _ShellRuntimeExecution:
    execution: ShellExecution
    output_state: ShellOutputBuffer
    display_state: ShellDisplayState


@dataclass(frozen=True, slots=True)
class _ManagedProcessOperation:
    kind: Literal["status", "wait", "stop"]
    process_id: str | None
    wait_sec: int | None


def _coerce_output_byte_limit(output_byte_limit: int | None) -> int:
    if type(output_byte_limit) is not int or output_byte_limit <= 0:
        return DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
    return min(output_byte_limit, MAX_TERMINAL_OUTPUT_BYTE_LIMIT)


class ShellRuntime:
    """Helper for managing the optional shell execute tool."""

    def __init__(
        self,
        activation_reason: str | None,
        logger,
        timeout_seconds: float = 90,
        warning_interval_seconds: int = 30,
        working_directory: Path | None = None,
        output_byte_limit: int | None = None,
        process_poll_default_wait_seconds: int = 0,
        config: Settings | None = None,
        agent_name: str | None = None,
        shell_environment: ShellEnvironment | None = None,
        idle_yield_seconds: float = _DEFAULT_IDLE_YIELD_SECONDS,
        foreground_yield_seconds: float = _DEFAULT_FOREGROUND_YIELD_SECONDS,
    ) -> None:
        self._working_directory = str(working_directory) if working_directory is not None else None
        self._environment = shell_environment or LocalShellExecutor(
            logger=logger,
            timeout_seconds=timeout_seconds,
            warning_interval_seconds=warning_interval_seconds,
            working_directory=working_directory,
            config=config,
        )
        self._activation_reason = activation_reason
        self._logger = logger
        self._timeout_seconds = timeout_seconds
        self._warning_interval_seconds = warning_interval_seconds
        self._output_byte_limit = DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
        self.set_output_byte_limit(output_byte_limit)
        self.enabled: bool = activation_reason is not None
        self._tool: Tool | None = None
        self._display = ConsoleDisplay(config=config)
        self._config = config
        self._agent_name = agent_name
        self._idle_yield_seconds = idle_yield_seconds
        self._foreground_yield_seconds = foreground_yield_seconds
        self._managed_processes: dict[str, ManagedShellProcess] = {}
        self._next_process_id = 1
        self._processes_lock = asyncio.Lock()
        self._output_display_lines: int | None = None
        self._show_bash_output = True
        self._prefer_local_shell = False
        self._max_process_poll_seconds = _default_max_process_poll_seconds()
        self._minimal_process_profile = _default_minimal_process_profile()
        if config is not None:
            shell_config = config.shell_execution
            self._output_display_lines = shell_config.output_display_lines
            self._show_bash_output = shell_config.show_bash
            self._prefer_local_shell = shell_config.prefer_local_shell
            self._max_process_poll_seconds = shell_config.process_poll_max_wait_seconds
            self._minimal_process_profile = shell_config.tool_profile == "minimal_process"
        self._process_poll_default_wait_seconds = min(
            process_poll_default_wait_seconds,
            self._max_process_poll_seconds,
        )
        self._resource_observations_enabled = self.runtime_info().kind == "local"

        if self.enabled:
            shell_name = self.runtime_info().name
            if self._minimal_process_profile:
                self._tool = set_tool_source(
                    build_minimal_bash_tool(shell_name=shell_name),
                    SHELL_TOOL_SOURCE,
                )
                self._poll_process_tool = set_tool_source(
                    build_minimal_process_tool(),
                    SHELL_TOOL_SOURCE,
                )
                self._terminate_process_tool = None
            else:
                self._tool = set_tool_source(
                    build_execute_tool(shell_name=shell_name),
                    SHELL_TOOL_SOURCE,
                )
                self._poll_process_tool = set_tool_source(
                    build_poll_process_tool(
                        default_wait_seconds=self._process_poll_default_wait_seconds,
                        max_wait_seconds=self._max_process_poll_seconds,
                    ),
                    SHELL_TOOL_SOURCE,
                )
                self._terminate_process_tool = set_tool_source(
                    build_terminate_process_tool(),
                    SHELL_TOOL_SOURCE,
                )
        else:
            self._poll_process_tool = None
            self._terminate_process_tool = None

    @property
    def tool(self) -> Tool | None:
        return self._tool

    @property
    def tools(self) -> list[Tool]:
        """Return all model-facing shell and process lifecycle tools."""
        return [
            tool
            for tool in (self._tool, self._poll_process_tool, self._terminate_process_tool)
            if tool is not None
        ]

    def owns_tool(self, name: str) -> bool:
        """Return whether this runtime owns a model-facing tool name."""
        return any(tool.name == name for tool in self.tools)

    @property
    def active_process_count(self) -> int:
        """Return the number of managed processes that are currently alive."""
        return sum(not process.task.done() for process in self._managed_processes.values())

    async def process_snapshots(self) -> tuple[ManagedProcessSnapshot, ...]:
        """Return retained process state for interactive status displays."""
        async with self._processes_lock:
            processes = tuple(self._managed_processes.values())
        now = time.monotonic()
        return tuple(self._process_snapshot(process, now=now) for process in processes)

    @staticmethod
    def _process_snapshot(
        process: ManagedShellProcess,
        *,
        now: float,
    ) -> ManagedProcessSnapshot:
        exit_code: int | None = None
        if not process.task.done():
            status = "running"
        elif process.task.cancelled():
            status = "terminated" if process.terminated else "cancelled"
        elif process.task.exception() is not None:
            status = "failed"
        else:
            exit_code = process.task.result().result.exit_code
            status = "completed" if exit_code == 0 else "failed"
        if process.task.done() and process.completed_at is None:
            process.completed_at = now
        elapsed_at = process.completed_at if process.completed_at is not None else now
        return ManagedProcessSnapshot(
            process_id=process.process_id,
            command=process.command,
            working_directory=process.working_directory,
            status=status,
            elapsed_seconds=max(elapsed_at - process.started_at, 0),
            os_process_id=process.callbacks.os_process_id,
            total_output_bytes=process.output_state.lifetime_output_bytes,
            exit_code=exit_code,
        )

    @property
    def prefer_local_shell(self) -> bool:
        """Whether ACP mode should keep this local shell runtime instead of client terminal."""
        return self._prefer_local_shell

    @property
    def output_byte_limit(self) -> int:
        """Return the current byte limit used to retain command output."""
        return self._output_byte_limit

    @property
    def timeout_seconds(self) -> float:
        """Return the idle/no-output timeout used for shell execution."""
        return self._timeout_seconds

    def set_output_byte_limit(self, output_byte_limit: int | None) -> None:
        """Set output retention byte limit, honoring global defaults and hard cap."""
        self._output_byte_limit = _coerce_output_byte_limit(output_byte_limit)

    def announce(self) -> None:
        """Inform the user why the local shell tool is active."""
        if not self.enabled or not self._activation_reason:
            return

        message = f"Shell execute tool enabled {self._activation_reason}."
        self._logger.info(message)

    def _render_display_line(self, text: str, style: str | None) -> Text:
        display_text = text.rstrip("\n").expandtabs()
        renderable = Text(display_text, style=style or "")
        renderable.no_wrap = True
        width = max(1, console.console.size.width)
        if len(display_text) > width:
            renderable.truncate(width, overflow="ellipsis")
        return renderable

    def working_directory(self) -> Path:
        """Return the working directory used for shell execution."""
        from pathlib import Path

        return Path(self._working_directory or self._environment.cwd)

    def set_working_directory(self, working_directory: Path | None) -> None:
        """Set the working directory used for shell execution."""
        self._working_directory = str(working_directory) if working_directory is not None else None

    def runtime_info(self) -> ShellRuntimeInfo:
        """Best-effort detection of the shell runtime used for execution.

        Prefers modern shells like pwsh (PowerShell 7+) and bash.
        """
        info = self._environment.runtime_info()
        if info.environment_name is not None:
            return info

        from fast_agent.tools.environment_registry import environment_name

        name = environment_name(self._environment)
        if name is None:
            return info
        return ShellRuntimeInfo(
            name=info.name,
            path=info.path,
            kind=info.kind,
            provider=info.provider,
            environment_name=name,
        )

    def metadata(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Build metadata for display when the shell tool is invoked."""
        info = self.runtime_info()
        try:
            parsed = (
                parse_minimal_bash_arguments(arguments)
                if self._minimal_process_profile
                else parse_execute_arguments(arguments)
            )
        except ValueError:
            parsed = None
        command = parsed.command if parsed is not None else arguments.get("command")
        working_dir = Path(
            self._resolve_managed_working_directory(
                parsed.cwd if parsed is not None else None
            )
        )
        idle_yield_seconds = (
            parsed.yield_after_idle_sec
            if parsed is not None and parsed.yield_after_idle_sec is not None
            else self._idle_yield_seconds
        )
        output_byte_limit = (
            parsed.output_byte_limit
            if parsed is not None and parsed.output_byte_limit is not None
            else self._output_byte_limit
        )

        return {
            "variant": "shell",
            "command": command,
            "shell_name": info.name,
            "shell_path": info.path,
            "shell_kind": info.kind,
            "shell_provider": info.provider,
            "working_dir": str(working_dir),
            "working_dir_display": format_relative_path(working_dir),
            "idle_yield_seconds": idle_yield_seconds,
            "foreground_yield_seconds": self._foreground_yield_seconds,
            "background": parsed.background if parsed is not None else False,
            "lifecycle": parsed.lifecycle if parsed is not None else "session",
            "output_byte_limit": output_byte_limit,
            "streams_output": True,
            "returns_exit_code": True,
        }

    def process_tool_metadata(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Build compact display metadata for process lifecycle tools."""
        operation = self._managed_process_operation(tool_name, arguments)
        metadata: dict[str, Any] = {
            "variant": "shell_process",
            "action": "terminate" if operation.kind == "stop" else "poll",
            "process_id": operation.process_id,
            "wait_sec": operation.wait_sec,
        }
        process_id = operation.process_id
        process = (
            self._managed_processes.get(process_id)
            if isinstance(process_id, str)
            else None
        )
        if process is None:
            return metadata

        snapshot = self._process_snapshot(process, now=time.monotonic())
        metadata.update(
            {
                "command": snapshot.command,
                "command_summary": summarize_command(snapshot.command),
                "elapsed_seconds": snapshot.elapsed_seconds,
                "os_process_id": snapshot.os_process_id,
                "total_output_bytes": snapshot.total_output_bytes,
                "process_status": snapshot.status,
                "seconds_since_last_output": max(
                    time.monotonic() - process.callbacks.last_output_time,
                    0.0,
                ),
                "has_observed_output": process.output_state.had_stream_output,
            }
        )
        return metadata

    def _managed_process_operation(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> _ManagedProcessOperation:
        if tool_name == PROCESS_TOOL_NAME and self._minimal_process_profile:
            try:
                parsed = parse_minimal_process_arguments(arguments)
            except ValueError:
                return _ManagedProcessOperation(
                    kind="status",
                    process_id=None,
                    wait_sec=0,
                )
            return _ManagedProcessOperation(
                kind=parsed.action,
                process_id=parsed.process_id,
                wait_sec=(
                    None
                    if parsed.action == "stop"
                    else 0
                    if parsed.action == "status"
                    else self._process_poll_default_wait_seconds
                ),
            )

        process_id = arguments.get("process_id")
        return _ManagedProcessOperation(
            kind="stop" if tool_name == TERMINATE_PROCESS_TOOL_NAME else "wait",
            process_id=process_id if isinstance(process_id, str) else None,
            wait_sec=(
                None
                if tool_name == TERMINATE_PROCESS_TOOL_NAME
                else arguments.get("wait_sec", self._process_poll_default_wait_seconds)
            ),
        )

    def _process_progress_details(
        self,
        metadata: dict[str, Any],
    ) -> str:
        return str(metadata.get("process_id") or "process")

    def _emit_managed_process_output_progress(
        self,
        process: ManagedShellProcess,
    ) -> None:
        """Refresh an active poll row, coalescing chatty output bursts."""
        active_poll = process.active_poll
        if active_poll is None:
            return

        now = time.monotonic()
        remaining = (
            _PROCESS_PROGRESS_EMIT_INTERVAL_SECONDS
            - (now - active_poll.last_progress_emitted_at)
        )
        if remaining > 0:
            pending = active_poll.pending_progress_task
            if pending is None or pending.done():
                active_poll.pending_progress_task = asyncio.create_task(
                    self._emit_deferred_managed_process_output_progress(
                        process,
                        active_poll,
                        delay=remaining,
                    ),
                    name=f"fast-agent-{process.process_id}-progress",
                )
            return
        self._emit_managed_process_output_progress_now(process, active_poll)

    async def _emit_deferred_managed_process_output_progress(
        self,
        process: ManagedShellProcess,
        active_poll: ActiveProcessPoll,
        *,
        delay: float,
    ) -> None:
        await asyncio.sleep(delay)
        active_poll.pending_progress_task = None
        self._emit_managed_process_output_progress_now(process, active_poll)

    def _emit_managed_process_output_progress_now(
        self,
        process: ManagedShellProcess,
        active_poll: ActiveProcessPoll,
    ) -> None:
        self._emit_managed_process_poll_progress_now(
            process,
            active_poll,
            has_fresh_output=True,
            log_message="Process output progress",
        )

    def _emit_managed_process_poll_progress_now(
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
        self._emit_progress_event(
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
                True
                if has_fresh_output
                else process.output_state.had_stream_output
            ),
            process_seconds_since_last_output=(
                0.0 if has_fresh_output else seconds_since_last_output
            ),
            process_total_output_bytes=process.output_state.lifetime_output_bytes,
            log_message=log_message,
        )

    async def _poll_progress_heartbeat(
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
                self._emit_managed_process_poll_progress_now(
                    process,
                    active_poll,
                    has_fresh_output=False,
                    log_message="Process poll heartbeat",
                )
        except asyncio.CancelledError:
            return

    def _invalid_execute_result(self, message: str) -> CallToolResult:
        return _text_result(message, is_error=True)

    def set_process_poll_default_wait_seconds(self, value: int) -> None:
        """Update the model-specific default used when wait_sec is omitted."""
        default_wait = value if type(value) is int and value >= 0 else 0
        self._process_poll_default_wait_seconds = min(
            default_wait,
            self._max_process_poll_seconds,
        )
        if self._poll_process_tool is not None:
            set_poll_process_tool_default_wait_seconds(
                self._poll_process_tool,
                default_wait_seconds=self._process_poll_default_wait_seconds,
            )

    def _build_display_state(
        self,
        *,
        defer_display_to_tool_result: bool,
        display_line_limit: int | None = None,
    ) -> ShellDisplayState:
        use_live_shell_display = (
            self._show_bash_output and not defer_display_to_tool_result and display_tools_enabled()
        )
        state = ShellDisplayState(
            use_live_shell_display=use_live_shell_display,
            display_line_limit=display_line_limit,
        )
        if display_line_limit is not None and display_line_limit > 0:
            display_window = split_shell_output_line_limit(display_line_limit)
            state.display_head_limit = display_window.head_lines
            state.display_tail_limit = display_window.tail_lines
            state.display_tail_buffer = deque(maxlen=max(display_window.tail_lines, 1))
        return state

    def _maybe_print_truncation_notice(
        self,
        *,
        output_state: ShellOutputBuffer,
        display_state: ShellDisplayState,
    ) -> None:
        if output_state.truncation_notice_printed or not output_state.output_truncated:
            return
        if display_state.use_live_shell_display and (
            display_state.display_line_limit is None or display_state.display_line_limit > 0
        ):
            estimated_tokens = int(output_state.output_byte_limit / TERMINAL_BYTES_PER_TOKEN)
            console.console.print(
                " ".join(
                    [
                        "▶ Shell to agent output reached",
                        f"{output_state.output_byte_limit} bytes",
                        f"(~{estimated_tokens} tokens);",
                        "additional output omitted from tool result.",
                    ]
                ),
                style=self._truncation_notice_style(output_state),
            )
        output_state.truncation_notice_printed = True

    @staticmethod
    def _truncation_notice_style(output_state: ShellOutputBuffer) -> str:
        return (
            "black on blue"
            if output_state.output_byte_limit_requested
            else "black on red"
        )

    def _print_timeout_notice(
        self,
        display_state: ShellDisplayState,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        if not display_state.use_live_shell_display or display_state.timeout_notice_printed:
            return
        message = "▶ Timeout exceeded - terminating process"
        if timeout_seconds is not None:
            message = f"▶ Timeout after {timeout_seconds:g}s - process terminated"
        console.console.print(message, style="black on red")
        display_state.timeout_notice_printed = True

    def _render_live_shell_output(
        self,
        text: str,
        style: str | None,
        *,
        display_state: ShellDisplayState,
    ) -> None:
        if not display_state.use_live_shell_display:
            return
        if display_state.display_line_limit is None:
            console.console.print(
                self._render_display_line(text, style),
                markup=False,
            )
            return
        if display_state.display_line_limit <= 0:
            return

        display_state.display_total_line_count += 1
        current_line_index = display_state.display_total_line_count
        if display_state.displayed_head_count < display_state.display_head_limit:
            console.console.print(
                self._render_display_line(text, style),
                markup=False,
            )
            display_state.displayed_head_count += 1
            return

        if display_state.display_tail_limit > 0:
            display_state.display_tail_buffer.append((current_line_index, text, style))
        if current_line_index > display_state.display_line_limit:
            display_state.display_overflowed = True
            if not display_state.display_ellipsis_printed:
                console.console.print(
                    SHELL_OUTPUT_TRUNCATION_MARKER,
                    style="dim",
                    markup=False,
                )
                display_state.display_ellipsis_printed = True

    def _record_stream_output(
        self,
        text: str,
        *,
        style: str | None,
        output_state: ShellOutputBuffer,
        display_state: ShellDisplayState,
        is_stderr: bool,
    ) -> None:
        output_state.had_stream_output = True
        output_state.output_line_count += 1
        output_state.unread_output_line_count += 1
        output_text = text if not is_stderr else f"[stderr] {text}"
        output_state.append(output_text)
        self._maybe_print_truncation_notice(
            output_state=output_state,
            display_state=display_state,
        )
        self._render_live_shell_output(
            text,
            style,
            display_state=display_state,
        )

    async def _emit_watchdog_progress(self, elapsed: float) -> None:
        ctx = _tool_progress_context.get()
        if not ctx:
            return
        handler, tool_call_id = ctx
        try:
            await handler.on_tool_progress(
                tool_call_id,
                0.5,
                None,
                f"Waiting for output ({int(elapsed)}) seconds ...",
            )
        except Exception:
            return

    def _flush_live_display_tail(self, display_state: ShellDisplayState) -> None:
        if (
            not display_state.use_live_shell_display
            or display_state.display_line_limit is None
            or display_state.display_line_limit <= 0
        ):
            return
        if display_state.display_overflowed and not display_state.display_ellipsis_printed:
            console.console.print(
                SHELL_OUTPUT_TRUNCATION_MARKER,
                style="dim",
                markup=False,
            )
        for buffered_index, buffered_text, buffered_style in display_state.display_tail_buffer:
            if display_state.display_overflowed and buffered_index <= display_state.display_line_limit:
                continue
            console.console.print(
                self._render_display_line(buffered_text, buffered_style),
                markup=False,
            )

    def _finalize_shell_result_display(
        self,
        result: CallToolResult,
        *,
        shell_result: ShellExecutionResult,
        output_state: ShellOutputBuffer,
        display_state: ShellDisplayState,
        tool_use_id: str | None,
        show_tool_call_id: bool,
        defer_display_to_tool_result: bool,
    ) -> CallToolResult:
        self._flush_live_display_tail(display_state)
        if display_state.use_live_shell_display:
            self._display.show_shell_exit_code(
                shell_result.exit_code,
                no_output=not output_state.had_stream_output,
                output_line_count=output_state.output_line_count
                if output_state.had_stream_output
                else None,
                tool_call_id=tool_use_id if show_tool_call_id else None,
            )

        suppress_display = True
        if defer_display_to_tool_result and self._show_bash_output:
            suppress_display = False
        result_meta = cast("Any", result)
        result_meta._suppress_display = suppress_display
        result_meta.exit_code = shell_result.exit_code
        result_meta.output_line_count = output_state.output_line_count
        return result

    async def _execute_shell_command(
        self,
        command: str,
        *,
        cwd: str | Path | None,
        env: Mapping[str, str] | None,
        timeout: float | None,
        output_byte_limit: int | None,
        defer_display_to_tool_result: bool,
        display_line_limit: int | None,
    ) -> _ShellRuntimeExecution:
        output_state = ShellOutputBuffer(
            output_byte_limit=(
                self._output_byte_limit
                if output_byte_limit is None
                else output_byte_limit
            ),
            output_byte_limit_requested=output_byte_limit is not None,
        )
        display_state = self._build_display_state(
            defer_display_to_tool_result=defer_display_to_tool_result,
            display_line_limit=display_line_limit,
        )
        execution = await self._environment.execute(
            ShellExecutionRequest(
                command=command,
                cwd=str(cwd) if cwd is not None else self._working_directory,
                env=env,
                timeout=self._timeout_seconds if timeout is None else timeout,
            ),
            callbacks=ShellRuntimeCallbacks(
                runtime=self,
                output_state=output_state,
                display_state=display_state,
            ),
        )
        return _ShellRuntimeExecution(
            execution=execution,
            output_state=output_state,
            display_state=display_state,
        )

    async def _start_managed_process(
        self,
        parsed: ShellExecuteArguments,
        *,
        defer_display_to_tool_result: bool,
    ) -> ManagedShellProcess:
        output_state = ShellOutputBuffer(
            output_byte_limit=(
                self._output_byte_limit
                if parsed.output_byte_limit is None
                else parsed.output_byte_limit
            ),
            output_byte_limit_requested=parsed.output_byte_limit is not None,
        )
        display_state = self._build_display_state(
            defer_display_to_tool_result=defer_display_to_tool_result,
            display_line_limit=self._output_display_lines,
        )
        callbacks = ShellRuntimeCallbacks(
            runtime=self,
            output_state=output_state,
            display_state=display_state,
        )
        working_directory = self._resolve_managed_working_directory(parsed.cwd)

        async with self._processes_lock:
            completed_ids = [
                process_id
                for process_id, process in self._managed_processes.items()
                if process.task.done()
            ]
            while (
                len(self._managed_processes) >= MAX_MANAGED_SHELL_PROCESSES
                and completed_ids
            ):
                self._managed_processes.pop(completed_ids.pop(0))
            if len(self._managed_processes) >= MAX_MANAGED_SHELL_PROCESSES:
                raise RuntimeError(
                    f"at most {MAX_MANAGED_SHELL_PROCESSES} managed shell processes may run at once"
                )

            process_id = f"process-{self._next_process_id}"
            self._next_process_id += 1
            request = ShellExecutionRequest(
                command=parsed.command,
                cwd=working_directory,
                env=None,
                timeout=None,
                terminate_after_idle=False,
                retain_output=False,
                terminate_on_cancel=parsed.lifecycle == "session",
            )
            task = asyncio.create_task(
                self._environment.execute(request, callbacks=callbacks),
                name=f"fast-agent-{process_id}",
            )
            process = ManagedShellProcess(
                process_id=process_id,
                command=parsed.command,
                working_directory=working_directory,
                started_at=time.monotonic(),
                task=task,
                request=request,
                lifecycle=parsed.lifecycle,
                callbacks=callbacks,
                output_state=output_state,
                display_state=display_state,
            )
            callbacks.process = process
            task.add_done_callback(
                lambda completed_task: self._record_managed_process_completion(
                    process,
                    completed_task,
                )
            )
            self._managed_processes[process_id] = process
            return process

    @staticmethod
    def _record_managed_process_completion(
        process: ManagedShellProcess,
        completed_task: asyncio.Task[ShellExecution],
    ) -> None:
        del completed_task
        if process.completed_at is None:
            process.completed_at = time.monotonic()

    def _resolve_managed_working_directory(self, requested_cwd: str | None) -> str:
        base_cwd = self._working_directory or self._environment.cwd
        runtime_info = self.runtime_info()
        if runtime_info.kind == "local":
            base_path = Path(base_cwd)
            if not base_path.is_absolute():
                base_path = Path(self._environment.cwd) / base_path
            if requested_cwd is None:
                candidate = str(base_path)
            else:
                requested_path = Path(requested_cwd)
                candidate = str(
                    requested_path
                    if requested_path.is_absolute()
                    else base_path / requested_path
                )
        else:
            resolved_base = (
                base_cwd
                if posixpath.isabs(base_cwd)
                else posixpath.join(self._environment.cwd, base_cwd)
            )
            if requested_cwd is None:
                candidate = resolved_base
            else:
                candidate = (
                    requested_cwd
                    if posixpath.isabs(requested_cwd)
                    else posixpath.join(resolved_base, requested_cwd)
                )

        resolver = getattr(self._environment, "resolve_path", None)
        if callable(resolver):
            return str(resolver(candidate))
        if runtime_info.kind == "local":
            return str(Path(candidate).resolve())
        return posixpath.normpath(candidate)

    async def _get_managed_process(self, process_id: str) -> ManagedShellProcess | None:
        async with self._processes_lock:
            return self._managed_processes.get(process_id)

    async def _sample_managed_process_resources(
        self,
        process: ManagedShellProcess,
    ) -> ProcessResourceSnapshot | None:
        if not self._resource_observations_enabled:
            return None
        pid = process.callbacks.os_process_id if not process.task.done() else None
        try:
            async with asyncio.timeout(_RESOURCE_OBSERVATION_TIMEOUT_SECONDS):
                return await sample_process_resources(process.working_directory, pid)
        except Exception:
            return None

    async def _capture_process_resource_baseline(
        self,
        process: ManagedShellProcess,
    ) -> None:
        snapshot = await self._sample_managed_process_resources(process)
        if snapshot is not None:
            observe_resource_changes(process.resource_observations, snapshot)

    async def _wait_for_initial_process_result(
        self,
        process: ManagedShellProcess,
        *,
        idle_yield_seconds: float,
    ) -> str | None:
        if process.task.done():
            return None
        foreground_deadline = process.started_at + self._foreground_yield_seconds

        while not process.task.done():
            now = time.monotonic()
            idle_deadline = process.callbacks.last_output_time + idle_yield_seconds
            deadline = min(idle_deadline, foreground_deadline)
            if now >= deadline:
                return "idle" if idle_deadline <= foreground_deadline else "foreground"

            process.callbacks.activity_event.clear()
            if process.task.done():
                return None
            activity_task = asyncio.create_task(process.callbacks.activity_event.wait())
            try:
                done, _ = await asyncio.wait(
                    (process.task, activity_task),
                    timeout=max(deadline - time.monotonic(), 0),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not activity_task.done():
                    activity_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await activity_task
            if process.task in done:
                return None

        return None

    def _record_buffered_process_result(self, process: ManagedShellProcess) -> None:
        if process.buffered_result_recorded or not process.task.done():
            return
        process.buffered_result_recorded = True
        if process.task.cancelled() or process.task.exception() is not None:
            return
        execution = process.task.result()
        if process.output_state.had_stream_output:
            return
        recorded_output = False
        if execution.result.stdout:
            self._record_stream_output(
                execution.result.stdout,
                style=None,
                output_state=process.output_state,
                display_state=process.display_state,
                is_stderr=False,
            )
            recorded_output = True
        if execution.result.stderr:
            self._record_stream_output(
                execution.result.stderr,
                style="red",
                output_state=process.output_state,
                display_state=process.display_state,
                is_stderr=True,
            )
            recorded_output = True
        if recorded_output:
            process.callbacks.last_output_time = time.monotonic()

    @staticmethod
    def _append_poll_output_activity(
        result: CallToolResult,
        *,
        output_bytes: int,
        output_lines: int,
        seconds_since_last_output: float,
        output_observed: bool,
    ) -> None:
        activity = (
            f"{seconds_since_last_output:.1f}s since last output"
            if output_observed
            else f"no output observed for {seconds_since_last_output:.1f}s"
        )
        line = (
            f"output_activity: {output_lines} lines / {output_bytes} bytes "
            f"since last poll; {activity}"
        )
        for block in result.content:
            if isinstance(block, TextContent):
                block.text = f"{block.text}\n{line}"
                return

    @staticmethod
    def _append_resource_observation(
        result: CallToolResult,
        observation: str,
    ) -> None:
        for block in result.content:
            if isinstance(block, TextContent):
                block.text = f"{block.text}\nresource_observation: {observation}"
                return

    def _managed_process_result(
        self,
        process: ManagedShellProcess,
        *,
        yielded_reason: str | None = None,
    ) -> CallToolResult:
        self._record_buffered_process_result(process)
        return build_managed_process_result(
            process,
            yielded_reason=yielded_reason,
            minimal_process_profile=self._minimal_process_profile,
            io_drain_timeout_seconds=_IO_DRAIN_TIMEOUT_SECONDS,
        )

    async def poll_process(
        self,
        arguments: dict[str, Any] | None = None,
        *,
        progress_tool_use_id: str | None = None,
    ) -> CallToolResult:
        """Return incremental output and status for a managed process."""
        poll_started_at = time.monotonic()
        try:
            parsed = parse_poll_process_arguments(
                arguments,
                default_wait_seconds=self._process_poll_default_wait_seconds,
                max_wait_seconds=self._max_process_poll_seconds,
            )
        except ValueError as exc:
            return _text_result(str(exc), is_error=True)

        process = await self._get_managed_process(parsed.process_id)
        if process is None:
            return _text_result(
                f"Error: managed shell process {parsed.process_id!r} was not found",
                is_error=True,
            )

        async with process.poll_lock:
            async with process.lock:
                should_wait = (
                    parsed.wait_sec > 0
                    and not process.task.done()
                )

            waited = False
            output_wake = False
            poll_started_at_monotonic = time.monotonic()
            active_poll = (
                ActiveProcessPoll(
                    tool_use_id=progress_tool_use_id,
                    deadline_at=poll_started_at_monotonic + parsed.wait_sec,
                    started_at=poll_started_at_monotonic,
                )
                if should_wait and progress_tool_use_id is not None
                else None
            )
            process.active_poll = active_poll
            if active_poll is not None:
                # Quiet processes (sleep) never stream output; heartbeat keeps
                # the live countdown bar visible for the whole wait.
                active_poll.heartbeat_task = asyncio.create_task(
                    self._poll_progress_heartbeat(process, active_poll),
                    name=f"fast-agent-{process.process_id}-poll-heartbeat",
                )
            try:
                if should_wait:
                    waited = True
                    output_wake = await self._wait_for_managed_process_poll(
                        process,
                        wait_sec=parsed.wait_sec,
                        wake_on_output=parsed.wake_on_output,
                    )
            finally:
                if process.active_poll is active_poll:
                    process.active_poll = None
                if active_poll is not None:
                    if active_poll.pending_progress_task is not None:
                        active_poll.pending_progress_task.cancel()
                    if active_poll.heartbeat_task is not None:
                        active_poll.heartbeat_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await active_poll.heartbeat_task

            poll_elapsed_seconds = time.monotonic() - poll_started_at
            resource_snapshot = await self._sample_managed_process_resources(process)

            async with process.lock:
                if process.task.done():
                    poll_yield_reason = "completion"
                elif output_wake:
                    poll_yield_reason = "output"
                elif waited:
                    poll_yield_reason = "deadline"
                else:
                    poll_yield_reason = "nonblocking"
                self._record_buffered_process_result(process)
                output_bytes_since_last_poll = (
                    process.output_state.total_output_bytes
                )
                output_lines_since_last_poll = (
                    process.output_state.unread_output_line_count
                )
                seconds_since_last_output = max(
                    time.monotonic() - process.callbacks.last_output_time,
                    0.0,
                )
                output_observed = process.output_state.had_stream_output
                result = self._managed_process_result(process)
                metadata = cast(
                    "ProcessResultMetadata", process_result_metadata(result)
                )
                metadata["process_yield_reason"] = poll_yield_reason
                metadata["output_bytes_since_last_poll"] = (
                    output_bytes_since_last_poll
                )
                metadata["seconds_since_last_output"] = (
                    seconds_since_last_output
                )
                metadata["has_observed_output"] = output_observed
                metadata["poll_wait_sec"] = parsed.wait_sec
                metadata["poll_wake_on_output"] = parsed.wake_on_output
                metadata["poll_elapsed_seconds"] = poll_elapsed_seconds
                if poll_yield_reason == "deadline":
                    metadata["poll_deadline_overshoot_seconds"] = max(
                        poll_elapsed_seconds - parsed.wait_sec,
                        0.0,
                    )
                if resource_snapshot is not None:
                    metadata["resource_snapshot"] = resource_snapshot.metadata()
                    observation = observe_resource_changes(
                        process.resource_observations,
                        resource_snapshot,
                    )
                    if observation is not None:
                        metadata["resource_observation"] = observation
                        self._append_resource_observation(result, observation)
                self._append_poll_output_activity(
                    result,
                    output_bytes=output_bytes_since_last_poll,
                    output_lines=output_lines_since_last_poll,
                    seconds_since_last_output=seconds_since_last_output,
                    output_observed=output_observed,
                )
                return result

    @staticmethod
    async def _wait_for_managed_process_poll(
        process: ManagedShellProcess,
        *,
        wait_sec: int,
        wake_on_output: bool,
    ) -> bool:
        """Wait for completion/deadline, optionally returning after output settles."""
        if not wake_on_output:
            await asyncio.wait(
                (process.task,),
                timeout=wait_sec,
                return_when=asyncio.FIRST_COMPLETED,
            )
            return False

        deadline = time.monotonic() + wait_sec
        while not process.task.done():
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                return False

            process.callbacks.activity_event.clear()
            pending_output = process.output_state.total_output_bytes > 0
            seconds_since_last_output = max(
                now - process.callbacks.last_output_time,
                0.0,
            )
            if (
                pending_output
                and seconds_since_last_output >= _PROCESS_OUTPUT_DEBOUNCE_SECONDS
            ):
                return True

            quiet_wait = (
                _PROCESS_OUTPUT_DEBOUNCE_SECONDS - seconds_since_last_output
                if pending_output
                else remaining
            )
            activity_task = asyncio.create_task(
                process.callbacks.activity_event.wait()
            )
            try:
                await asyncio.wait(
                    (process.task, activity_task),
                    timeout=min(remaining, quiet_wait),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not activity_task.done():
                    activity_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await activity_task

        return False

    async def terminate_process(
        self,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Terminate one managed process through the environment cancellation contract."""
        try:
            process_id = parse_terminate_process_arguments(arguments)
        except ValueError as exc:
            return _text_result(str(exc), is_error=True)

        process = await self._get_managed_process(process_id)
        if process is None:
            return _text_result(
                f"Error: managed shell process {process_id!r} was not found",
                is_error=True,
            )

        async with process.lock:
            if process.task.done():
                return process_result(
                    f"process_id: {process_id}\noutcome: already_exited",
                    is_error=False,
                    metadata={
                        "process_id": process_id,
                        "process_status": "already_exited",
                    },
                )
            process.terminated = True
            process.request.terminate_on_cancel = True
            process.task.cancel()
            await asyncio.gather(process.task, return_exceptions=True)
            if not process.task.cancelled():
                exception = process.task.exception()
                if exception is not None:
                    process.terminated = False
                    return process_result(
                        f"process_id: {process_id}\noutcome: termination_failed\n"
                        f"error: {exception}",
                        is_error=True,
                        metadata={
                            "process_id": process_id,
                            "process_status": "termination_failed",
                        },
                    )
            return process_result(
                f"process_id: {process_id}\noutcome: terminated",
                is_error=False,
                metadata={
                    "process_id": process_id,
                    "process_status": "terminated",
                },
            )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        show_tool_call_id: bool = False,
        defer_display_to_tool_result: bool = False,
    ) -> CallToolResult:
        """Dispatch one model-facing shell lifecycle tool."""
        if name == BASH_TOOL_NAME and self._minimal_process_profile:
            try:
                parsed = parse_minimal_bash_arguments(arguments)
            except ValueError as exc:
                return self._invalid_execute_result(str(exc))
            return await self._execute_parsed(
                parsed,
                tool_use_id,
                show_tool_call_id=show_tool_call_id,
                defer_display_to_tool_result=defer_display_to_tool_result,
            )
        if name == PROCESS_TOOL_NAME and self._minimal_process_profile:
            try:
                parsed_process = parse_minimal_process_arguments(arguments)
            except ValueError as exc:
                return _text_result(str(exc), is_error=True)
            if parsed_process.action == "stop":
                return await self._call_process_lifecycle_tool(
                    TERMINATE_PROCESS_TOOL_NAME,
                    {"process_id": parsed_process.process_id},
                    tool_use_id=tool_use_id,
                )
            wait_sec = (
                0
                if parsed_process.action == "status"
                else self._process_poll_default_wait_seconds
            )
            return await self._call_process_lifecycle_tool(
                POLL_PROCESS_TOOL_NAME,
                {
                    "process_id": parsed_process.process_id,
                    "wait_sec": wait_sec,
                },
                tool_use_id=tool_use_id,
            )
        if name == EXECUTE_TOOL_NAME:
            return await self.execute(
                arguments,
                tool_use_id,
                show_tool_call_id=show_tool_call_id,
                defer_display_to_tool_result=defer_display_to_tool_result,
            )
        if name == POLL_PROCESS_TOOL_NAME:
            return await self._call_process_lifecycle_tool(
                name,
                arguments,
                tool_use_id=tool_use_id,
            )
        if name == TERMINATE_PROCESS_TOOL_NAME:
            return await self._call_process_lifecycle_tool(
                name,
                arguments,
                tool_use_id=tool_use_id,
            )
        return _text_result(f"Error: unknown shell runtime tool {name!r}", is_error=True)

    async def _call_process_lifecycle_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        *,
        tool_use_id: str | None,
    ) -> CallToolResult:
        process_id = (arguments or {}).get("process_id")
        payload = arguments or {}
        process_metadata = self.process_tool_metadata(name, payload)
        if name == POLL_PROCESS_TOOL_NAME:
            start_details = self._process_progress_details(process_metadata)
            operation = self.poll_process(
                arguments,
                progress_tool_use_id=tool_use_id,
            )
        else:
            start_details = self._process_progress_details(process_metadata)
            operation = self.terminate_process(arguments)

        elapsed = process_metadata.get("elapsed_seconds")
        process_elapsed_seconds = (
            float(elapsed)
            if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
            else None
        )
        command = process_metadata.get("command_summary")
        wait_sec = process_metadata.get("wait_sec")
        seconds_since_last_output = process_metadata.get("seconds_since_last_output")
        has_observed_output = process_metadata.get("has_observed_output")
        total_output_bytes = process_metadata.get("total_output_bytes")
        self._emit_progress_event(
            action=ProgressAction.CALLING_TOOL,
            tool_use_id=tool_use_id,
            tool_name=name,
            tool_event="start",
            details=start_details,
            process_elapsed_seconds=process_elapsed_seconds,
            process_command=command if isinstance(command, str) else None,
            process_id=str(process_metadata.get("process_id") or "process"),
            process_wait_seconds=(
                wait_sec
                if name == POLL_PROCESS_TOOL_NAME and type(wait_sec) is int
                else None
            ),
            process_has_observed_output=(
                has_observed_output if isinstance(has_observed_output, bool) else None
            ),
            process_seconds_since_last_output=(
                float(seconds_since_last_output)
                if isinstance(seconds_since_last_output, (int, float))
                and not isinstance(seconds_since_last_output, bool)
                else None
            ),
            process_total_output_bytes=(
                total_output_bytes
                if type(total_output_bytes) is int and total_output_bytes >= 0
                else None
            ),
        )
        result = await operation
        metadata = process_result_metadata(result)
        status = metadata.get("process_status") if metadata is not None else None
        yield_reason = (
            metadata.get("process_yield_reason") if metadata is not None else None
        )
        details = f"{process_id}: {status}" if process_id and status else status
        self._emit_progress_event(
            action=ProgressAction.TOOL_PROGRESS,
            tool_use_id=tool_use_id,
            tool_name=name,
            details=details or ("failed" if result.isError else "completed"),
            tool_state="failed" if result.isError else "completed",
            tool_terminal=True,
            process_yield_reason=yield_reason,
        )
        return result

    async def close(self) -> None:
        """Terminate session processes and detach persistent processes."""
        async with self._processes_lock:
            processes = list(self._managed_processes.values())
            self._managed_processes.clear()
        running = [process for process in processes if not process.task.done()]
        if running:
            console.console.print(
                f"Warning: {len(running)} background process"
                f"{'es are' if len(running) != 1 else ' is'} still running "
                "at fast-agent shutdown:",
                style="yellow",
            )
            for process in running:
                os_pid = process.callbacks.os_process_id
                pid_details = (
                    f", os_pid={os_pid}" if os_pid is not None else ""
                )
                console.console.print(
                    f"  {process.process_id}{pid_details}, "
                    f"lifecycle={process.lifecycle}",
                    style="yellow",
                )
        for process in processes:
            if not process.task.done():
                process.terminated = process.lifecycle == "session"
                process.task.cancel()
        if processes:
            await asyncio.gather(
                *(process.task for process in processes),
                return_exceptions=True,
            )

    async def execute_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        execution = await execute_shell(
            self._environment,
            command,
            cwd=cwd if cwd is not None else self._working_directory,
            env=env,
            timeout=self._timeout_seconds if timeout is None else timeout,
        )
        return execution

    async def execute_direct_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        """Execute a user-entered shell command and display output live when available."""
        runtime_execution = await self._execute_shell_command(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            output_byte_limit=None,
            defer_display_to_tool_result=False,
            display_line_limit=None,
        )
        execution = runtime_execution.execution
        output_state = runtime_execution.output_state
        display_state = runtime_execution.display_state

        if not output_state.had_stream_output:
            if execution.result.stdout:
                self._record_stream_output(
                    execution.result.stdout,
                    style=None,
                    output_state=output_state,
                    display_state=display_state,
                    is_stderr=False,
                )
            if execution.result.stderr:
                self._record_stream_output(
                    execution.result.stderr,
                    style="red",
                    output_state=output_state,
                    display_state=display_state,
                    is_stderr=True,
                )
        self._flush_live_display_tail(display_state)
        if execution.timed_out:
            self._print_timeout_notice(
                display_state,
                timeout_seconds=execution.options.timeout_seconds,
            )
        return execution.result

    async def execute(
        self,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        show_tool_call_id: bool = False,
        defer_display_to_tool_result: bool = False,
    ) -> CallToolResult:
        """Execute a command until completion or yield it as a managed process."""
        try:
            parsed = parse_execute_arguments(arguments)
        except ValueError as exc:
            return self._invalid_execute_result(str(exc))
        return await self._execute_parsed(
            parsed,
            tool_use_id,
            show_tool_call_id=show_tool_call_id,
            defer_display_to_tool_result=defer_display_to_tool_result,
        )

    async def _execute_parsed(
        self,
        parsed: ShellExecuteArguments,
        tool_use_id: str | None,
        *,
        show_tool_call_id: bool,
        defer_display_to_tool_result: bool,
    ) -> CallToolResult:
        idle_yield_seconds = (
            self._idle_yield_seconds
            if parsed.yield_after_idle_sec is None
            else parsed.yield_after_idle_sec
        )
        self._logger.debug(
            "Executing command with "
            f"idle_yield={idle_yield_seconds}s, "
            f"foreground_yield={self._foreground_yield_seconds}s, "
            f"background={parsed.background}, "
            f"lifecycle={parsed.lifecycle}"
        )

        progress_context = progress_display.paused() if display_tools_enabled() else nullcontext()
        with progress_context:
            try:
                self._emit_progress_event(
                    action=ProgressAction.CALLING_TOOL,
                    tool_use_id=tool_use_id,
                    tool_event="start",
                )

                process = await self._start_managed_process(
                    parsed,
                    defer_display_to_tool_result=defer_display_to_tool_result,
                )
                if parsed.background:
                    started_task = asyncio.create_task(
                        process.callbacks.started_event.wait()
                    )
                    try:
                        await asyncio.wait(
                            (process.task, started_task),
                            timeout=1,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        if not started_task.done():
                            started_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await started_task
                    yielded_reason = "background"
                else:
                    yielded_reason = await self._wait_for_initial_process_result(
                        process,
                        idle_yield_seconds=idle_yield_seconds,
                    )

                result = self._managed_process_result(
                    process,
                    yielded_reason=yielded_reason,
                )
                if process.task.done() and not process.task.cancelled():
                    exception = process.task.exception()
                    if exception is None:
                        result = self._finalize_shell_result_display(
                            result,
                            shell_result=process.task.result().result,
                            output_state=process.output_state,
                            display_state=process.display_state,
                            tool_use_id=tool_use_id,
                            show_tool_call_id=show_tool_call_id,
                            defer_display_to_tool_result=defer_display_to_tool_result,
                        )
                else:
                    self._flush_live_display_tail(process.display_state)
                    process.display_state.use_live_shell_display = False
                    if defer_display_to_tool_result:
                        cast("Any", result)._suppress_display = False
                    else:
                        self._display.show_managed_process_status(
                            process_id=process.process_id,
                            status="running",
                            reason=yielded_reason,
                            elapsed_seconds=time.monotonic() - process.started_at,
                            os_process_id=process.callbacks.os_process_id,
                        )

                metadata = cast(
                    "ProcessResultMetadata", process_result_metadata(result)
                )
                process_status = metadata["process_status"]
                if process_status == "running":
                    completion_details = f"running ({process.process_id})"
                elif (
                    process.task.done()
                    and not process.task.cancelled()
                    and process.task.exception() is not None
                ):
                    completion_details = f"failed: {process.task.exception()}"
                elif (
                    process.task.done()
                    and not process.task.cancelled()
                    and process.task.exception() is None
                ):
                    completion_details = (
                        f"{process_status} (exit "
                        f"{process.task.result().result.exit_code})"
                    )
                else:
                    completion_details = process_status
                self._emit_progress_event(
                    action=ProgressAction.TOOL_PROGRESS,
                    tool_use_id=tool_use_id,
                    details=completion_details,
                    tool_state="failed" if result.isError else "completed",
                    tool_terminal=True,
                )
                return result

            except Exception as exc:
                self._logger.error(f"Execute tool failed: {exc}")
                self._emit_progress_event(
                    action=ProgressAction.TOOL_PROGRESS,
                    tool_use_id=tool_use_id,
                    details=f"failed: {exc}",
                    tool_state="failed",
                    tool_terminal=True,
                )
                return _text_result(f"Command execution failed: {exc}", is_error=True)

    def _emit_progress_event(
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
