from __future__ import annotations

import asyncio
import posixpath
import time
from collections import deque
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
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
from fast_agent.tools.filesystem_tool_args import (
    coerce_optional_string_argument,
    coerce_positive_int_argument,
    coerce_required_string_argument,
    coerce_tool_arguments,
)
from fast_agent.tools.local_shell_executor import LocalShellExecutor
from fast_agent.tools.output_truncation import format_output_truncation_notice
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
    EXECUTE_TOOL_NAME,
    POLL_PROCESS_TOOL_NAME,
    TERMINATE_PROCESS_TOOL_NAME,
)

_IO_DRAIN_TIMEOUT_SECONDS = 2.0
_DEFAULT_IDLE_YIELD_SECONDS = 10
_DEFAULT_FOREGROUND_YIELD_SECONDS = 30
_MAX_IDLE_YIELD_SECONDS = 30
_MAX_PROCESS_POLL_SECONDS = 30
_EXECUTE_ARGUMENTS = frozenset(
    {
        "command",
        "cwd",
        "background",
        "lifecycle",
        "yield_after_idle_sec",
        "output_byte_limit",
    }
)
_POLL_PROCESS_ARGUMENTS = frozenset({"process_id", "wait_sec", "wake_on_output"})
_TERMINATE_PROCESS_ARGUMENTS = frozenset({"process_id"})


def _text_result(message: str, *, is_error: bool) -> CallToolResult:
    return CallToolResult(
        isError=is_error,
        content=[TextContent(type="text", text=message)],
    )


@dataclass(slots=True)
class _ShellOutputState:
    output_byte_limit: int
    output_segments: list[str] = field(default_factory=list)
    output_tail_bytes: bytearray = field(default_factory=bytearray)
    output_bytes: int = 0
    total_output_bytes: int = 0
    output_truncated: bool = False
    truncation_notice_printed: bool = False
    had_stream_output: bool = False
    output_line_count: int = 0
    unread_output_line_count: int = 0
    lifetime_output_bytes: int = 0


@dataclass(slots=True)
class _ShellDisplayState:
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
class _ShellRuntimeCallbacks:
    runtime: ShellRuntime
    output_state: _ShellOutputState
    display_state: _ShellDisplayState
    activity_event: asyncio.Event = field(default_factory=asyncio.Event)
    started_event: asyncio.Event = field(default_factory=asyncio.Event)
    os_process_id: int | None = None
    last_output_time: float = field(default_factory=time.monotonic)

    async def on_started(self, process_id: int | None) -> None:
        self.os_process_id = process_id
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
class _ShellRuntimeExecution:
    execution: ShellExecution
    output_state: _ShellOutputState
    display_state: _ShellDisplayState


@dataclass(frozen=True, slots=True)
class _ShellExecuteArguments:
    command: str
    cwd: str | None
    background: bool
    lifecycle: Literal["session", "persistent"]
    yield_after_idle_sec: int | None
    output_byte_limit: int | None


@dataclass(frozen=True, slots=True)
class _PollProcessArguments:
    process_id: str
    wait_sec: int
    wake_on_output: bool


@dataclass(slots=True)
class _ManagedShellProcess:
    process_id: str
    command: str
    working_directory: str
    started_at: float
    task: asyncio.Task[ShellExecution]
    request: ShellExecutionRequest
    lifecycle: Literal["session", "persistent"]
    callbacks: _ShellRuntimeCallbacks
    output_state: _ShellOutputState
    display_state: _ShellDisplayState
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    completed_at: float | None = None
    terminated: bool = False
    buffered_result_recorded: bool = False


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
        self._managed_processes: dict[str, _ManagedShellProcess] = {}
        self._next_process_id = 1
        self._processes_lock = asyncio.Lock()
        self._output_display_lines: int | None = None
        self._show_bash_output = True
        self._prefer_local_shell = False
        if config is not None:
            shell_config = config.shell_execution
            self._output_display_lines = shell_config.output_display_lines
            self._show_bash_output = shell_config.show_bash
            self._prefer_local_shell = shell_config.prefer_local_shell

        if self.enabled:
            # Detect the shell early so we can include it in the tool description
            runtime_info = self.runtime_info()
            shell_name = runtime_info.name

            self._tool = set_tool_source(
                Tool(
                    name=EXECUTE_TOOL_NAME,
                    description=(
                        f"Run one shell command in {shell_name}. Most commands return when they "
                        "exit. If a foreground command remains active for 10 seconds without "
                        "output or 30 seconds total, it keeps running and returns a process ID; "
                        "use poll_process to monitor it or terminate_process to stop it. Set "
                        "`background=true` for known long-running commands. Background commands "
                        "default to `lifecycle='session'` and are terminated when the agent "
                        "runtime exits; use `lifecycle='persistent'` only when a command must "
                        "remain running afterward. Do not append '&'. "
                        "`cwd` and `output_byte_limit` apply only to this command. Pipelines report "
                        "the final command's status unless you enable `pipefail`."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Command string only - no shell executable prefix (correct: 'pwd', incorrect: 'bash -c pwd').",
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Optional working directory for this command only.",
                            },
                            "background": {
                                "type": "boolean",
                                "description": (
                                    "Return promptly while the command continues running as a "
                                    "managed process. By default it is terminated when the agent "
                                    "runtime exits. Set lifecycle='persistent' only when it must "
                                    "remain running afterward. Do not append '&' to the command."
                                ),
                            },
                            "lifecycle": {
                                "type": "string",
                                "enum": ["session", "persistent"],
                                "default": "session",
                                "description": (
                                    "Lifetime of a background command. 'session' terminates it "
                                    "when the agent runtime exits. 'persistent' leaves it running "
                                    "in the execution environment after the agent exits. Applies "
                                    "only when background=true."
                                ),
                            },
                            "yield_after_idle_sec": {
                                "type": "integer",
                                "description": (
                                    "Optional seconds without output before returning a live "
                                    "process ID without stopping the command. Defaults to 10."
                                ),
                                "minimum": 1,
                                "maximum": _MAX_IDLE_YIELD_SECONDS,
                            },
                            "output_byte_limit": {
                                "type": "integer",
                                "description": (
                                    "Optional maximum output bytes returned to the model for this "
                                    "command. Complete output is not retained after truncation."
                                ),
                                "minimum": 1,
                                "maximum": MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
                            },
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                ),
                SHELL_TOOL_SOURCE,
            )
            self._poll_process_tool = set_tool_source(
                Tool(
                    name=POLL_PROCESS_TOOL_NAME,
                    description=(
                        "Get new output and status from a managed shell process. Omit `wait_sec` "
                        "or use 0 for a non-blocking poll; a positive value waits for completion "
                        "or new output. Set `wake_on_output=false` to buffer output and wait only "
                        "for completion or the deadline. Repeated polls return only output not "
                        "returned previously."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "process_id": {
                                "type": "string",
                                "description": "Process ID returned by execute.",
                            },
                            "wait_sec": {
                                "type": "integer",
                                "description": "Optional wait in seconds, from 0 through 30.",
                                "minimum": 0,
                                "maximum": _MAX_PROCESS_POLL_SECONDS,
                            },
                            "wake_on_output": {
                                "type": "boolean",
                                "default": True,
                                "description": (
                                    "Wake when new output arrives. Set to false to buffer output "
                                    "until the process completes or wait_sec elapses."
                                ),
                            },
                        },
                        "required": ["process_id"],
                        "additionalProperties": False,
                    },
                ),
                SHELL_TOOL_SOURCE,
            )
            self._terminate_process_tool = set_tool_source(
                Tool(
                    name=TERMINATE_PROCESS_TOOL_NAME,
                    description=(
                        "Terminate a managed shell process and its process group. Returns success "
                        "if the process was terminated or had already exited."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "process_id": {
                                "type": "string",
                                "description": "Process ID returned by execute.",
                            }
                        },
                        "required": ["process_id"],
                        "additionalProperties": False,
                    },
                ),
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
        process: _ManagedShellProcess,
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
            parsed = self._parse_execute_arguments(arguments)
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
        metadata: dict[str, Any] = {
            "variant": "shell_process",
            "action": "poll"
            if tool_name == POLL_PROCESS_TOOL_NAME
            else "terminate",
            "process_id": arguments.get("process_id"),
            "wait_sec": arguments.get("wait_sec", 0),
        }
        process_id = arguments.get("process_id")
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
            }
        )
        return metadata

    def _process_progress_details(
        self,
        tool_name: str,
        metadata: dict[str, Any],
    ) -> str:
        process_id = str(metadata.get("process_id") or "process")
        parts: list[str] = []

        os_process_id = metadata.get("os_process_id")
        if isinstance(os_process_id, int) and not isinstance(os_process_id, bool):
            parts.append(f"pid {os_process_id}")
        else:
            parts.append(process_id)

        wait_sec = metadata.get("wait_sec")
        if tool_name == POLL_PROCESS_TOOL_NAME and type(wait_sec) is int and wait_sec > 0:
            parts.append(f"≤{wait_sec}s")
        return " · ".join(parts)

    def _invalid_execute_result(self, message: str) -> CallToolResult:
        return _text_result(message, is_error=True)

    def _parse_execute_arguments(
        self,
        arguments: dict[str, Any] | None,
    ) -> _ShellExecuteArguments:
        payload = coerce_tool_arguments(arguments)
        unknown = sorted(set(payload) - _EXECUTE_ARGUMENTS)
        if unknown:
            if unknown in (["timeout"], ["timeout_sec"]):
                raise ValueError(
                    f"Error: unknown argument {unknown[0]!r}; use 'yield_after_idle_sec' to "
                    "return a live process ID without stopping the command"
                )
            rendered = ", ".join(repr(name) for name in unknown)
            raise ValueError(f"Error: unknown execute argument(s): {rendered}")

        yield_after_idle_sec = coerce_positive_int_argument(
            payload.get("yield_after_idle_sec"),
            "yield_after_idle_sec",
        )
        if (
            yield_after_idle_sec is not None
            and yield_after_idle_sec > _MAX_IDLE_YIELD_SECONDS
        ):
            raise ValueError(
                f"Error: 'yield_after_idle_sec' argument must be at most "
                f"{_MAX_IDLE_YIELD_SECONDS}"
            )
        background = payload.get("background", False)
        if type(background) is not bool:
            raise ValueError("Error: 'background' argument must be a boolean")
        lifecycle = payload.get("lifecycle", "session")
        if lifecycle not in {"session", "persistent"}:
            raise ValueError(
                "Error: 'lifecycle' argument must be 'session' or 'persistent'"
            )
        if lifecycle == "persistent" and not background:
            raise ValueError(
                "Error: lifecycle='persistent' requires background=true"
            )

        output_byte_limit = coerce_positive_int_argument(
            payload.get("output_byte_limit"),
            "output_byte_limit",
        )
        if (
            output_byte_limit is not None
            and output_byte_limit > MAX_TERMINAL_OUTPUT_BYTE_LIMIT
        ):
            raise ValueError(
                "Error: 'output_byte_limit' argument must be at most "
                f"{MAX_TERMINAL_OUTPUT_BYTE_LIMIT}"
            )

        return _ShellExecuteArguments(
            command=coerce_required_string_argument(
                payload.get("command"),
                "command",
                strip=True,
            ),
            cwd=coerce_optional_string_argument(
                payload.get("cwd"),
                "cwd",
                empty_as_none=True,
                strip=True,
            ),
            background=background,
            lifecycle=cast("Literal['session', 'persistent']", lifecycle),
            yield_after_idle_sec=yield_after_idle_sec,
            output_byte_limit=output_byte_limit,
        )

    @staticmethod
    def _parse_poll_process_arguments(
        arguments: dict[str, Any] | None,
    ) -> _PollProcessArguments:
        payload = coerce_tool_arguments(arguments)
        unknown = sorted(set(payload) - _POLL_PROCESS_ARGUMENTS)
        if unknown:
            rendered = ", ".join(repr(name) for name in unknown)
            raise ValueError(f"Error: unknown poll_process argument(s): {rendered}")
        wait_sec = payload.get("wait_sec", 0)
        if type(wait_sec) is not int or wait_sec < 0:
            raise ValueError("Error: 'wait_sec' argument must be a non-negative integer")
        if wait_sec > _MAX_PROCESS_POLL_SECONDS:
            raise ValueError(
                f"Error: 'wait_sec' argument must be at most {_MAX_PROCESS_POLL_SECONDS}"
            )
        wake_on_output = payload.get("wake_on_output", True)
        if type(wake_on_output) is not bool:
            raise ValueError("Error: 'wake_on_output' argument must be a boolean")
        return _PollProcessArguments(
            process_id=coerce_required_string_argument(
                payload.get("process_id"),
                "process_id",
                strip=True,
            ),
            wait_sec=wait_sec,
            wake_on_output=wake_on_output,
        )

    @staticmethod
    def _parse_terminate_process_arguments(
        arguments: dict[str, Any] | None,
    ) -> str:
        payload = coerce_tool_arguments(arguments)
        unknown = sorted(set(payload) - _TERMINATE_PROCESS_ARGUMENTS)
        if unknown:
            rendered = ", ".join(repr(name) for name in unknown)
            raise ValueError(f"Error: unknown terminate_process argument(s): {rendered}")
        return coerce_required_string_argument(
            payload.get("process_id"),
            "process_id",
            strip=True,
        )

    def _build_display_state(
        self,
        *,
        defer_display_to_tool_result: bool,
        display_line_limit: int | None = None,
    ) -> _ShellDisplayState:
        use_live_shell_display = (
            self._show_bash_output and not defer_display_to_tool_result and display_tools_enabled()
        )
        state = _ShellDisplayState(
            use_live_shell_display=use_live_shell_display,
            display_line_limit=display_line_limit,
        )
        if display_line_limit is not None and display_line_limit > 0:
            display_window = split_shell_output_line_limit(display_line_limit)
            state.display_head_limit = display_window.head_lines
            state.display_tail_limit = display_window.tail_lines
            state.display_tail_buffer = deque(maxlen=max(display_window.tail_lines, 1))
        return state

    def _append_output_text(self, output_text: str, state: _ShellOutputState) -> None:
        output_blob = output_text.encode("utf-8", errors="replace")
        state.total_output_bytes += len(output_blob)
        state.lifetime_output_bytes += len(output_blob)
        self._append_output_tail(output_blob, state)
        if state.output_truncated:
            return

        remaining = state.output_byte_limit - state.output_bytes
        if remaining <= 0:
            state.output_truncated = True
            return
        if len(output_blob) <= remaining:
            state.output_segments.append(output_text)
            state.output_bytes += len(output_blob)
            return

        truncated_text = output_blob[:remaining].decode("utf-8", errors="replace")
        if truncated_text:
            state.output_segments.append(truncated_text)
        state.output_bytes += remaining
        state.output_truncated = True

    def _append_output_tail(self, output_blob: bytes, state: _ShellOutputState) -> None:
        tail_limit = self._truncated_tail_byte_limit(state.output_byte_limit)
        if len(output_blob) >= tail_limit:
            state.output_tail_bytes = bytearray(output_blob[-tail_limit:])
            return

        state.output_tail_bytes.extend(output_blob)
        overflow = len(state.output_tail_bytes) - tail_limit
        if overflow > 0:
            del state.output_tail_bytes[:overflow]

    @staticmethod
    def _truncated_tail_byte_limit(output_byte_limit: int) -> int:
        return max(output_byte_limit // 2, 1)

    @staticmethod
    def _truncated_head_byte_limit(output_byte_limit: int) -> int:
        return max(
            output_byte_limit - ShellRuntime._truncated_tail_byte_limit(output_byte_limit),
            1,
        )

    def _maybe_print_truncation_notice(
        self,
        *,
        output_state: _ShellOutputState,
        display_state: _ShellDisplayState,
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
                style="black on red",
            )
        output_state.truncation_notice_printed = True

    def _print_timeout_notice(
        self,
        display_state: _ShellDisplayState,
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
        display_state: _ShellDisplayState,
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
        output_state: _ShellOutputState,
        display_state: _ShellDisplayState,
        is_stderr: bool,
    ) -> None:
        output_state.had_stream_output = True
        output_state.output_line_count += 1
        output_state.unread_output_line_count += 1
        output_text = text if not is_stderr else f"[stderr] {text}"
        self._append_output_text(output_text, output_state)
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

    def _truncation_summary(
        self,
        output_state: _ShellOutputState,
        *,
        head_bytes: int | None = None,
        tail_bytes: int | None = None,
    ) -> str | None:
        if not output_state.output_truncated:
            return None
        retained_bytes = (
            output_state.output_bytes
            if head_bytes is None or tail_bytes is None
            else head_bytes + tail_bytes
        )
        retained_tokens = max(int(retained_bytes / TERMINAL_BYTES_PER_TOKEN), 1)
        total_tokens = max(int(output_state.total_output_bytes / TERMINAL_BYTES_PER_TOKEN), 1)
        omitted_bytes = max(output_state.total_output_bytes - retained_bytes, 0)
        if head_bytes is not None and tail_bytes is not None:
            return format_output_truncation_notice(
                label="Output",
                total_bytes=output_state.total_output_bytes,
                head_bytes=head_bytes,
                tail_bytes=tail_bytes,
                guidance="Increase shell_execution.output_byte_limit to retain more.",
            )
        return (
            "[Output truncated: retained "
            f"{output_state.output_bytes} of {output_state.total_output_bytes} bytes "
            f"(~{retained_tokens} of ~{total_tokens} tokens); "
            f"omitted {omitted_bytes} bytes. "
            "Increase shell_execution.output_byte_limit to retain more.]"
        )

    def _truncated_combined_output(self, output_state: _ShellOutputState) -> str:
        head_limit = self._truncated_head_byte_limit(output_state.output_byte_limit)
        head_blob = "".join(output_state.output_segments).encode("utf-8", errors="replace")[
            :head_limit
        ]
        tail_blob = bytes(output_state.output_tail_bytes)[
            -self._truncated_tail_byte_limit(output_state.output_byte_limit) :
        ]

        parts: list[str] = []
        if head_blob:
            head_text = head_blob.decode("utf-8", errors="replace")
            parts.append(head_text if head_text.endswith("\n") else f"{head_text}\n")

        truncation_summary = self._truncation_summary(
            output_state,
            head_bytes=len(head_blob),
            tail_bytes=len(tail_blob),
        )
        if truncation_summary:
            parts.append(f"{truncation_summary}\n")

        if tail_blob:
            tail_text = tail_blob.decode("utf-8", errors="replace")
            parts.append(tail_text if tail_text.endswith("\n") else f"{tail_text}\n")

        return "".join(parts)

    def _consume_combined_output(self, output_state: _ShellOutputState) -> str:
        combined_output = (
            self._truncated_combined_output(output_state)
            if output_state.output_truncated
            else "".join(output_state.output_segments)
        )
        output_state.output_segments.clear()
        output_state.output_tail_bytes.clear()
        output_state.output_bytes = 0
        output_state.total_output_bytes = 0
        output_state.output_truncated = False
        output_state.truncation_notice_printed = False
        output_state.unread_output_line_count = 0
        return combined_output

    def _build_shell_result(
        self,
        *,
        execution: ShellExecution,
        output_state: _ShellOutputState,
    ) -> tuple[CallToolResult, str]:
        shell_result = execution.result
        combined_output = (
            self._truncated_combined_output(output_state)
            if output_state.output_truncated
            else "".join(output_state.output_segments)
        )
        if combined_output and not combined_output.endswith("\n"):
            combined_output += "\n"

        if execution.io_drain_timed_out:
            combined_output += (
                f"(output collection stopped after {_IO_DRAIN_TIMEOUT_SECONDS:.1f}s "
                "because stdout/stderr pipes remained open)\n"
            )

        if execution.timed_out:
            combined_output += (
                f"(timeout after {execution.options.timeout_seconds:g}s - process terminated)"
            )
            return (
                _text_result(combined_output, is_error=True),
                f"failed (timeout after {execution.options.timeout_seconds:g}s)",
            )

        combined_output += f"process exit code was {shell_result.exit_code}"
        completion_state = "completed" if shell_result.exit_code == 0 else "failed"
        return (
            _text_result(combined_output, is_error=shell_result.exit_code != 0),
            f"{completion_state} (exit {shell_result.exit_code})",
        )

    def _flush_live_display_tail(self, display_state: _ShellDisplayState) -> None:
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
        output_state: _ShellOutputState,
        display_state: _ShellDisplayState,
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
        output_state = _ShellOutputState(
            output_byte_limit=(
                self._output_byte_limit
                if output_byte_limit is None
                else output_byte_limit
            )
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
            callbacks=_ShellRuntimeCallbacks(
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
        parsed: _ShellExecuteArguments,
        *,
        defer_display_to_tool_result: bool,
    ) -> _ManagedShellProcess:
        output_state = _ShellOutputState(
            output_byte_limit=(
                self._output_byte_limit
                if parsed.output_byte_limit is None
                else parsed.output_byte_limit
            )
        )
        display_state = self._build_display_state(
            defer_display_to_tool_result=defer_display_to_tool_result,
            display_line_limit=self._output_display_lines,
        )
        callbacks = _ShellRuntimeCallbacks(
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
            process = _ManagedShellProcess(
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
        process: _ManagedShellProcess,
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

    async def _get_managed_process(self, process_id: str) -> _ManagedShellProcess | None:
        async with self._processes_lock:
            return self._managed_processes.get(process_id)

    async def _wait_for_initial_process_result(
        self,
        process: _ManagedShellProcess,
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

    def _record_buffered_process_result(self, process: _ManagedShellProcess) -> None:
        if process.buffered_result_recorded or not process.task.done():
            return
        process.buffered_result_recorded = True
        if process.task.cancelled() or process.task.exception() is not None:
            return
        execution = process.task.result()
        if process.output_state.had_stream_output:
            return
        if execution.result.stdout:
            self._record_stream_output(
                execution.result.stdout,
                style=None,
                output_state=process.output_state,
                display_state=process.display_state,
                is_stderr=False,
            )
        if execution.result.stderr:
            self._record_stream_output(
                execution.result.stderr,
                style="red",
                output_state=process.output_state,
                display_state=process.display_state,
                is_stderr=True,
            )

    def _managed_process_result(
        self,
        process: _ManagedShellProcess,
        *,
        yielded_reason: str | None = None,
    ) -> CallToolResult:
        self._record_buffered_process_result(process)
        unread_output_line_count = process.output_state.unread_output_line_count
        output = self._consume_combined_output(process.output_state)
        sections: list[str] = []
        if output:
            sections.append(output.rstrip("\n"))

        elapsed = time.monotonic() - process.started_at
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
                        f"Use {POLL_PROCESS_TOOL_NAME} to monitor it or "
                        f"{TERMINATE_PROCESS_TOOL_NAME} to stop it."
                    ),
                ]
            )
            if process.callbacks.os_process_id is not None:
                sections.insert(-3, f"os_pid: {process.callbacks.os_process_id}")
            result = _text_result("\n".join(sections), is_error=False)
            result_meta = cast("Any", result)
            result_meta.process_id = process.process_id
            result_meta.process_status = "running"
            result_meta.process_yield_reason = yielded_reason
            result_meta.process_elapsed_seconds = elapsed
            result_meta.os_process_id = process.callbacks.os_process_id
            result_meta.output_line_count = unread_output_line_count
            result_meta._suppress_display = yielded_reason is not None or not output
            return result

        if process.task.cancelled():
            status = "terminated" if process.terminated else "cancelled"
            sections.extend(
                [
                    f"process_id: {process.process_id}",
                    f"process status: {status}",
                ]
            )
            result = _text_result("\n".join(sections), is_error=False)
            result_meta = cast("Any", result)
            result_meta.process_id = process.process_id
            result_meta.process_status = status
            result_meta.output_line_count = unread_output_line_count
            return result

        exception = process.task.exception()
        if exception is not None:
            sections.extend(
                [
                    f"process_id: {process.process_id}",
                    f"Command execution failed: {exception}",
                ]
            )
            result = _text_result("\n".join(sections), is_error=True)
            result_meta = cast("Any", result)
            result_meta.process_id = process.process_id
            result_meta.process_status = "failed"
            result_meta.output_line_count = unread_output_line_count
            return result

        execution = process.task.result()
        if execution.io_drain_timed_out:
            sections.append(
                f"Output collection stopped after {_IO_DRAIN_TIMEOUT_SECONDS:.1f}s because "
                "stdout/stderr pipes remained open."
            )
        sections.extend(
            [
                f"process_id: {process.process_id}",
                f"process exit code was {execution.result.exit_code}",
            ]
        )
        result = _text_result(
            "\n".join(sections),
            is_error=execution.result.exit_code != 0,
        )
        result_meta = cast("Any", result)
        result_meta.process_id = process.process_id
        result_meta.process_status = (
            "completed" if execution.result.exit_code == 0 else "failed"
        )
        result_meta.exit_code = execution.result.exit_code
        result_meta.output_line_count = unread_output_line_count
        return result

    async def poll_process(
        self,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Return incremental output and status for a managed process."""
        try:
            parsed = self._parse_poll_process_arguments(arguments)
        except ValueError as exc:
            return _text_result(str(exc), is_error=True)

        process = await self._get_managed_process(parsed.process_id)
        if process is None:
            return _text_result(
                f"Error: managed shell process {parsed.process_id!r} was not found",
                is_error=True,
            )

        async with process.lock:
            if (
                parsed.wait_sec > 0
                and not process.task.done()
                and (
                    not parsed.wake_on_output
                    or process.output_state.total_output_bytes == 0
                )
            ):
                if parsed.wake_on_output:
                    process.callbacks.activity_event.clear()
                    should_wait = (
                        not process.task.done()
                        and process.output_state.total_output_bytes == 0
                    )
                else:
                    should_wait = not process.task.done()
                if should_wait:
                    wait_tasks: tuple[asyncio.Future[Any] | asyncio.Task[Any], ...]
                    activity_task: asyncio.Task[bool] | None = None
                    if parsed.wake_on_output:
                        activity_task = asyncio.create_task(
                            process.callbacks.activity_event.wait()
                        )
                        wait_tasks = (process.task, activity_task)
                    else:
                        wait_tasks = (process.task,)
                    try:
                        await asyncio.wait(
                            wait_tasks,
                            timeout=parsed.wait_sec,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        if activity_task is not None and not activity_task.done():
                            activity_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await activity_task
            return self._managed_process_result(process)

    async def terminate_process(
        self,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Terminate one managed process through the environment cancellation contract."""
        try:
            process_id = self._parse_terminate_process_arguments(arguments)
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
                result = _text_result(
                    f"process_id: {process_id}\noutcome: already_exited",
                    is_error=False,
                )
                result_meta = cast("Any", result)
                result_meta.process_id = process_id
                result_meta.process_status = "already_exited"
                return result
            process.terminated = True
            process.request.terminate_on_cancel = True
            process.task.cancel()
            await asyncio.gather(process.task, return_exceptions=True)
            if not process.task.cancelled():
                exception = process.task.exception()
                if exception is not None:
                    process.terminated = False
                    result = _text_result(
                        f"process_id: {process_id}\noutcome: termination_failed\n"
                        f"error: {exception}",
                        is_error=True,
                    )
                    result_meta = cast("Any", result)
                    result_meta.process_id = process_id
                    result_meta.process_status = "termination_failed"
                    return result
            result = _text_result(
                f"process_id: {process_id}\noutcome: terminated",
                is_error=False,
            )
            result_meta = cast("Any", result)
            result_meta.process_id = process_id
            result_meta.process_status = "terminated"
            return result

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
            start_details = self._process_progress_details(name, process_metadata)
            operation = self.poll_process(arguments)
        else:
            start_details = self._process_progress_details(name, process_metadata)
            operation = self.terminate_process(arguments)

        elapsed = process_metadata.get("elapsed_seconds")
        process_elapsed_seconds = (
            float(elapsed)
            if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
            else None
        )
        command = process_metadata.get("command_summary")
        self._emit_progress_event(
            action=ProgressAction.CALLING_TOOL,
            tool_use_id=tool_use_id,
            tool_name=name,
            tool_event="start",
            details=start_details,
            process_elapsed_seconds=process_elapsed_seconds,
            process_command=command if isinstance(command, str) else None,
        )
        result = await operation
        status = getattr(result, "process_status", None)
        details = f"{process_id}: {status}" if process_id and status else status
        self._emit_progress_event(
            action=ProgressAction.TOOL_PROGRESS,
            tool_use_id=tool_use_id,
            tool_name=name,
            details=details or ("failed" if result.isError else "completed"),
            tool_state="failed" if result.isError else "completed",
            tool_terminal=True,
        )
        return result

    async def close(self) -> None:
        """Terminate session processes and detach persistent processes."""
        async with self._processes_lock:
            processes = list(self._managed_processes.values())
            self._managed_processes.clear()
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
            parsed = self._parse_execute_arguments(arguments)
        except ValueError as exc:
            return self._invalid_execute_result(str(exc))
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
                    yielded_reason = "background" if not process.task.done() else None
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

                process_status = cast("Any", result).process_status
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
                }.items()
                if value is not None
            },
        )

        try:
            info("Shell tool lifecycle", data=payload)
        except TypeError:
            # Standard library loggers reject custom keyword arguments.
            return
        except Exception:
            return
