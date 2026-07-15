from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from mcp.types import CallToolResult, TextContent, Tool
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from fast_agent.config import Settings
    from fast_agent.tools.execution_environment import ShellEnvironment, ShellExecutionResult

# Import tool progress context for reporting shell execution progress
from fast_agent.agents.tool_agent import _tool_progress_context
from fast_agent.constants import (
    DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
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
from fast_agent.utils.tool_names import EXECUTE_TOOL_NAME

_IO_DRAIN_TIMEOUT_SECONDS = 2.0


def _text_result(message: str, *, is_error: bool) -> CallToolResult:
    return CallToolResult(
        isError=is_error,
        content=[TextContent(type="text", text=message)],
    )


@dataclass(slots=True)
class _ShellOutputState:
    output_segments: list[str] = field(default_factory=list)
    output_tail_bytes: bytearray = field(default_factory=bytearray)
    output_bytes: int = 0
    total_output_bytes: int = 0
    output_truncated: bool = False
    truncation_notice_printed: bool = False
    had_stream_output: bool = False
    output_line_count: int = 0


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

    async def on_stdout(self, text: str) -> None:
        self.runtime._record_stream_output(
            text,
            style=None,
            output_state=self.output_state,
            display_state=self.display_state,
            is_stderr=False,
        )

    async def on_stderr(self, text: str) -> None:
        self.runtime._record_stream_output(
            text,
            style="red",
            output_state=self.output_state,
            display_state=self.display_state,
            is_stderr=True,
        )

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
                    description=f"Run a shell command directly in {shell_name}.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Command string only - no shell executable prefix (correct: 'pwd', incorrect: 'bash -c pwd').",
                            }
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                ),
                SHELL_TOOL_SOURCE,
            )

    @property
    def tool(self) -> Tool | None:
        return self._tool

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

    def metadata(self, command: str | None) -> dict[str, Any]:
        """Build metadata for display when the shell tool is invoked."""
        info = self.runtime_info()
        working_dir = self.working_directory()

        return {
            "variant": "shell",
            "command": command,
            "shell_name": info.name,
            "shell_path": info.path,
            "shell_kind": info.kind,
            "shell_provider": info.provider,
            "working_dir": str(working_dir),
            "working_dir_display": format_relative_path(working_dir),
            "timeout_seconds": self._timeout_seconds,
            "warning_interval_seconds": self._warning_interval_seconds,
            "output_byte_limit": self._output_byte_limit,
            "streams_output": True,
            "returns_exit_code": True,
        }

    def _invalid_execute_result(self, message: str) -> CallToolResult:
        return _text_result(message, is_error=True)

    def _extract_command(self, arguments: dict[str, Any] | None) -> str:
        payload = coerce_tool_arguments(arguments)
        return coerce_required_string_argument(payload.get("command"), "command", strip=True)

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
        self._append_output_tail(output_blob, state)
        if state.output_truncated:
            return

        remaining = self._output_byte_limit - state.output_bytes
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
        tail_limit = self._truncated_tail_byte_limit()
        if len(output_blob) >= tail_limit:
            state.output_tail_bytes = bytearray(output_blob[-tail_limit:])
            return

        state.output_tail_bytes.extend(output_blob)
        overflow = len(state.output_tail_bytes) - tail_limit
        if overflow > 0:
            del state.output_tail_bytes[:overflow]

    def _truncated_tail_byte_limit(self) -> int:
        return max(self._output_byte_limit // 2, 1)

    def _truncated_head_byte_limit(self) -> int:
        return max(self._output_byte_limit - self._truncated_tail_byte_limit(), 1)

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
            estimated_tokens = int(self._output_byte_limit / TERMINAL_BYTES_PER_TOKEN)
            console.console.print(
                " ".join(
                    [
                        "▶ Shell to agent output reached",
                        f"{self._output_byte_limit} bytes",
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
        head_limit = self._truncated_head_byte_limit()
        head_blob = "".join(output_state.output_segments).encode("utf-8", errors="replace")[
            :head_limit
        ]
        tail_blob = bytes(output_state.output_tail_bytes)[-self._truncated_tail_byte_limit() :]

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
        defer_display_to_tool_result: bool,
        display_line_limit: int | None,
    ) -> _ShellRuntimeExecution:
        output_state = _ShellOutputState()
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
        """Execute a shell command and stream output to the console with timeout detection."""
        try:
            command = self._extract_command(arguments)
        except ValueError as exc:
            return self._invalid_execute_result(str(exc))
        self._logger.debug(
            f"Executing command with idle_timeout={self._timeout_seconds}s, warning_interval={self._warning_interval_seconds}s"
        )

        progress_context = progress_display.paused() if display_tools_enabled() else nullcontext()
        with progress_context:
            try:
                self._emit_progress_event(
                    action=ProgressAction.CALLING_TOOL,
                    tool_use_id=tool_use_id,
                    tool_event="start",
                )

                runtime_execution = await self._execute_shell_command(
                    command,
                    cwd=None,
                    env=None,
                    timeout=None,
                    defer_display_to_tool_result=defer_display_to_tool_result,
                    display_line_limit=self._output_display_lines,
                )
                result, completion_details = self._build_shell_result(
                    execution=runtime_execution.execution,
                    output_state=runtime_execution.output_state,
                )
                result = self._finalize_shell_result_display(
                    result,
                    shell_result=runtime_execution.execution.result,
                    output_state=runtime_execution.output_state,
                    display_state=runtime_execution.display_state,
                    tool_use_id=tool_use_id,
                    show_tool_call_id=show_tool_call_id,
                    defer_display_to_tool_result=defer_display_to_tool_result,
                )

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
        tool_event: str | None = None,
        progress: float | None = None,
        total: float | None = None,
        details: str | None = None,
        tool_state: str | None = None,
        tool_terminal: bool | None = None,
    ) -> None:
        """Emit shell tool lifecycle events for progress display when supported."""
        info = getattr(self._logger, "info", None)
        if not callable(info):
            return

        payload: dict[str, Any] = build_progress_payload(
            action=action,
            tool_name=EXECUTE_TOOL_NAME,
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
        )

        try:
            info("Shell tool lifecycle", data=payload)
        except TypeError:
            # Standard library loggers reject custom keyword arguments.
            return
        except Exception:
            return
