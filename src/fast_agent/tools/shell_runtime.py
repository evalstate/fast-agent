from __future__ import annotations

import asyncio
import os
import platform
import shutil
import signal
import subprocess
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, TextContent, Tool
from rich.text import Text

if TYPE_CHECKING:
    from fast_agent.config import Settings
# Import tool progress context for reporting shell execution progress
from fast_agent.agents.tool_agent import _tool_progress_context
from fast_agent.constants import (
    DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
    MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
    TERMINAL_BYTES_PER_TOKEN,
)
from fast_agent.core.logging.progress_payloads import build_progress_payload
from fast_agent.event_progress import ProgressAction
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.display_suppression import display_tools_enabled
from fast_agent.ui.progress_display import progress_display
from fast_agent.ui.shell_output_truncation import (
    SHELL_OUTPUT_TRUNCATION_MARKER,
    split_shell_output_line_limit,
)
from fast_agent.utils.async_utils import gather_with_cancel

_STREAM_READ_CHUNK_SIZE = 4096
_MAX_PENDING_STREAM_BYTES = 65536


@dataclass(frozen=True, slots=True)
class _ShellProcessPlan:
    working_dir: Path
    shell_name: str
    shell_path: str | None
    is_windows: bool
    process_kwargs: dict[str, Any]


@dataclass(slots=True)
class _ShellOutputState:
    output_segments: list[str] = field(default_factory=list)
    output_bytes: int = 0
    total_output_bytes: int = 0
    output_truncated: bool = False
    truncation_notice_printed: bool = False
    had_stream_output: bool = False
    output_line_count: int = 0
    last_output_time: float = field(default_factory=time.time)
    timeout_occurred: bool = False


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
    display_tail_buffer: deque[tuple[int, str, str | None]] = field(
        default_factory=lambda: deque(maxlen=1)
    )


class ShellRuntime:
    """Helper for managing the optional local shell execute tool."""

    def __init__(
        self,
        activation_reason: str | None,
        logger,
        timeout_seconds: int = 90,
        warning_interval_seconds: int = 30,
        skills_directory: Path | None = None,
        working_directory: Path | None = None,
        output_byte_limit: int | None = None,
        config: Settings | None = None,
        agent_name: str | None = None,
    ) -> None:
        self._activation_reason = activation_reason
        self._logger = logger
        self._timeout_seconds = timeout_seconds
        self._warning_interval_seconds = warning_interval_seconds
        self._skills_directory = skills_directory
        self._working_directory = working_directory
        self._output_byte_limit = DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
        self.set_output_byte_limit(output_byte_limit)
        self.enabled: bool = activation_reason is not None
        self._tool: Tool | None = None
        self._display = ConsoleDisplay(config=config)
        self._agent_name = agent_name
        self._output_display_lines: int | None = None
        self._show_bash_output = True
        if config is not None:
            shell_config = getattr(config, "shell_execution", None)
            if shell_config is not None:
                self._output_display_lines = getattr(shell_config, "output_display_lines", None)
                self._show_bash_output = bool(getattr(shell_config, "show_bash", True))

        if self.enabled:
            # Detect the shell early so we can include it in the tool description
            runtime_info = self.runtime_info()
            shell_name = runtime_info.get("name", "shell")

            self._tool = Tool(
                name="execute",
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
            )

    @property
    def tool(self) -> Tool | None:
        return self._tool

    @property
    def output_byte_limit(self) -> int:
        """Return the current byte limit used to retain command output."""
        return self._output_byte_limit

    def set_output_byte_limit(self, output_byte_limit: int | None) -> None:
        """Set output retention byte limit, honoring global defaults and hard cap."""
        resolved_limit = output_byte_limit or DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
        self._output_byte_limit = min(resolved_limit, MAX_TERMINAL_OUTPUT_BYTE_LIMIT)

    def announce(self) -> None:
        """Inform the user why the local shell tool is active."""
        if not self.enabled or not self._activation_reason:
            return

        message = f"Local shell execute tool enabled {self._activation_reason}."
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
        if self._working_directory is not None:
            return self._working_directory
        # Skills now show their location relative to cwd in the system prompt
        return Path.cwd()

    @staticmethod
    def _resolve_working_directory(path: Path) -> Path:
        """Resolve working directory to an absolute path for subprocess execution."""
        if path.is_absolute():
            return path.resolve()
        return (Path.cwd() / path).resolve()

    def _validate_working_directory(self, configured_path: Path) -> str | None:
        """Return an actionable validation error when cwd is missing/invalid."""
        resolved_path = self._resolve_working_directory(configured_path)

        if not resolved_path.exists():
            return " ".join(
                [
                    f"Shell working directory does not exist: {resolved_path}.",
                    f"Configured cwd: {configured_path}.",
                    "Check the agent card 'cwd' setting or create the directory.",
                ]
            )

        if not resolved_path.is_dir():
            return " ".join(
                [
                    f"Shell working directory is not a directory: {resolved_path}.",
                    f"Configured cwd: {configured_path}.",
                    "Check the agent card 'cwd' setting.",
                ]
            )

        return None

    def runtime_info(self) -> dict[str, str | None]:
        """Best-effort detection of the shell runtime used for local execution.

        Uses modern Python APIs (platform.system(), shutil.which()) to detect
        and prefer modern shells like pwsh (PowerShell 7+) and bash.
        """
        system = platform.system()

        if system == "Windows":
            # Preference order: pwsh > powershell > cmd
            for shell_name in ["pwsh", "powershell", "cmd"]:
                shell_path = shutil.which(shell_name)
                if shell_path:
                    return {"name": shell_name, "path": shell_path}

            # Fallback to COMSPEC if nothing found in PATH
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            return {"name": Path(comspec).name, "path": comspec}
        else:
            # Unix-like: check SHELL env, then search for common shells
            shell_env = os.environ.get("SHELL")
            if shell_env and Path(shell_env).exists():
                return {"name": Path(shell_env).name, "path": shell_env}

            # Preference order: bash > zsh > sh
            for shell_name in ["bash", "zsh", "sh"]:
                shell_path = shutil.which(shell_name)
                if shell_path:
                    return {"name": shell_name, "path": shell_path}

            # Fallback to generic sh
            return {"name": "sh", "path": None}

    def metadata(self, command: str | None) -> dict[str, Any]:
        """Build metadata for display when the shell tool is invoked."""
        info = self.runtime_info()
        working_dir = self.working_directory()
        try:
            working_dir_display = str(working_dir.relative_to(Path.cwd()))
        except ValueError:
            working_dir_display = str(working_dir)

        return {
            "variant": "shell",
            "command": command,
            "shell_name": info.get("name"),
            "shell_path": info.get("path"),
            "working_dir": str(working_dir),
            "working_dir_display": working_dir_display,
            "timeout_seconds": self._timeout_seconds,
            "warning_interval_seconds": self._warning_interval_seconds,
            "output_byte_limit": self._output_byte_limit,
            "streams_output": True,
            "returns_exit_code": True,
        }

    def _invalid_execute_result(self, message: str) -> CallToolResult:
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=message)],
        )

    def _extract_command(self, arguments: dict[str, Any] | None) -> str | None:
        command_value = (arguments or {}).get("command") if arguments else None
        if not isinstance(command_value, str) or not command_value.strip():
            return None
        return command_value.strip()

    def _build_process_plan(self, configured_working_dir: Path) -> _ShellProcessPlan:
        working_dir = self._resolve_working_directory(configured_working_dir)
        runtime_details = self.runtime_info()
        shell_name = str(runtime_details.get("name") or "").lower()
        shell_path = runtime_details.get("path")
        is_windows = platform.system() == "Windows"
        process_kwargs: dict[str, Any] = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "cwd": working_dir,
        }
        if is_windows:
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if creation_flags:
                process_kwargs["creationflags"] = creation_flags
        else:
            process_kwargs["start_new_session"] = True
        return _ShellProcessPlan(
            working_dir=working_dir,
            shell_name=shell_name,
            shell_path=shell_path,
            is_windows=is_windows,
            process_kwargs=process_kwargs,
        )

    async def _start_shell_process(
        self,
        command: str,
        plan: _ShellProcessPlan,
    ) -> asyncio.subprocess.Process:
        if plan.is_windows and plan.shell_path and plan.shell_name in {"pwsh", "powershell"}:
            return await asyncio.create_subprocess_exec(
                plan.shell_path,
                "-NoLogo",
                "-NoProfile",
                "-Command",
                command,
                **plan.process_kwargs,
            )

        process_kwargs = dict(plan.process_kwargs)
        if plan.shell_path:
            process_kwargs["executable"] = plan.shell_path
        return await asyncio.create_subprocess_shell(command, **process_kwargs)

    def _build_display_state(
        self,
        *,
        defer_display_to_tool_result: bool,
    ) -> _ShellDisplayState:
        use_live_shell_display = (
            self._show_bash_output
            and not defer_display_to_tool_result
            and display_tools_enabled()
        )
        display_line_limit = self._output_display_lines
        state = _ShellDisplayState(
            use_live_shell_display=use_live_shell_display,
            display_line_limit=display_line_limit,
        )
        if display_line_limit is not None and display_line_limit > 0:
            display_head_limit, display_tail_limit = split_shell_output_line_limit(
                display_line_limit
            )
            state.display_head_limit = display_head_limit
            state.display_tail_limit = display_tail_limit
            state.display_tail_buffer = deque(maxlen=max(display_tail_limit, 1))
        return state

    def _append_output_text(self, output_text: str, state: _ShellOutputState) -> None:
        output_blob = output_text.encode("utf-8", errors="replace")
        state.total_output_bytes += len(output_blob)
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
        output_state.last_output_time = time.time()

    async def _stream_process_output(
        self,
        stream: asyncio.StreamReader | None,
        *,
        style: str | None,
        output_state: _ShellOutputState,
        display_state: _ShellDisplayState,
        is_stderr: bool = False,
    ) -> None:
        if stream is None:
            return

        pending = bytearray()

        while True:
            chunk = await stream.read(_STREAM_READ_CHUNK_SIZE)
            if not chunk:
                if pending:
                    self._record_stream_output(
                        pending.decode(errors="replace"),
                        style=style,
                        output_state=output_state,
                        display_state=display_state,
                        is_stderr=is_stderr,
                    )
                break
            pending.extend(chunk)

            while pending:
                newline_index = pending.find(b"\n")
                if newline_index >= 0:
                    line = bytes(pending[: newline_index + 1])
                    del pending[: newline_index + 1]
                    self._record_stream_output(
                        line.decode(errors="replace"),
                        style=style,
                        output_state=output_state,
                        display_state=display_state,
                        is_stderr=is_stderr,
                    )
                    continue

                if len(pending) < _MAX_PENDING_STREAM_BYTES:
                    break

                line = bytes(pending[:_MAX_PENDING_STREAM_BYTES])
                del pending[:_MAX_PENDING_STREAM_BYTES]
                self._record_stream_output(
                    line.decode(errors="replace"),
                    style=style,
                    output_state=output_state,
                    display_state=display_state,
                    is_stderr=is_stderr,
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

    async def _terminate_timed_out_process(
        self,
        process: asyncio.subprocess.Process,
        *,
        is_windows: bool,
    ) -> None:
        try:
            if is_windows:
                await self._terminate_windows_process(process)
            else:
                await self._terminate_unix_process(process)
        except (ProcessLookupError, OSError):
            return
        except Exception as exc:
            self._logger.debug(f"Error terminating process: {exc}")
            try:
                process.kill()
            except Exception:
                return

    async def _terminate_windows_process(self, process: asyncio.subprocess.Process) -> None:
        try:
            ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
            if ctrl_break is not None:
                process.send_signal(ctrl_break)
            await asyncio.sleep(2)
        except AttributeError:
            self._logger.debug("Watchdog: CTRL_BREAK_EVENT unsupported, skipping")
        except ValueError:
            self._logger.debug("Watchdog: no console attached for CTRL_BREAK_EVENT")
        except ProcessLookupError:
            return

        if process.returncode is None:
            process.terminate()
            await asyncio.sleep(2)
        if process.returncode is None:
            process.kill()

    async def _terminate_unix_process(self, process: asyncio.subprocess.Process) -> None:
        os.killpg(process.pid, signal.SIGTERM)
        await asyncio.sleep(2)
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)

    async def _watch_process_timeout(
        self,
        process: asyncio.subprocess.Process,
        *,
        tool_use_id: str | None,
        is_windows: bool,
        output_state: _ShellOutputState,
        display_state: _ShellDisplayState,
    ) -> None:
        last_warning_time = 0.0
        self._logger.debug(
            f"Watchdog started: timeout={self._timeout_seconds}s, warning_interval={self._warning_interval_seconds}s"
        )

        while True:
            await asyncio.sleep(1)
            if process.returncode is not None:
                self._logger.debug("Watchdog: process exited normally")
                return

            elapsed = time.time() - output_state.last_output_time
            remaining = self._timeout_seconds - elapsed
            time_since_warning = elapsed - last_warning_time
            if time_since_warning >= self._warning_interval_seconds and remaining > 0:
                self._logger.debug(f"Watchdog: warning at {int(remaining)}s remaining")
                if display_state.use_live_shell_display:
                    console.console.print(
                        f"▶ No output detected - terminating in {int(remaining)}s",
                        style="black on red",
                    )
                await self._emit_watchdog_progress(elapsed)
                last_warning_time = elapsed

            if elapsed < self._timeout_seconds:
                continue

            output_state.timeout_occurred = True
            self._logger.debug("Watchdog: timeout exceeded, terminating process group")
            if display_state.use_live_shell_display:
                console.console.print(
                    "▶ Timeout exceeded - terminating process",
                    style="black on red",
                )
            await self._terminate_timed_out_process(process, is_windows=is_windows)
            return

    async def _cancel_task_if_running(self, task: asyncio.Task[None] | None) -> None:
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return

    async def _wait_for_process_exit(
        self,
        process: asyncio.subprocess.Process,
        *,
        is_windows: bool,
    ) -> int:
        try:
            return await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            try:
                if is_windows:
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                return await process.wait()
            except Exception:
                return -1

    def _truncation_summary(self, output_state: _ShellOutputState) -> str | None:
        if not output_state.output_truncated:
            return None
        retained_tokens = max(int(output_state.output_bytes / TERMINAL_BYTES_PER_TOKEN), 1)
        total_tokens = max(int(output_state.total_output_bytes / TERMINAL_BYTES_PER_TOKEN), 1)
        omitted_bytes = max(output_state.total_output_bytes - output_state.output_bytes, 0)
        return (
            "[Output truncated: retained "
            f"{output_state.output_bytes} of {output_state.total_output_bytes} bytes "
            f"(~{retained_tokens} of ~{total_tokens} tokens); "
            f"omitted {omitted_bytes} bytes. "
            "Increase shell_execution.output_byte_limit to retain more.]"
        )

    def _build_shell_result(
        self,
        *,
        return_code: int,
        output_state: _ShellOutputState,
    ) -> tuple[CallToolResult, str]:
        combined_output = "".join(output_state.output_segments)
        if combined_output and not combined_output.endswith("\n"):
            combined_output += "\n"

        truncation_summary = self._truncation_summary(output_state)
        if truncation_summary:
            combined_output += f"{truncation_summary}\n"

        if output_state.timeout_occurred:
            combined_output += f"(timeout after {self._timeout_seconds}s - process terminated)"
            return (
                CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text=combined_output)],
                ),
                f"failed (timeout after {self._timeout_seconds}s)",
            )

        combined_output += f"process exit code was {return_code}"
        completion_state = "completed" if return_code == 0 else "failed"
        return (
            CallToolResult(
                isError=return_code != 0,
                content=[TextContent(type="text", text=combined_output)],
            ),
            f"{completion_state} (exit {return_code})",
        )

    def _flush_live_display_tail(self, display_state: _ShellDisplayState) -> None:
        if (
            not display_state.use_live_shell_display
            or display_state.display_line_limit is None
            or display_state.display_line_limit <= 0
            or not display_state.display_overflowed
        ):
            return
        if not display_state.display_ellipsis_printed:
            console.console.print(
                SHELL_OUTPUT_TRUNCATION_MARKER,
                style="dim",
                markup=False,
            )
        for buffered_index, buffered_text, buffered_style in display_state.display_tail_buffer:
            if buffered_index <= display_state.display_line_limit:
                continue
            console.console.print(
                self._render_display_line(buffered_text, buffered_style),
                markup=False,
            )

    def _finalize_shell_result_display(
        self,
        result: CallToolResult,
        *,
        return_code: int,
        output_state: _ShellOutputState,
        display_state: _ShellDisplayState,
        tool_use_id: str | None,
        show_tool_call_id: bool,
        defer_display_to_tool_result: bool,
    ) -> CallToolResult:
        self._flush_live_display_tail(display_state)
        if display_state.use_live_shell_display:
            self._display.show_shell_exit_code(
                return_code,
                no_output=not output_state.had_stream_output,
                output_line_count=output_state.output_line_count if output_state.had_stream_output else None,
                tool_call_id=tool_use_id if show_tool_call_id else None,
            )

        suppress_display = True
        if defer_display_to_tool_result and self._show_bash_output:
            suppress_display = False
        setattr(result, "_suppress_display", suppress_display)
        setattr(result, "exit_code", return_code)
        setattr(result, "output_line_count", output_state.output_line_count)
        return result

    async def execute(
        self,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        show_tool_call_id: bool = False,
        defer_display_to_tool_result: bool = False,
    ) -> CallToolResult:
        """Execute a shell command and stream output to the console with timeout detection."""
        command = self._extract_command(arguments)
        if command is None:
            return self._invalid_execute_result(
                "The execute tool requires a 'command' string argument."
            )
        self._logger.debug(
            f"Executing command with timeout={self._timeout_seconds}s, warning_interval={self._warning_interval_seconds}s"
        )

        configured_working_dir = self.working_directory()
        working_dir_error = self._validate_working_directory(configured_working_dir)
        if working_dir_error:
            return self._invalid_execute_result(working_dir_error)

        progress_context = progress_display.paused() if display_tools_enabled() else nullcontext()
        with progress_context:
            try:
                self._emit_progress_event(
                    action=ProgressAction.CALLING_TOOL,
                    tool_use_id=tool_use_id,
                    tool_event="start",
                )

                plan = self._build_process_plan(configured_working_dir)
                process = await self._start_shell_process(command, plan)
                output_state = _ShellOutputState()
                display_state = self._build_display_state(
                    defer_display_to_tool_result=defer_display_to_tool_result
                )

                stdout_task = asyncio.create_task(
                    self._stream_process_output(
                        process.stdout,
                        style=None,
                        output_state=output_state,
                        display_state=display_state,
                    )
                )
                stderr_task = asyncio.create_task(
                    self._stream_process_output(
                        process.stderr,
                        style="red",
                        output_state=output_state,
                        display_state=display_state,
                        is_stderr=True,
                    )
                )
                watchdog_task = asyncio.create_task(
                    self._watch_process_timeout(
                        process,
                        tool_use_id=tool_use_id,
                        is_windows=plan.is_windows,
                        output_state=output_state,
                        display_state=display_state,
                    )
                )

                await gather_with_cancel([stdout_task, stderr_task])
                await self._cancel_task_if_running(watchdog_task)
                return_code = await self._wait_for_process_exit(
                    process,
                    is_windows=plan.is_windows,
                )
                result, completion_details = self._build_shell_result(
                    return_code=return_code,
                    output_state=output_state,
                )
                result = self._finalize_shell_result_display(
                    result,
                    return_code=return_code,
                    output_state=output_state,
                    display_state=display_state,
                    tool_use_id=tool_use_id,
                    show_tool_call_id=show_tool_call_id,
                    defer_display_to_tool_result=defer_display_to_tool_result,
                )

                self._emit_progress_event(
                    action=ProgressAction.TOOL_PROGRESS,
                    tool_use_id=tool_use_id,
                    progress=1.0,
                    total=1.0,
                    details=completion_details,
                )
                return result

            except Exception as exc:
                self._logger.error(f"Execute tool failed: {exc}")
                self._emit_progress_event(
                    action=ProgressAction.TOOL_PROGRESS,
                    tool_use_id=tool_use_id,
                    progress=1.0,
                    total=1.0,
                    details=f"failed: {exc}",
                )
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text=f"Command failed to start: {exc}")],
                )

    def _emit_progress_event(
        self,
        *,
        action: ProgressAction,
        tool_use_id: str | None,
        tool_event: str | None = None,
        progress: float | None = None,
        total: float | None = None,
        details: str | None = None,
    ) -> None:
        """Emit shell tool lifecycle events for progress display when supported."""
        info = getattr(self._logger, "info", None)
        if not callable(info):
            return

        payload: dict[str, Any] = build_progress_payload(
            action=action,
            tool_name="execute",
            server_name="local",
            agent_name=self._agent_name,
            tool_use_id=tool_use_id,
            tool_call_id=tool_use_id,
            tool_event=tool_event,
            progress=progress,
            total=total,
            details=details,
        )

        try:
            info("Local shell tool lifecycle", data=payload)
        except TypeError:
            # Standard library loggers reject custom keyword arguments.
            return
        except Exception:
            return
