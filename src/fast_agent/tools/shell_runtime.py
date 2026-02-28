from __future__ import annotations

import asyncio
import os
import platform
import shutil
import signal
import subprocess
import time
from collections import deque
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
from fast_agent.ui.progress_display import progress_display
from fast_agent.ui.shell_output_truncation import split_shell_output_line_limit
from fast_agent.utils.async_utils import gather_with_cancel


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

    async def execute(
        self,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        show_tool_call_id: bool = False,
        defer_display_to_tool_result: bool = False,
    ) -> CallToolResult:
        """Execute a shell command and stream output to the console with timeout detection."""
        command_value = (arguments or {}).get("command") if arguments else None
        if not isinstance(command_value, str) or not command_value.strip():
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text="The execute tool requires a 'command' string argument.",
                    )
                ],
            )

        command = command_value.strip()
        self._logger.debug(
            f"Executing command with timeout={self._timeout_seconds}s, warning_interval={self._warning_interval_seconds}s"
        )

        configured_working_dir = self.working_directory()
        working_dir_error = self._validate_working_directory(configured_working_dir)
        if working_dir_error:
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=working_dir_error)],
            )

        # Pause progress display during shell execution to avoid overlaying output
        with progress_display.paused():
            try:
                self._emit_progress_event(
                    action=ProgressAction.CALLING_TOOL,
                    tool_use_id=tool_use_id,
                    tool_event="start",
                )

                working_dir = self._resolve_working_directory(configured_working_dir)
                runtime_details = self.runtime_info()
                shell_name = (runtime_details.get("name") or "").lower()
                shell_path = runtime_details.get("path")

                # Detect platform for process group handling
                is_windows = platform.system() == "Windows"

                # Shared process kwargs
                process_kwargs: dict[str, Any] = {
                    "stdout": asyncio.subprocess.PIPE,
                    "stderr": asyncio.subprocess.PIPE,
                    "cwd": working_dir,
                }

                if is_windows:
                    # Windows: CREATE_NEW_PROCESS_GROUP allows killing process tree
                    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    if creation_flags:
                        process_kwargs["creationflags"] = creation_flags
                else:
                    # Unix: start_new_session creates new process group
                    process_kwargs["start_new_session"] = True

                # Create the subprocess, preferring PowerShell on Windows when available
                if is_windows and shell_path and shell_name in {"pwsh", "powershell"}:
                    process = await asyncio.create_subprocess_exec(
                        shell_path,
                        "-NoLogo",
                        "-NoProfile",
                        "-Command",
                        command,
                        **process_kwargs,
                    )
                else:
                    if shell_path:
                        process_kwargs["executable"] = shell_path
                    process = await asyncio.create_subprocess_shell(
                        command,
                        **process_kwargs,
                    )

                output_segments: list[str] = []
                output_bytes = 0
                total_output_bytes = 0
                output_truncated = False
                truncation_notice_printed = False
                had_stream_output = False
                use_live_shell_display = (
                    self._show_bash_output and not defer_display_to_tool_result
                )
                display_line_limit = self._output_display_lines
                displayed_head_count = 0
                display_total_line_count = 0
                display_overflowed = False
                display_ellipsis_printed = False
                display_head_limit, display_tail_limit = (0, 0)
                display_tail_buffer: deque[tuple[int, str, str | None]] = deque(maxlen=1)
                if display_line_limit is not None and display_line_limit > 0:
                    display_head_limit, display_tail_limit = split_shell_output_line_limit(
                        display_line_limit
                    )
                    display_tail_buffer = deque(
                        maxlen=max(display_tail_limit, 1),
                    )
                # Track last output time in a mutable container for sharing across coroutines
                last_output_time = [time.time()]
                timeout_occurred = [False]
                watchdog_task = None

                async def stream_output(stream, style: str | None, is_stderr: bool = False) -> None:
                    nonlocal output_bytes, total_output_bytes, output_truncated
                    nonlocal truncation_notice_printed
                    nonlocal displayed_head_count, display_total_line_count, display_overflowed
                    nonlocal display_ellipsis_printed
                    nonlocal had_stream_output
                    if not stream:
                        return
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        had_stream_output = True
                        text = line.decode(errors="replace")
                        output_text = text if not is_stderr else f"[stderr] {text}"
                        output_blob = output_text.encode("utf-8", errors="replace")
                        total_output_bytes += len(output_blob)
                        if not output_truncated:
                            remaining = self._output_byte_limit - output_bytes
                            if remaining > 0:
                                if len(output_blob) <= remaining:
                                    output_segments.append(output_text)
                                    output_bytes += len(output_blob)
                                else:
                                    truncated_text = output_blob[:remaining].decode(
                                        "utf-8", errors="replace"
                                    )
                                    if truncated_text:
                                        output_segments.append(truncated_text)
                                    output_bytes += remaining
                                    output_truncated = True
                            else:
                                output_truncated = True

                        if output_truncated and not truncation_notice_printed:
                            if use_live_shell_display and (
                                display_line_limit is None or display_line_limit > 0
                            ):
                                estimated_tokens = int(
                                    self._output_byte_limit / TERMINAL_BYTES_PER_TOKEN
                                )
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
                            truncation_notice_printed = True

                        if use_live_shell_display:
                            if display_line_limit is None:
                                console.console.print(
                                    self._render_display_line(text, style),
                                    markup=False,
                                )
                            elif display_line_limit <= 0:
                                pass
                            else:
                                display_total_line_count += 1
                                current_line_index = display_total_line_count
                                if displayed_head_count < display_head_limit:
                                    console.console.print(
                                        self._render_display_line(text, style),
                                        markup=False,
                                    )
                                    displayed_head_count += 1
                                else:
                                    if display_tail_limit > 0:
                                        display_tail_buffer.append((current_line_index, text, style))
                                    if current_line_index > display_line_limit:
                                        display_overflowed = True
                                        if not display_ellipsis_printed:
                                            console.console.print("...", style="dim", markup=False)
                                            display_ellipsis_printed = True

                        # Update last output time whenever we receive a line
                        last_output_time[0] = time.time()

                async def watchdog() -> None:
                    """Monitor output timeout and emit warnings."""
                    last_warning_time = 0.0
                    self._logger.debug(
                        f"Watchdog started: timeout={self._timeout_seconds}s, warning_interval={self._warning_interval_seconds}s"
                    )

                    while True:
                        await asyncio.sleep(1)  # Check every second

                        # Check if process has exited
                        if process.returncode is not None:
                            self._logger.debug("Watchdog: process exited normally")
                            break

                        elapsed = time.time() - last_output_time[0]
                        remaining = self._timeout_seconds - elapsed

                        # Emit warnings every warning_interval_seconds throughout execution
                        time_since_warning = elapsed - last_warning_time
                        if time_since_warning >= self._warning_interval_seconds and remaining > 0:
                            self._logger.debug(f"Watchdog: warning at {int(remaining)}s remaining")
                            if use_live_shell_display:
                                console.console.print(
                                    f"▶ No output detected - terminating in {int(remaining)}s",
                                    style="black on red",
                                )
                            # Report progress to parent agent if in tool context
                            ctx = _tool_progress_context.get()
                            if ctx:
                                handler, tool_call_id = ctx
                                try:
                                    await handler.on_tool_progress(
                                        tool_call_id,
                                        0.5,
                                        None,
                                        f"Waiting for output ({int(elapsed)}) seconds ...",
                                    )
                                except Exception:
                                    pass
                            last_warning_time = elapsed

                        # Timeout exceeded
                        if elapsed >= self._timeout_seconds:
                            timeout_occurred[0] = True
                            self._logger.debug(
                                "Watchdog: timeout exceeded, terminating process group"
                            )
                            if use_live_shell_display:
                                console.console.print(
                                    "▶ Timeout exceeded - terminating process", style="black on red"
                                )
                            try:
                                if is_windows:
                                    # Windows: try to signal the entire process group before terminating
                                    try:
                                        ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
                                        if ctrl_break is not None:
                                            process.send_signal(ctrl_break)
                                        await asyncio.sleep(2)
                                    except AttributeError:
                                        # Older Python/asyncio may not support send_signal on Windows
                                        self._logger.debug(
                                            "Watchdog: CTRL_BREAK_EVENT unsupported, skipping"
                                        )
                                    except ValueError:
                                        # Raised when no console is attached; fall back to terminate
                                        self._logger.debug(
                                            "Watchdog: no console attached for CTRL_BREAK_EVENT"
                                        )
                                    except ProcessLookupError:
                                        pass  # Process already exited

                                    if process.returncode is None:
                                        process.terminate()
                                        await asyncio.sleep(2)
                                    if process.returncode is None:
                                        process.kill()
                                else:
                                    # Unix: kill entire process group for clean cleanup
                                    os.killpg(process.pid, signal.SIGTERM)
                                    await asyncio.sleep(2)
                                    if process.returncode is None:
                                        os.killpg(process.pid, signal.SIGKILL)
                            except (ProcessLookupError, OSError):
                                pass  # Process already terminated
                            except Exception as e:
                                self._logger.debug(f"Error terminating process: {e}")
                                # Fallback: kill just the main process
                                try:
                                    process.kill()
                                except Exception:
                                    pass
                            break

                stdout_task = asyncio.create_task(stream_output(process.stdout, None))
                stderr_task = asyncio.create_task(stream_output(process.stderr, "red", True))
                watchdog_task = asyncio.create_task(watchdog())

                # Wait for streams to complete
                await gather_with_cancel([stdout_task, stderr_task])

                # Cancel watchdog if still running
                if watchdog_task and not watchdog_task.done():
                    watchdog_task.cancel()
                    try:
                        await watchdog_task
                    except asyncio.CancelledError:
                        pass

                # Wait for process to finish
                try:
                    return_code = await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    # Process didn't exit, force kill
                    try:
                        if is_windows:
                            # Windows: force kill main process
                            process.kill()
                        else:
                            # Unix: SIGKILL to process group
                            os.killpg(process.pid, signal.SIGKILL)
                        return_code = await process.wait()
                    except Exception:
                        return_code = -1

                # Build result based on timeout or normal completion
                truncation_summary: str | None = None
                if output_truncated:
                    retained_tokens = max(int(output_bytes / TERMINAL_BYTES_PER_TOKEN), 1)
                    total_tokens = max(int(total_output_bytes / TERMINAL_BYTES_PER_TOKEN), 1)
                    omitted_bytes = max(total_output_bytes - output_bytes, 0)
                    truncation_summary = (
                        "[Output truncated: retained "
                        f"{output_bytes} of {total_output_bytes} bytes "
                        f"(~{retained_tokens} of ~{total_tokens} tokens); "
                        f"omitted {omitted_bytes} bytes. "
                        "Increase shell_execution.output_byte_limit to retain more.]"
                    )

                if timeout_occurred[0]:
                    combined_output = "".join(output_segments)
                    if combined_output and not combined_output.endswith("\n"):
                        combined_output += "\n"
                    if truncation_summary:
                        combined_output += f"{truncation_summary}\n"
                    combined_output += (
                        f"(timeout after {self._timeout_seconds}s - process terminated)"
                    )

                    result = CallToolResult(
                        isError=True,
                        content=[
                            TextContent(
                                type="text",
                                text=combined_output,
                            )
                        ],
                    )
                    completion_details = f"failed (timeout after {self._timeout_seconds}s)"
                else:
                    combined_output = "".join(output_segments)
                    # Add explicit exit code message for the LLM
                    if combined_output and not combined_output.endswith("\n"):
                        combined_output += "\n"
                    if truncation_summary:
                        combined_output += f"{truncation_summary}\n"
                    combined_output += f"process exit code was {return_code}"

                    result = CallToolResult(
                        isError=return_code != 0,
                        content=[
                            TextContent(
                                type="text",
                                text=combined_output,
                            )
                        ],
                    )
                    completion_state = "completed" if return_code == 0 else "failed"
                    completion_details = f"{completion_state} (exit {return_code})"

                if use_live_shell_display and display_line_limit is not None and display_line_limit > 0:
                    if display_overflowed:
                        if not display_ellipsis_printed:
                            console.console.print("...", style="dim", markup=False)
                        for buffered_index, buffered_text, buffered_style in display_tail_buffer:
                            if buffered_index <= display_line_limit:
                                continue
                            console.console.print(
                                self._render_display_line(buffered_text, buffered_style),
                                markup=False,
                            )

                # Display bottom separator with exit code
                if use_live_shell_display:
                    self._display.show_shell_exit_code(
                        return_code,
                        no_output=not had_stream_output,
                        tool_call_id=tool_use_id if show_tool_call_id else None,
                    )

                suppress_display = True
                if defer_display_to_tool_result and self._show_bash_output:
                    suppress_display = False
                setattr(result, "_suppress_display", suppress_display)
                setattr(result, "exit_code", return_code)

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
