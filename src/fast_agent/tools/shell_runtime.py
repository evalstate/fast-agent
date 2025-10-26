from __future__ import annotations

import asyncio
import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.types import CallToolResult, TextContent, Tool

from fast_agent.ui import console
from fast_agent.ui.progress_display import progress_display


class ShellRuntime:
    """Helper for managing the optional local shell execute tool."""

    def __init__(
        self,
        activation_reason: str | None,
        logger,
        timeout_seconds: int = 90,
        warning_interval_seconds: int = 30,
        skills_directory: Path | None = None,
    ) -> None:
        self._activation_reason = activation_reason
        self._logger = logger
        self._timeout_seconds = timeout_seconds
        self._warning_interval_seconds = warning_interval_seconds
        self._skills_directory = skills_directory
        self.enabled: bool = activation_reason is not None
        self._tool: Tool | None = None

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

    def announce(self) -> None:
        """Inform the user why the local shell tool is active."""
        if not self.enabled or not self._activation_reason:
            return

        message = f"Local shell execute tool enabled {self._activation_reason}."
        self._logger.info(message)

    def working_directory(self) -> Path:
        """Return the working directory used for shell execution."""
        # TODO -- reinstate when we provide duplication/isolation of skill workspaces
        if self._skills_directory and self._skills_directory.exists():
            return self._skills_directory
        return Path.cwd()

    def runtime_info(self) -> Dict[str, str | None]:
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

    def metadata(self, command: Optional[str]) -> Dict[str, Any]:
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
            "streams_output": True,
            "returns_exit_code": True,
        }

    async def execute(self, arguments: Dict[str, Any] | None = None) -> CallToolResult:
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

        # Pause progress display during shell execution to avoid overlaying output
        with progress_display.paused():
            try:
                working_dir = self.working_directory()
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
                    process_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
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
                # Track last output time in a mutable container for sharing across coroutines
                last_output_time = [time.time()]
                timeout_occurred = [False]
                watchdog_task = None

                async def stream_output(
                    stream, style: Optional[str], is_stderr: bool = False
                ) -> None:
                    if not stream:
                        return
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        text = line.decode(errors="replace")
                        output_segments.append(text if not is_stderr else f"[stderr] {text}")
                        console.console.print(
                            text.rstrip("\n"),
                            style=style,
                            markup=False,
                        )
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
                            console.console.print(
                                f"▶ No output detected - terminating in {int(remaining)}s",
                                style="black on red",
                            )
                            last_warning_time = elapsed

                        # Timeout exceeded
                        if elapsed >= self._timeout_seconds:
                            timeout_occurred[0] = True
                            self._logger.debug(
                                "Watchdog: timeout exceeded, terminating process group"
                            )
                            console.console.print(
                                "▶ Timeout exceeded - terminating process", style="black on red"
                            )
                            try:
                                if is_windows:
                                    # Windows: try to signal the entire process group before terminating
                                    try:
                                        process.send_signal(signal.CTRL_BREAK_EVENT)
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
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

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
                if timeout_occurred[0]:
                    combined_output = "".join(output_segments)
                    if combined_output and not combined_output.endswith("\n"):
                        combined_output += "\n"
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
                else:
                    combined_output = "".join(output_segments)
                    # Add explicit exit code message for the LLM
                    if combined_output and not combined_output.endswith("\n"):
                        combined_output += "\n"
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

                # Display bottom separator with exit code
                try:
                    from rich.text import Text
                except Exception:  # pragma: no cover
                    Text = None  # type: ignore[assignment]

                if Text:
                    # Build bottom separator matching the style: ─| exit code 0 |─────────
                    width = console.console.size.width
                    exit_code_style = "red" if return_code != 0 else "dim"
                    exit_code_text = f"exit code {return_code}"

                    prefix = Text("─| ")
                    prefix.stylize("dim")
                    exit_text = Text(exit_code_text, style=exit_code_style)
                    suffix = Text(" |")
                    suffix.stylize("dim")

                    separator = Text()
                    separator.append_text(prefix)
                    separator.append_text(exit_text)
                    separator.append_text(suffix)
                    remaining = width - separator.cell_len
                    if remaining > 0:
                        separator.append("─" * remaining, style="dim")

                    console.console.print()
                    console.console.print(separator)
                else:
                    console.console.print(f"exit code {return_code}", style="dim")

                setattr(result, "_suppress_display", True)
                setattr(result, "exit_code", return_code)
                return result

            except Exception as exc:
                self._logger.error(f"Execute tool failed: {exc}")
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text=f"Command failed to start: {exc}")],
                )
