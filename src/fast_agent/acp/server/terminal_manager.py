"""
TerminalManager - Manages terminal execution for ACP protocol.

This handles the lifecycle of background command execution following the ACP terminal specification:
- Creates terminals to run commands in the background
- Captures and buffers output for retrieval
- Tracks exit status
- Supports kill and release operations
"""

import asyncio
import os
import platform
import shutil
import signal
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TerminalState:
    """State of a single terminal."""

    terminal_id: str
    command: str
    args: list[str]
    env: Dict[str, str]
    cwd: Path
    output_byte_limit: int
    process: asyncio.subprocess.Process | None = None
    output_buffer: list[str] = field(default_factory=list)
    truncated: bool = False
    exit_code: int | None = None
    signal: int | None = None
    _output_task: asyncio.Task | None = None
    _wait_task: asyncio.Task | None = None


class TerminalManager:
    """Manages terminal execution for ACP protocol."""

    def __init__(self) -> None:
        self.terminals: Dict[str, TerminalState] = {}
        self._lock = asyncio.Lock()

    def _runtime_info(self) -> Dict[str, str | None]:
        """Detect the shell runtime (similar to ShellRuntime)."""
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

    async def create_terminal(
        self,
        command: str,
        args: list[str] | None = None,
        env: Dict[str, str] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
    ) -> str:
        """
        Create a new terminal and start executing the command in the background.

        Args:
            command: The executable name (e.g., "ls", "python")
            args: Command-line arguments
            env: Environment variables (name/value pairs)
            cwd: Working directory path
            output_byte_limit: Maximum retained output bytes (default: 10MB)

        Returns:
            terminal_id: Unique identifier for the terminal
        """
        terminal_id = str(uuid.uuid4())
        args = args or []
        env_dict = env or {}
        cwd_path = Path(cwd) if cwd else Path.cwd()
        output_byte_limit = output_byte_limit or 10 * 1024 * 1024  # 10MB default

        logger.info(
            "Creating terminal",
            name="acp_terminal_create",
            terminal_id=terminal_id,
            command=command,
            args=args,
            cwd=str(cwd_path),
        )

        # Create terminal state
        terminal = TerminalState(
            terminal_id=terminal_id,
            command=command,
            args=args,
            env=env_dict,
            cwd=cwd_path,
            output_byte_limit=output_byte_limit,
        )

        # Start the process
        try:
            runtime_details = self._runtime_info()
            is_windows = platform.system() == "Windows"

            # Build command line
            full_command = [command] + args

            # Merge environment variables
            process_env = os.environ.copy()
            process_env.update(env_dict)

            # Shared process kwargs
            process_kwargs: dict[str, Any] = {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                "cwd": cwd_path,
                "env": process_env,
            }

            if is_windows:
                # Windows: CREATE_NEW_PROCESS_GROUP allows killing process tree
                process_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                # Unix: start_new_session creates new process group
                process_kwargs["start_new_session"] = True

            # Create the subprocess
            terminal.process = await asyncio.create_subprocess_exec(
                *full_command,
                **process_kwargs,
            )

            # Start background tasks to capture output and wait for exit
            terminal._output_task = asyncio.create_task(
                self._capture_output(terminal_id, terminal)
            )
            terminal._wait_task = asyncio.create_task(self._wait_for_exit(terminal_id, terminal))

            # Store terminal
            async with self._lock:
                self.terminals[terminal_id] = terminal

            logger.info(
                "Terminal created successfully",
                name="acp_terminal_created",
                terminal_id=terminal_id,
            )

            return terminal_id

        except Exception as e:
            logger.error(
                f"Failed to create terminal: {e}",
                name="acp_terminal_create_error",
                exc_info=True,
            )
            raise

    async def _capture_output(self, terminal_id: str, terminal: TerminalState) -> None:
        """Background task to capture output from the process."""
        if not terminal.process or not terminal.process.stdout:
            return

        try:
            total_bytes = 0
            while True:
                line = await terminal.process.stdout.readline()
                if not line:
                    break

                text = line.decode(errors="replace")
                line_bytes = len(line)

                # Check byte limit
                if total_bytes + line_bytes > terminal.output_byte_limit:
                    terminal.truncated = True
                    logger.debug(
                        "Terminal output truncated",
                        name="acp_terminal_truncated",
                        terminal_id=terminal_id,
                    )
                    break

                terminal.output_buffer.append(text)
                total_bytes += line_bytes

        except Exception as e:
            logger.error(
                f"Error capturing terminal output: {e}",
                name="acp_terminal_output_error",
                terminal_id=terminal_id,
                exc_info=True,
            )

    async def _wait_for_exit(self, terminal_id: str, terminal: TerminalState) -> None:
        """Background task to wait for process to exit."""
        if not terminal.process:
            return

        try:
            exit_code = await terminal.process.wait()
            terminal.exit_code = exit_code

            logger.info(
                "Terminal process exited",
                name="acp_terminal_exited",
                terminal_id=terminal_id,
                exit_code=exit_code,
            )

        except Exception as e:
            logger.error(
                f"Error waiting for terminal exit: {e}",
                name="acp_terminal_wait_error",
                terminal_id=terminal_id,
                exc_info=True,
            )

    async def get_output(
        self, terminal_id: str
    ) -> tuple[str, bool, Dict[str, int | None] | None]:
        """
        Get current output from the terminal.

        Args:
            terminal_id: The terminal identifier

        Returns:
            tuple of (output, truncated, exit_status)
            - output: Captured text
            - truncated: Boolean indicating if output was truncated
            - exit_status: Dict with exitCode and signal (if completed), or None
        """
        async with self._lock:
            terminal = self.terminals.get(terminal_id)
            if not terminal:
                raise ValueError(f"Terminal {terminal_id} not found")

            output = "".join(terminal.output_buffer)
            truncated = terminal.truncated

            exit_status = None
            if terminal.exit_code is not None:
                exit_status = {
                    "exitCode": terminal.exit_code,
                    "signal": terminal.signal,
                }

            return output, truncated, exit_status

    async def wait_for_exit(self, terminal_id: str) -> tuple[int, int | None]:
        """
        Wait for the terminal process to exit.

        Args:
            terminal_id: The terminal identifier

        Returns:
            tuple of (exit_code, signal)
        """
        async with self._lock:
            terminal = self.terminals.get(terminal_id)
            if not terminal:
                raise ValueError(f"Terminal {terminal_id} not found")

        # If already exited, return immediately
        if terminal.exit_code is not None:
            return terminal.exit_code, terminal.signal

        # Wait for the wait task to complete
        if terminal._wait_task:
            await terminal._wait_task

        return terminal.exit_code or -1, terminal.signal

    async def kill_terminal(self, terminal_id: str) -> None:
        """
        Kill the terminal process.

        The terminal remains valid for output retrieval and exit status checking.

        Args:
            terminal_id: The terminal identifier
        """
        async with self._lock:
            terminal = self.terminals.get(terminal_id)
            if not terminal:
                raise ValueError(f"Terminal {terminal_id} not found")

            if not terminal.process:
                return

            # Check if already exited
            if terminal.exit_code is not None:
                logger.debug(
                    "Terminal already exited",
                    name="acp_terminal_already_exited",
                    terminal_id=terminal_id,
                )
                return

            try:
                is_windows = platform.system() == "Windows"

                if is_windows:
                    # Windows: send CTRL_BREAK signal, then terminate
                    try:
                        terminal.process.send_signal(signal.CTRL_BREAK_EVENT)
                        await asyncio.sleep(0.5)
                    except (AttributeError, ValueError, ProcessLookupError):
                        pass

                    if terminal.process.returncode is None:
                        terminal.process.terminate()
                        await asyncio.sleep(0.5)

                    if terminal.process.returncode is None:
                        terminal.process.kill()
                else:
                    # Unix: kill process group
                    try:
                        os.killpg(terminal.process.pid, signal.SIGTERM)
                        await asyncio.sleep(0.5)

                        if terminal.process.returncode is None:
                            os.killpg(terminal.process.pid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        # Process already exited
                        pass

                logger.info(
                    "Terminal killed",
                    name="acp_terminal_killed",
                    terminal_id=terminal_id,
                )

            except Exception as e:
                logger.error(
                    f"Error killing terminal: {e}",
                    name="acp_terminal_kill_error",
                    terminal_id=terminal_id,
                    exc_info=True,
                )

    async def release_terminal(self, terminal_id: str) -> None:
        """
        Release the terminal and clean up resources.

        This kills any running process and invalidates the terminal ID.

        Args:
            terminal_id: The terminal identifier
        """
        async with self._lock:
            terminal = self.terminals.get(terminal_id)
            if not terminal:
                raise ValueError(f"Terminal {terminal_id} not found")

            # Kill the process if still running
            if terminal.process and terminal.exit_code is None:
                await self.kill_terminal(terminal_id)

            # Cancel background tasks
            if terminal._output_task and not terminal._output_task.done():
                terminal._output_task.cancel()
                try:
                    await terminal._output_task
                except asyncio.CancelledError:
                    pass

            if terminal._wait_task and not terminal._wait_task.done():
                terminal._wait_task.cancel()
                try:
                    await terminal._wait_task
                except asyncio.CancelledError:
                    pass

            # Remove from terminals dict
            del self.terminals[terminal_id]

            logger.info(
                "Terminal released",
                name="acp_terminal_released",
                terminal_id=terminal_id,
            )

    async def cleanup_all(self) -> None:
        """Clean up all terminals."""
        terminal_ids = list(self.terminals.keys())
        for terminal_id in terminal_ids:
            try:
                await self.release_terminal(terminal_id)
            except Exception as e:
                logger.error(
                    f"Error releasing terminal {terminal_id}: {e}",
                    name="acp_terminal_cleanup_error",
                )
