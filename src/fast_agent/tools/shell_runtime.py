from __future__ import annotations

import asyncio
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.types import CallToolResult, TextContent, Tool

from fast_agent.ui import console


class ShellRuntime:
    """Helper for managing the optional local shell execute tool."""

    def __init__(self, activation_reason: str | None, logger) -> None:
        self._activation_reason = activation_reason
        self._logger = logger
        self.enabled: bool = activation_reason is not None
        self._tool: Tool | None = None

        if self.enabled:
            # Detect the shell early so we can include it in the tool description
            runtime_info = self.runtime_info()
            shell_name = runtime_info.get("name", "shell")

            self._tool = Tool(
                name="execute",
                description=f"Run a shell command ({shell_name}) inside the agent workspace and return its output.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute (e.g. 'cat README.md').",
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
        skills_cwd = Path.cwd() / "fast-agent" / "skills"
        return skills_cwd if skills_cwd.exists() else Path.cwd()

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
            "streams_output": True,
            "returns_exit_code": True,
        }

    async def execute(self, arguments: Dict[str, Any] | None = None) -> CallToolResult:
        """Execute a shell command and stream output to the console."""
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
        try:
            from rich.text import Text
        except Exception:  # pragma: no cover - fallback if rich is unavailable
            Text = None  # type: ignore[assignment]

        try:
            if Text:
                command_line = Text("$ ", style="magenta")
                command_line.append(command, style="white")
                console.console.print(command_line)
            else:
                console.console.print(f"$ {command}", style="magenta", markup=False)

            working_dir = self.working_directory()

            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            output_segments: list[str] = []

            async def stream_output(stream, style: Optional[str], is_stderr: bool = False) -> None:
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

            stdout_task = asyncio.create_task(stream_output(process.stdout, None))
            stderr_task = asyncio.create_task(stream_output(process.stderr, "red", True))

            await asyncio.gather(stdout_task, stderr_task)
            return_code = await process.wait()

            if Text:
                status_line = Text("exit code ", style="dim")
                status_line.append(str(return_code), style="dim")
                console.console.print(status_line)
            else:
                console.console.print(f"exit code {return_code}", style="dim")

            combined_output = "".join(output_segments)
            if combined_output and not combined_output.endswith("\n"):
                combined_output += "\n"
            combined_output += f"(exit code: {return_code})"

            result = CallToolResult(
                isError=return_code != 0,
                content=[
                    TextContent(
                        type="text",
                        text=combined_output if combined_output else f"(exit code: {return_code})",
                    )
                ],
            )
            setattr(result, "_suppress_display", True)
            return result

        except Exception as exc:
            self._logger.error(f"Execute tool failed: {exc}")
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Command failed to start: {exc}")],
            )
