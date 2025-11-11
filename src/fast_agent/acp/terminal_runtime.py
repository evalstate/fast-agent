"""
ACPTerminalRuntime - Execute commands via ACP terminal support.

This runtime allows FastAgent to execute commands through the ACP client's terminal
capabilities when available (e.g., in Zed editor). This provides better integration
compared to local process execution.
"""

import asyncio
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, Tool

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.content_helpers import text_content

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)


class ACPTerminalRuntime:
    """
    Provides command execution through ACP terminal support.

    This runtime implements the "execute" tool by delegating to the ACP client's
    terminal capabilities. The flow is:
    1. terminal/create - Start command execution
    2. terminal/wait_for_exit - Wait for completion
    3. terminal/output - Retrieve output
    4. terminal/release - Clean up resources

    The client (e.g., Zed editor) handles displaying the terminal UI to the user.
    """

    def __init__(
        self,
        connection: "AgentSideConnection",
        session_id: str,
        activation_reason: str,
        logger_instance=None,
        timeout_seconds: int = 90,
    ):
        """
        Initialize the ACP terminal runtime.

        Args:
            connection: The ACP connection to use for terminal operations
            session_id: The ACP session ID for this runtime
            activation_reason: Human-readable reason for activation
            logger_instance: Optional logger instance
            timeout_seconds: Default timeout for command execution
        """
        self.connection = connection
        self.session_id = session_id
        self.activation_reason = activation_reason
        self.logger = logger_instance or logger
        self.timeout_seconds = timeout_seconds

        # Tool definition for LLM
        self._tool = Tool(
            name="execute",
            description="Run a shell command in the client's terminal. "
            "The client will display the terminal and handle execution. "
            "You will receive the output when the command completes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute. Do not include shell "
                        "prefix (bash -c, etc.) - just provide the command string.",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional array of command arguments (alternative to including in command string).",
                    },
                    "env": {
                        "type": "object",
                        "description": "Optional environment variables as key-value pairs.",
                        "additionalProperties": {"type": "string"},
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional absolute path for working directory.",
                    },
                    "outputByteLimit": {
                        "type": "integer",
                        "description": "Optional maximum bytes of output to retain (prevents unbounded buffers).",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        )

        self.logger.info(
            "ACPTerminalRuntime initialized",
            session_id=session_id,
            reason=activation_reason,
            timeout=timeout_seconds,
        )

    @property
    def tool(self) -> Tool:
        """Get the execute tool definition."""
        return self._tool

    async def execute(self, arguments: dict[str, Any]) -> CallToolResult:
        """
        Execute a command using ACP terminal support.

        Args:
            arguments: Tool arguments containing 'command' key

        Returns:
            CallToolResult with command output and exit status
        """
        # Validate arguments
        if not isinstance(arguments, dict):
            return CallToolResult(
                content=[text_content("Error: arguments must be a dict")],
                isError=True,
            )

        command = arguments.get("command")
        if not command or not isinstance(command, str):
            return CallToolResult(
                content=[
                    text_content(
                        "Error: 'command' argument is required and must be a string"
                    )
                ],
                isError=True,
            )

        self.logger.info(
            "Executing command via ACP terminal",
            session_id=self.session_id,
            command=command[:100],  # Log first 100 chars
        )

        terminal_id = None  # Will be set by client in terminal/create response

        try:
            # Step 1: Create terminal and start command execution
            # NOTE: Client creates and returns the terminal ID, we don't generate it
            self.logger.debug("Creating terminal")

            # Build create params per ACP spec (sessionId, command, args, env, cwd, outputByteLimit)
            # Extract optional parameters from arguments
            create_params: dict[str, Any] = {
                "sessionId": self.session_id,
                "command": command,
            }

            # Add optional parameters if provided
            if args := arguments.get("args"):
                create_params["args"] = args
            if env := arguments.get("env"):
                # Transform env from object format (LLM-friendly) to ACP array format
                # Input: {"PATH": "/usr/bin", "HOME": "/home/user"}
                # Output: [{"name": "PATH", "value": "/usr/bin"}, {"name": "HOME", "value": "/home/user"}]
                if isinstance(env, dict):
                    create_params["env"] = [
                        {"name": name, "value": value} for name, value in env.items()
                    ]
                else:
                    # If already in array format, pass through
                    create_params["env"] = env
            if cwd := arguments.get("cwd"):
                create_params["cwd"] = cwd
            if output_limit := arguments.get("outputByteLimit"):
                create_params["outputByteLimit"] = output_limit

            create_result = await self.connection._conn.send_request("terminal/create", create_params)
            terminal_id = create_result.get("terminalId")

            if not terminal_id:
                return CallToolResult(
                    content=[text_content("Error: Client did not return terminal ID")],
                    isError=True,
                )

            self.logger.debug(f"Terminal created with ID: {terminal_id}")

            # Step 2: Wait for command to complete (with timeout)
            self.logger.debug(f"Waiting for terminal {terminal_id} to exit")
            try:
                wait_params = {"sessionId": self.session_id, "terminalId": terminal_id}
                wait_result = await asyncio.wait_for(
                    self.connection._conn.send_request("terminal/wait_for_exit", wait_params),
                    timeout=self.timeout_seconds,
                )
                exit_code = wait_result.get("exitCode", -1)
                signal = wait_result.get("signal")
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Terminal {terminal_id} timed out after {self.timeout_seconds}s"
                )
                # Kill the terminal
                try:
                    kill_params = {"sessionId": self.session_id, "terminalId": terminal_id}
                    await self.connection._conn.send_request("terminal/kill", kill_params)
                except Exception as kill_error:
                    self.logger.error(f"Error killing terminal: {kill_error}")

                # Still try to get output
                output_params = {"sessionId": self.session_id, "terminalId": terminal_id}
                output_result = await self.connection._conn.send_request(
                    "terminal/output", output_params
                )
                output_text = output_result.get("output", "")

                # Release terminal
                await self._release_terminal(terminal_id)

                return CallToolResult(
                    content=[
                        text_content(
                            f"Command timed out after {self.timeout_seconds}s\n\n"
                            f"Output so far:\n{output_text}"
                        )
                    ],
                    isError=True,
                )

            # Step 3: Get the output
            self.logger.debug(f"Retrieving output from terminal {terminal_id}")
            output_params = {"sessionId": self.session_id, "terminalId": terminal_id}
            output_result = await self.connection._conn.send_request("terminal/output", output_params)
            output_text = output_result.get("output", "")
            truncated = output_result.get("truncated", False)

            # Step 4: Release the terminal
            await self._release_terminal(terminal_id)

            # Format result
            is_error = exit_code != 0
            result_text = output_text

            if truncated:
                result_text = f"[Output truncated]\n{result_text}"

            if signal:
                result_text = f"{result_text}\n\n[Terminated by signal: {signal}]"

            result_text = f"{result_text}\n\n[Exit code: {exit_code}]"

            self.logger.info(
                "Terminal execution completed",
                terminal_id=terminal_id,
                exit_code=exit_code,
                output_length=len(output_text),
                truncated=truncated,
            )

            return CallToolResult(
                content=[text_content(result_text)],
                isError=is_error,
            )

        except Exception as e:
            self.logger.error(
                f"Error executing terminal command: {e}",
                terminal_id=terminal_id,
                exc_info=True,
            )
            # Try to clean up if we have a terminal ID
            if terminal_id:
                try:
                    await self._release_terminal(terminal_id)
                except Exception:
                    pass  # Best effort cleanup

            return CallToolResult(
                content=[text_content(f"Terminal execution error: {e}")],
                isError=True,
            )

    async def _release_terminal(self, terminal_id: str) -> None:
        """
        Release a terminal (cleanup).

        Args:
            terminal_id: The terminal ID to release
        """
        try:
            self.logger.debug(f"Releasing terminal {terminal_id}")
            release_params = {"sessionId": self.session_id, "terminalId": terminal_id}
            await self.connection._conn.send_request("terminal/release", release_params)
        except Exception as e:
            self.logger.error(f"Error releasing terminal {terminal_id}: {e}")

    def metadata(self) -> dict[str, Any]:
        """
        Get metadata about this runtime for display/logging.

        Returns:
            Dict with runtime information
        """
        return {
            "type": "acp_terminal",
            "session_id": self.session_id,
            "activation_reason": self.activation_reason,
            "timeout_seconds": self.timeout_seconds,
        }
