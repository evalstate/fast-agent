"""
ACPTerminalExecutor - Delegates command execution to ACP clients via terminal protocol.

This executor uses the ACP terminal/* methods to run commands in the client's environment
(e.g., the user's editor or IDE), rather than running them locally in the agent process.
"""

import asyncio
import shlex
from typing import Any

from acp import AgentSideConnection
from acp.schema import (
    CreateTerminalRequest,
    KillTerminalCommandRequest,
    ReleaseTerminalRequest,
    TerminalOutputRequest,
    WaitForTerminalExitRequest,
)

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.schema.tool_schema import Tool

logger = get_logger(__name__)


class ACPTerminalExecutor:
    """
    Executes commands in the client environment using ACP terminal protocol.

    This class provides a similar interface to ShellRuntime but delegates execution
    to the ACP client (editor/IDE) instead of running commands locally.
    """

    def __init__(
        self,
        connection: AgentSideConnection,
        session_id: str,
    ):
        """
        Initialize the ACP terminal executor.

        Args:
            connection: The ACP connection to use for terminal operations
            session_id: The session ID for this executor
        """
        self.connection = connection
        self.session_id = session_id
        self.tool: Tool | None = None

        # Create the execute tool
        self.tool = Tool(
            name="execute",
            description=(
                "Execute a command in the client's terminal environment. "
                "This runs commands on the client side (in the user's editor/IDE environment), "
                "not in the agent's environment. Use this to run shell commands, scripts, "
                "build tools, tests, or any other command-line operations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the command (optional, absolute path)",
                    },
                },
                "required": ["command"],
            },
        )

        logger.info(
            "ACPTerminalExecutor initialized",
            name="acp_terminal_executor_init",
            session_id=session_id,
        )

    async def execute(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Execute a command via ACP terminal protocol.

        This method:
        1. Creates a terminal on the client side
        2. Waits for the command to complete
        3. Retrieves the output
        4. Releases the terminal
        5. Returns the result in MCP tool response format

        Args:
            arguments: Dictionary with 'command' (required) and 'cwd' (optional)

        Returns:
            MCP tool response with output and status
        """
        command_str = arguments.get("command", "")
        cwd = arguments.get("cwd")

        if not command_str:
            return [
                {
                    "type": "text",
                    "text": "Error: No command provided",
                    "isError": True,
                }
            ]

        logger.info(
            "Executing command via ACP terminal",
            name="acp_terminal_execute",
            session_id=self.session_id,
            command=command_str,
            cwd=cwd,
        )

        terminal_id = None

        try:
            # Parse command into command + args
            # For complex commands with pipes, redirects, etc., we pass as shell command
            try:
                # Try to parse as shell command
                parsed = shlex.split(command_str)
                if len(parsed) > 0:
                    command = parsed[0]
                    args = parsed[1:] if len(parsed) > 1 else None
                else:
                    # Empty command
                    return [
                        {
                            "type": "text",
                            "text": "Error: Empty command",
                            "isError": True,
                        }
                    ]
            except ValueError:
                # If parsing fails (e.g., unclosed quotes), treat entire string as command
                command = command_str
                args = None

            # Create terminal request
            create_request = CreateTerminalRequest(
                sessionId=self.session_id,
                command=command,
                args=args,
                cwd=cwd,
            )

            # Create the terminal
            logger.debug(
                "Creating terminal",
                name="acp_terminal_create",
                session_id=self.session_id,
                command=command,
                args=args,
            )

            handle = await self.connection.createTerminal(create_request)
            terminal_id = handle.terminalId

            logger.info(
                "Terminal created",
                name="acp_terminal_created",
                session_id=self.session_id,
                terminal_id=terminal_id,
            )

            # Wait for the terminal to complete
            wait_request = WaitForTerminalExitRequest(
                sessionId=self.session_id,
                terminalId=terminal_id,
            )

            logger.debug(
                "Waiting for terminal exit",
                name="acp_terminal_wait",
                session_id=self.session_id,
                terminal_id=terminal_id,
            )

            exit_response = await self.connection.waitForTerminalExit(wait_request)

            logger.info(
                "Terminal exited",
                name="acp_terminal_exited",
                session_id=self.session_id,
                terminal_id=terminal_id,
                exit_code=exit_response.exitCode,
                signal=exit_response.signal,
            )

            # Get the output
            output_request = TerminalOutputRequest(
                sessionId=self.session_id,
                terminalId=terminal_id,
            )

            output_response = await self.connection.terminalOutput(output_request)

            logger.info(
                "Retrieved terminal output",
                name="acp_terminal_output",
                session_id=self.session_id,
                terminal_id=terminal_id,
                output_length=len(output_response.output),
                truncated=output_response.truncated,
            )

            # Build result message
            output_text = output_response.output
            exit_code = exit_response.exitCode
            signal = exit_response.signal
            truncated = output_response.truncated

            # Construct result message
            result_parts = []

            if output_text:
                result_parts.append(output_text)

            # Add exit status information
            if signal:
                result_parts.append(f"\n[Process terminated by signal: {signal}]")
            elif exit_code is not None:
                result_parts.append(f"\n[Exit code: {exit_code}]")

            if truncated:
                result_parts.append("\n[Output was truncated]")

            result_text = "".join(result_parts)

            # Determine if this is an error (non-zero exit code or signal)
            is_error = (exit_code is not None and exit_code != 0) or signal is not None

            return [
                {
                    "type": "text",
                    "text": result_text,
                    "isError": is_error,
                }
            ]

        except Exception as e:
            logger.error(
                f"Error executing command via ACP terminal: {e}",
                name="acp_terminal_error",
                session_id=self.session_id,
                terminal_id=terminal_id,
                exc_info=True,
            )

            return [
                {
                    "type": "text",
                    "text": f"Error executing command: {e}",
                    "isError": True,
                }
            ]

        finally:
            # Always release the terminal
            if terminal_id:
                try:
                    release_request = ReleaseTerminalRequest(
                        sessionId=self.session_id,
                        terminalId=terminal_id,
                    )

                    await self.connection.releaseTerminal(release_request)

                    logger.info(
                        "Terminal released",
                        name="acp_terminal_released",
                        session_id=self.session_id,
                        terminal_id=terminal_id,
                    )
                except Exception as e:
                    logger.error(
                        f"Error releasing terminal: {e}",
                        name="acp_terminal_release_error",
                        session_id=self.session_id,
                        terminal_id=terminal_id,
                    )

    async def kill_terminal(self, terminal_id: str) -> None:
        """
        Kill a running terminal.

        Args:
            terminal_id: The terminal ID to kill
        """
        try:
            kill_request = KillTerminalCommandRequest(
                sessionId=self.session_id,
                terminalId=terminal_id,
            )

            await self.connection.killTerminal(kill_request)

            logger.info(
                "Terminal killed",
                name="acp_terminal_killed",
                session_id=self.session_id,
                terminal_id=terminal_id,
            )
        except Exception as e:
            logger.error(
                f"Error killing terminal: {e}",
                name="acp_terminal_kill_error",
                session_id=self.session_id,
                terminal_id=terminal_id,
            )
