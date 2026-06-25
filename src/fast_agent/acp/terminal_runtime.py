"""
ACPTerminalRuntime - Execute commands via ACP terminal support.

This runtime allows FastAgent to execute commands through the ACP client's terminal
capabilities when available (e.g., in Zed editor). This provides better integration
compared to local process execution.
"""

from __future__ import annotations

import asyncio
import os
import re
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, Tool

from fast_agent.constants import DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT, TERMINAL_BYTES_PER_TOKEN
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.tools.filesystem_tool_args import (
    coerce_optional_string_argument,
    coerce_required_string_argument,
    coerce_tool_arguments,
)
from fast_agent.tools.tool_sources import ACP_TERMINAL_TOOL_SOURCE, set_tool_source
from fast_agent.utils.commandline import split_commandline
from fast_agent.utils.tool_names import EXECUTE_TOOL_NAME

if TYPE_CHECKING:
    from acp import AgentSideConnection

    from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
    from fast_agent.mcp.tool_permission_handler import ToolPermissionHandler

logger = get_logger(__name__)

_SHELL_ONLY_CHARS = frozenset("|&;><()\n")
_SHELL_EXPANSION_CHARS = frozenset("$`*?[]{}")
_SHELL_BUILTINS = frozenset(
    {
        "alias",
        "bg",
        "cd",
        "command",
        "dirs",
        "eval",
        "exec",
        "export",
        "fg",
        "hash",
        "jobs",
        "popd",
        "pushd",
        "pwd",
        "read",
        "readonly",
        "set",
        "shift",
        "source",
        "test",
        "times",
        "trap",
        "type",
        "ulimit",
        "umask",
        "unalias",
        "unset",
        "wait",
    }
)
_SHELL_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*")


@dataclass(frozen=True, slots=True)
class _TerminalOutputResult:
    output: str
    truncated: bool


def _error_result(message: str) -> CallToolResult:
    return CallToolResult(content=[text_content(message)], isError=True)


def _needs_shell_wrapper(command: str) -> bool:
    quote: str | None = None
    escaped = False
    at_word_start = True

    for char in command:
        if escaped:
            escaped = False
            at_word_start = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            if quote == "'":
                at_word_start = False
                continue
            if char in {"$", "`"}:
                return True
            at_word_start = False
            continue
        if char in {"'", '"'}:
            quote = char
            at_word_start = False
            continue
        if char.isspace():
            at_word_start = True
            continue
        if char in _SHELL_ONLY_CHARS:
            return True
        if char in _SHELL_EXPANSION_CHARS:
            return True
        if char == "~" and at_word_start:
            return True
        at_word_start = False

    try:
        tokens = split_commandline(command)
    except ValueError:
        return True

    if not tokens:
        return False
    if tokens[0] in _SHELL_BUILTINS:
        return True
    return _is_shell_assignment(tokens[0])


def _is_shell_assignment(token: str) -> bool:
    return bool(_SHELL_ASSIGNMENT_RE.match(token))


def _wrap_shell_command(command: str) -> tuple[str, list[str]]:
    if os.name == "nt":
        shell = os.environ.get("COMSPEC", "cmd.exe").strip() or "cmd.exe"
        return shell, ["/d", "/s", "/c", command]
    return "/bin/sh", ["-lc", command]


def _resolve_terminal_command(
    command: str,
    args: list[str] | None,
) -> tuple[str, list[str]]:
    if args is not None:
        return command, list(args)

    if _needs_shell_wrapper(command):
        return _wrap_shell_command(command)

    try:
        tokens = split_commandline(command)
    except ValueError:
        return _wrap_shell_command(command)

    if not tokens:
        return command, []

    return tokens[0], tokens[1:]


def _coerce_command(arguments: dict[str, Any]) -> str:
    return coerce_required_string_argument(arguments.get("command"), "command", strip=True)


def _coerce_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return coerce_tool_arguments(arguments)


def _coerce_terminal_args(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Error: 'args' argument must be a list of strings")
    return value


def _coerce_terminal_cwd(value: Any) -> str | None:
    if value is None:
        return None
    cwd = coerce_optional_string_argument(value, "cwd", strip=True)
    if not cwd:
        raise ValueError("Error: 'cwd' argument must be a non-empty string")
    return cwd


def _terminal_env_params(env: Any) -> list[dict[str, str]]:
    if isinstance(env, dict):
        if not all(isinstance(name, str) and isinstance(value, str) for name, value in env.items()):
            raise ValueError("Error: 'env' argument must contain string keys and values")
        return [{"name": name, "value": value} for name, value in env.items()]

    if isinstance(env, list) and all(
        isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("value"), str)
        for item in env
    ):
        return [{"name": item["name"], "value": item["value"]} for item in env]

    raise ValueError("Error: 'env' argument must be an object with string keys and values")


def _coerce_terminal_output_result(output_result: Any) -> _TerminalOutputResult:
    if not isinstance(output_result, dict):
        return _TerminalOutputResult(output="", truncated=False)

    output = output_result.get("output", "")
    truncated = output_result.get("truncated", False)
    return _TerminalOutputResult(
        output=output if isinstance(output, str) else "",
        truncated=truncated is True,
    )


def _terminal_output_byte_limit(arguments: dict[str, Any], default_limit: int) -> int:
    output_byte_limit = arguments.get("outputByteLimit", default_limit)
    if (
        isinstance(output_byte_limit, bool)
        or not isinstance(output_byte_limit, int)
        or output_byte_limit <= 0
    ):
        return default_limit
    return output_byte_limit


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
        connection: AgentSideConnection,
        session_id: str,
        activation_reason: str,
        logger_instance=None,
        timeout_seconds: int = 90,
        tool_handler: "ToolExecutionHandler | None" = None,
        default_output_byte_limit: int = DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
        permission_handler: "ToolPermissionHandler | None" = None,
    ):
        """
        Initialize the ACP terminal runtime.

        Args:
            connection: The ACP connection to use for terminal operations
            session_id: The ACP session ID for this runtime
            activation_reason: Human-readable reason for activation
            logger_instance: Optional logger instance
            timeout_seconds: Default timeout for command execution
            tool_handler: Optional tool execution handler for telemetry
            permission_handler: Optional permission handler for tool execution authorization
        """
        self.connection = connection
        self.session_id = session_id
        self.activation_reason = activation_reason
        self.logger = logger_instance or logger
        self.timeout_seconds = timeout_seconds
        self._tool_handler = tool_handler
        self._default_output_byte_limit = (
            default_output_byte_limit or DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
        )
        self._permission_handler = permission_handler

        # Tool definition for LLM
        self._tool = set_tool_source(
            Tool(
                name=EXECUTE_TOOL_NAME,
                description="Execute a shell command.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute. For simple commands, provide "
                            "the executable here and use args for arguments. Shell operators "
                            "(pipes, redirects, &&, etc.) may be included and will be run via "
                            "a shell wrapper automatically. Do not include your own shell prefix "
                            "(bash -c, etc.).",
                        },
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional array of command arguments. When provided, "
                            "command is treated as the executable path/name and args are passed "
                            "through directly.",
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
                        # Do not allow model to handle this for the moment.
                        # "outputByteLimit": {
                        #     "type": "integer",
                        #     "description": "Maximum bytes of output to retain.  (prevents unbounded buffers).",
                        # },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            ),
            ACP_TERMINAL_TOOL_SOURCE,
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

    async def execute(
        self, arguments: dict[str, Any], tool_use_id: str | None = None
    ) -> CallToolResult:
        """
        Execute a command using ACP terminal support.

        Args:
            arguments: Tool arguments containing 'command' key
            tool_use_id: LLM's tool use ID (for matching with stream events)

        Returns:
            CallToolResult with command output and exit status
        """
        try:
            arguments = _coerce_arguments(arguments)
            command = _coerce_command(arguments)
            create_params = self._build_create_params(command, arguments)
        except ValueError as exc:
            return _error_result(str(exc))

        self.logger.info(
            "Executing command via ACP terminal",
            session_id=self.session_id,
            command=command[:100],  # Log first 100 chars
        )

        permission_error = await self._permission_error(command, arguments, tool_use_id)
        if permission_error:
            await self._notify_permission_denied(arguments, tool_use_id, permission_error)
            return _error_result(permission_error)

        tool_call_id = await self._notify_tool_start(arguments, tool_use_id)

        terminal_id = None  # Will be set by client in terminal/create response

        try:
            output_byte_limit = create_params["outputByteLimit"]
            terminal_id = await self._create_terminal(command, create_params)
            if not terminal_id:
                error = "Error: Client did not return terminal ID"
                await self._notify_tool_complete(
                    tool_call_id,
                    success=False,
                    content=None,
                    error=error,
                )
                return _error_result(error)

            try:
                exit_code, signal = await self._wait_for_exit(terminal_id)
            except asyncio.TimeoutError:
                return await self._handle_timeout(
                    terminal_id=terminal_id,
                    tool_call_id=tool_call_id,
                )

            output_result = await self._terminal_output(terminal_id)
            terminal_output = _coerce_terminal_output_result(output_result)

            await self._release_terminal(terminal_id)
            is_error = exit_code != 0
            result_text = self._format_result_text(
                output_text=terminal_output.output,
                output_byte_limit=output_byte_limit,
                truncated=terminal_output.truncated,
                signal=signal,
                exit_code=exit_code,
            )

            self.logger.info(
                "Terminal execution completed",
                terminal_id=terminal_id,
                exit_code=exit_code,
                output_length=len(terminal_output.output),
                truncated=terminal_output.truncated,
            )

            result = CallToolResult(
                content=[text_content(result_text)],
                isError=is_error,
            )

            await self._notify_tool_complete(
                tool_call_id,
                success=not is_error,
                content=result.content if not is_error else None,
                error=result_text if is_error else None,
            )
            return result

        except Exception as e:
            self.logger.error(
                f"Error executing terminal command: {e}",
                terminal_id=terminal_id,
                exc_info=True,
            )
            # Try to clean up if we have a terminal ID
            if terminal_id:
                with suppress(Exception):
                    await self._release_terminal(terminal_id)

            await self._notify_tool_complete(
                tool_call_id, success=False, content=None, error=str(e)
            )
            return _error_result(f"Terminal execution error: {e}")

    async def _permission_error(
        self, command: str, arguments: dict[str, Any], tool_use_id: str | None
    ) -> str | None:
        if self._permission_handler is None:
            return None
        try:
            permission_result = await self._permission_handler.check_permission(
                tool_name=EXECUTE_TOOL_NAME,
                server_name=ACP_TERMINAL_TOOL_SOURCE,
                arguments=arguments,
                tool_use_id=tool_use_id,
            )
        except Exception as e:
            self.logger.error(f"Error checking terminal permission: {e}", exc_info=True)
            return f"Permission check failed: {e}"

        if permission_result.allowed:
            return None

        self.logger.info(
            "Terminal execution denied by permission handler",
            data={
                "command": command[:100],
                "cancelled": permission_result.is_cancelled,
            },
        )
        return permission_result.error_message or "Permission denied for terminal execution"

    async def _notify_tool_start(
        self, arguments: dict[str, Any], tool_use_id: str | None
    ) -> str | None:
        if self._tool_handler is None:
            return None
        try:
            return await self._tool_handler.on_tool_start(
                EXECUTE_TOOL_NAME, ACP_TERMINAL_TOOL_SOURCE, arguments, tool_use_id
            )
        except Exception as e:
            self.logger.error(f"Error in tool start handler: {e}", exc_info=True)
            return None

    async def _notify_permission_denied(
        self,
        arguments: dict[str, Any],
        tool_use_id: str | None,
        error_msg: str,
    ) -> None:
        if not tool_use_id or self._tool_handler is None:
            return
        try:
            await self._tool_handler.ensure_tool_call_exists(
                tool_use_id=tool_use_id,
                tool_name=EXECUTE_TOOL_NAME,
                server_name=ACP_TERMINAL_TOOL_SOURCE,
                arguments=arguments,
            )
            await self._tool_handler.on_tool_permission_denied(
                EXECUTE_TOOL_NAME,
                ACP_TERMINAL_TOOL_SOURCE,
                tool_use_id,
                error_msg,
            )
        except Exception as e:
            self.logger.error(
                f"Error notifying terminal permission denial: {e}",
                exc_info=True,
            )

    async def _notify_tool_complete(
        self,
        tool_call_id: str | None,
        *,
        success: bool,
        content: list[Any] | None,
        error: str | None,
    ) -> None:
        if self._tool_handler is None or not tool_call_id:
            return
        try:
            await self._tool_handler.on_tool_complete(tool_call_id, success, content, error)
        except Exception as e:
            self.logger.error(f"Error in tool complete handler: {e}", exc_info=True)

    def _build_create_params(self, command: str, arguments: dict[str, Any]) -> dict[str, Any]:
        resolved_command, resolved_args = _resolve_terminal_command(
            command,
            _coerce_terminal_args(arguments.get("args")),
        )
        create_params: dict[str, Any] = {
            "sessionId": self.session_id,
            "command": resolved_command,
        }
        if resolved_args:
            create_params["args"] = resolved_args
        if "env" in arguments and arguments["env"] is not None:
            create_params["env"] = _terminal_env_params(arguments["env"])
        cwd = _coerce_terminal_cwd(arguments.get("cwd"))
        if cwd is not None:
            create_params["cwd"] = cwd

        create_params["outputByteLimit"] = _terminal_output_byte_limit(
            arguments,
            self._default_output_byte_limit,
        )
        return create_params

    async def _create_terminal(self, command: str, create_params: dict[str, Any]) -> str | None:
        self.logger.debug("Creating terminal")
        create_result = await self.connection._conn.send_request("terminal/create", create_params)
        terminal_id = create_result.get("terminalId")
        if not terminal_id:
            self.logger.error(
                "terminal/create did not return terminalId",
                data={
                    "session_id": self.session_id,
                    "command": command,
                    "create_result": create_result,
                },
            )
            return None
        self.logger.debug(f"Terminal created with ID: {terminal_id}")
        return terminal_id

    async def _wait_for_exit(self, terminal_id: str) -> tuple[int, Any]:
        self.logger.debug(f"Waiting for terminal {terminal_id} to exit")
        wait_params = {"sessionId": self.session_id, "terminalId": terminal_id}
        wait_result = await asyncio.wait_for(
            self.connection._conn.send_request("terminal/wait_for_exit", wait_params),
            timeout=self.timeout_seconds,
        )
        return wait_result.get("exitCode", -1), wait_result.get("signal")

    async def _terminal_output(self, terminal_id: str) -> dict[str, Any]:
        self.logger.debug(f"Retrieving output from terminal {terminal_id}")
        output_params = {"sessionId": self.session_id, "terminalId": terminal_id}
        return await self.connection._conn.send_request("terminal/output", output_params)

    async def _handle_timeout(
        self, *, terminal_id: str, tool_call_id: str | None
    ) -> CallToolResult:
        self.logger.warning(f"Terminal {terminal_id} timed out after {self.timeout_seconds}s")
        try:
            kill_params = {"sessionId": self.session_id, "terminalId": terminal_id}
            await self.connection._conn.send_request("terminal/kill", kill_params)
        except Exception as kill_error:
            self.logger.error(f"Error killing terminal: {kill_error}")

        output_result = await self._terminal_output(terminal_id)
        terminal_output = _coerce_terminal_output_result(output_result)
        await self._release_terminal(terminal_id)

        await self._notify_tool_complete(
            tool_call_id,
            success=False,
            content=None,
            error=f"Command timed out after {self.timeout_seconds}s",
        )
        return _error_result(
            f"Command timed out after {self.timeout_seconds}s\n\n"
            f"Output so far:\n{terminal_output.output}"
        )

    @staticmethod
    def _format_result_text(
        *,
        output_text: str,
        output_byte_limit: int,
        truncated: bool,
        signal: Any,
        exit_code: int,
    ) -> str:
        result_text = output_text
        if truncated:
            estimated_tokens = max(int(output_byte_limit / TERMINAL_BYTES_PER_TOKEN), 1)
            result_text = "\n".join(
                [
                    "[Output truncated by ACP terminal outputByteLimit: "
                    f"{output_byte_limit} bytes (~{estimated_tokens} tokens). "
                    "Client returned partial output only.]",
                    result_text,
                ]
            )
        if signal:
            result_text = f"{result_text}\n\n[Terminated by signal: {signal}]"
        return f"{result_text}\n\n[Exit code: {exit_code}]"

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
            "type": ACP_TERMINAL_TOOL_SOURCE,
            "session_id": self.session_id,
            "activation_reason": self.activation_reason,
            "timeout_seconds": self.timeout_seconds,
        }
