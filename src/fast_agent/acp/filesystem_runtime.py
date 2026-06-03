"""
ACPFilesystemRuntime - Read and write text files via ACP filesystem support.

This runtime allows FastAgent to read and write files through the ACP client's filesystem
capabilities when available (e.g., in Zed editor). This provides better integration and
security compared to direct file system access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from acp.helpers import tool_diff_content
from acp.schema import ToolCallProgress
from mcp.types import CallToolResult, Tool

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.tools.filesystem_tool_args import (
    parse_read_text_file_arguments,
    parse_write_text_file_arguments,
)
from fast_agent.tools.filesystem_tool_definitions import (
    READ_TEXT_FILE_TOOL_NAME,
    WRITE_TEXT_FILE_TOOL_NAME,
    build_read_text_file_tool,
    build_write_text_file_tool,
)
from fast_agent.tools.filesystem_tool_specs import (
    FilesystemToolSpec,
    enabled_tool_spec,
    enabled_tool_specs,
)
from fast_agent.tools.tool_sources import ACP_FILESYSTEM_TOOL_SOURCE, set_tool_source

if TYPE_CHECKING:
    from acp import AgentSideConnection
    from acp.schema import ReadTextFileResponse

    from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
    from fast_agent.mcp.tool_permission_handler import ToolPermissionHandler

logger = get_logger(__name__)

_PERMISSION_ACTION_LOG_LABELS = {
    "reading": "read",
    "writing": "write",
}


def _error_result(message: str) -> CallToolResult:
    return CallToolResult(content=[text_content(message)], isError=True)


def _fatal_error_result(message: str) -> CallToolResult:
    result = _error_result(message)
    setattr(result, "_fast_agent_fatal_tool_error", message)
    return result


def _success_result(message: str) -> CallToolResult:
    return CallToolResult(content=[text_content(message)], isError=False)


def _write_display_arguments(path: str, content: str) -> dict[str, Any]:
    return {
        "path": path,
        "content_length": len(content),
    }


class ACPFilesystemRuntime:
    """
    Provides file reading and writing through ACP filesystem support.

    This runtime implements the "read_text_file" and "write_text_file" tools by delegating
    to the ACP client's filesystem capabilities. The client (e.g., Zed editor) handles
    file access and permissions, providing a secure sandboxed environment.
    """

    def __init__(
        self,
        connection: "AgentSideConnection",
        session_id: str,
        activation_reason: str,
        logger_instance=None,
        enable_read: bool = True,
        enable_write: bool = True,
        tool_handler: "ToolExecutionHandler | None" = None,
        permission_handler: "ToolPermissionHandler | None" = None,
    ):
        """
        Initialize the ACP filesystem runtime.

        Args:
            connection: The ACP connection to use for filesystem operations
            session_id: The ACP session ID for this runtime
            activation_reason: Human-readable reason for activation
            logger_instance: Optional logger instance
            enable_read: Whether to enable the read_text_file tool
            enable_write: Whether to enable the write_text_file tool
            tool_handler: Optional tool execution handler for telemetry
            permission_handler: Optional permission handler for tool execution authorization
        """
        self.connection = connection
        self.session_id = session_id
        self.activation_reason = activation_reason
        self.logger = logger_instance or logger
        self._enable_read = enable_read
        self._enable_write = enable_write
        self._tool_handler = tool_handler
        self._permission_handler = permission_handler

        self._read_tool = set_tool_source(build_read_text_file_tool(), ACP_FILESYSTEM_TOOL_SOURCE)
        self._write_tool = set_tool_source(
            build_write_text_file_tool(), ACP_FILESYSTEM_TOOL_SOURCE
        )
        self._tool_specs: tuple[FilesystemToolSpec, ...] = (
            FilesystemToolSpec(
                name=READ_TEXT_FILE_TOOL_NAME,
                enabled=lambda: self._enable_read,
                tool=lambda: self._read_tool,
                handler=self.read_text_file,
            ),
            FilesystemToolSpec(
                name=WRITE_TEXT_FILE_TOOL_NAME,
                enabled=lambda: self._enable_write,
                tool=lambda: self._write_tool,
                handler=self.write_text_file,
            ),
        )

        self.logger.info(
            "ACPFilesystemRuntime initialized",
            session_id=session_id,
            reason=activation_reason,
        )

    @property
    def read_tool(self) -> Tool:
        """Get the read_text_file tool definition."""
        return self._read_tool

    @property
    def write_tool(self) -> Tool:
        """Get the write_text_file tool definition."""
        return self._write_tool

    @property
    def tools(self) -> list[Tool]:
        """Get all enabled filesystem tools."""
        return [spec.tool() for spec in self._enabled_tool_specs()]

    def _enabled_tool_specs(self) -> tuple[FilesystemToolSpec, ...]:
        return enabled_tool_specs(self._tool_specs)

    def _enabled_tool_spec(self, tool_name: str) -> FilesystemToolSpec | None:
        return enabled_tool_spec(self._tool_specs, tool_name)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params=None,
    ) -> CallToolResult:
        del request_params

        payload = arguments if arguments is not None else {}
        spec = self._enabled_tool_spec(name)
        if spec is not None:
            return await spec.handler(payload, tool_use_id)

        return _fatal_error_result(f"Error: unsupported ACP filesystem tool '{name}'.")

    async def read_text_file(
        self, arguments: dict[str, Any] | None, tool_use_id: str | None = None
    ) -> CallToolResult:
        """
        Read a text file using ACP filesystem support.

        Args:
            arguments: Tool arguments containing 'path' and optionally 'line' and 'limit'
            tool_use_id: LLM's tool use ID (for matching with stream events)

        Returns:
            CallToolResult with file contents
        """
        try:
            parsed = parse_read_text_file_arguments(arguments)
        except ValueError as exc:
            return _error_result(str(exc))

        self.logger.info(
            "Reading file via ACP filesystem",
            session_id=self.session_id,
            path=parsed.path,
        )

        tool_call_id = (
            await self._ensure_tool_call(READ_TEXT_FILE_TOOL_NAME, parsed.payload, tool_use_id)
            if tool_use_id
            else None
        )

        denied_result = await self._permission_denied_result(
            tool_name=READ_TEXT_FILE_TOOL_NAME,
            action="reading",
            path=parsed.path,
            arguments=parsed.payload,
            tool_use_id=tool_use_id,
            tool_call_id=tool_call_id,
        )
        if denied_result is not None:
            return denied_result

        if tool_call_id is None:
            tool_call_id = await self._ensure_tool_call(
                READ_TEXT_FILE_TOOL_NAME, parsed.payload, tool_use_id
            )

        try:
            # Send request using the proper ACP method with flattened parameters
            response: ReadTextFileResponse = await self.connection.read_text_file(
                path=parsed.path,
                session_id=self.session_id,
                line=parsed.line,
                limit=parsed.limit,
            )
            content = response.content

            self.logger.info(
                "File read completed",
                session_id=self.session_id,
                path=parsed.path,
                content_length=len(content),
            )

            result = _success_result(content)
            await self._notify_tool_complete(tool_call_id, True, result.content, None)

            return result

        except Exception as e:
            self.logger.error(
                f"Error reading file: {e}",
                session_id=self.session_id,
                path=parsed.path,
                exc_info=True,
            )

            await self._notify_tool_complete(tool_call_id, False, None, str(e))

            return _error_result(f"Error reading file: {e}")

    async def write_text_file(
        self, arguments: dict[str, Any] | None, tool_use_id: str | None = None
    ) -> CallToolResult:
        """
        Write a text file using ACP filesystem support.

        Args:
            arguments: Tool arguments containing 'path' and 'content'
            tool_use_id: LLM's tool use ID (for matching with stream events)

        Returns:
            CallToolResult indicating success or failure
        """
        try:
            parsed = parse_write_text_file_arguments(arguments)
        except ValueError as exc:
            return _error_result(str(exc))

        self.logger.info(
            "Writing file via ACP filesystem",
            session_id=self.session_id,
            path=parsed.path,
            content_length=len(parsed.content),
        )

        display_arguments = _write_display_arguments(parsed.path, parsed.content)
        tool_call_id = (
            await self._ensure_tool_call(
                WRITE_TEXT_FILE_TOOL_NAME,
                display_arguments,
                tool_use_id,
            )
            if tool_use_id
            else None
        )

        await self._send_write_diff_update(
            tool_call_id=tool_call_id,
            path=parsed.path,
            content=parsed.content,
            old_text=None,
        )

        denied_result = await self._permission_denied_result(
            tool_name=WRITE_TEXT_FILE_TOOL_NAME,
            action="writing",
            path=parsed.path,
            arguments=display_arguments,
            tool_use_id=tool_use_id,
            tool_call_id=tool_call_id,
        )
        if denied_result is not None:
            return denied_result

        if tool_call_id is None:
            tool_call_id = await self._ensure_tool_call(
                WRITE_TEXT_FILE_TOOL_NAME, display_arguments, tool_use_id
            )

        old_text = await self._read_existing_text(parsed.path)

        await self._send_write_diff_update(
            tool_call_id=tool_call_id,
            path=parsed.path,
            content=parsed.content,
            old_text=old_text,
        )

        try:
            # Send request using the proper ACP method with flattened parameters
            await self.connection.write_text_file(
                content=parsed.content,
                path=parsed.path,
                session_id=self.session_id,
            )

            self.logger.info(
                "File write completed",
                session_id=self.session_id,
                path=parsed.path,
            )

            result = _success_result(
                f"Successfully wrote {len(parsed.content)} characters to {parsed.path}"
            )

            # Pass None for content to preserve the diff content we already sent.
            await self._notify_tool_complete(tool_call_id, True, None, None)

            return result

        except Exception as e:
            self.logger.error(
                f"Error writing file: {e}",
                session_id=self.session_id,
                path=parsed.path,
                exc_info=True,
            )

            await self._notify_tool_complete(tool_call_id, False, None, str(e))

            return _error_result(f"Error writing file: {e}")

    def metadata(self) -> dict[str, Any]:
        """
        Get metadata about this runtime for display/logging.

        Returns:
            Dict with runtime information
        """
        return {
            "type": ACP_FILESYSTEM_TOOL_SOURCE,
            "session_id": self.session_id,
            "activation_reason": self.activation_reason,
            "tools": [spec.name for spec in self._enabled_tool_specs()],
        }

    async def _permission_error(
        self,
        *,
        tool_name: str,
        action: str,
        path: str,
        arguments: dict[str, Any],
        tool_use_id: str | None,
    ) -> str | None:
        if self._permission_handler is None:
            return None

        try:
            permission_result = await self._permission_handler.check_permission(
                tool_name=tool_name,
                server_name=ACP_FILESYSTEM_TOOL_SOURCE,
                arguments=arguments,
                tool_use_id=tool_use_id,
            )
        except Exception as e:
            self.logger.error(f"Error checking file {action} permission: {e}", exc_info=True)
            return f"Permission check failed: {e}"

        if permission_result.allowed:
            return None

        self.logger.info(
            f"File {_PERMISSION_ACTION_LOG_LABELS.get(action, action)} denied by permission handler",
            data={
                "path": path,
                "cancelled": permission_result.is_cancelled,
            },
        )
        return permission_result.error_message or f"Permission denied for {action} file: {path}"

    async def _permission_denied_result(
        self,
        *,
        tool_name: str,
        action: str,
        path: str,
        arguments: dict[str, Any],
        tool_use_id: str | None,
        tool_call_id: str | None,
    ) -> CallToolResult | None:
        permission_error = await self._permission_error(
            tool_name=tool_name,
            action=action,
            path=path,
            arguments=arguments,
            tool_use_id=tool_use_id,
        )
        if permission_error is None:
            return None
        await self._notify_permission_denied(
            tool_name=tool_name,
            arguments=arguments,
            tool_use_id=tool_use_id,
            tool_call_id=tool_call_id,
            error_msg=permission_error,
        )
        return _error_result(permission_error)

    async def _ensure_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_use_id: str | None,
    ) -> str | None:
        if self._tool_handler is None:
            return None
        try:
            if tool_use_id:
                return await self._tool_handler.ensure_tool_call_exists(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    server_name=ACP_FILESYSTEM_TOOL_SOURCE,
                    arguments=arguments,
                )
            return await self._tool_handler.on_tool_start(
                tool_name, ACP_FILESYSTEM_TOOL_SOURCE, arguments, tool_use_id
            )
        except Exception as e:
            self.logger.error(f"Error ensuring tool call for {tool_name}: {e}", exc_info=True)
            return None

    async def _notify_tool_complete(
        self,
        tool_call_id: str | None,
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

    async def _read_existing_text(self, path: str) -> str | None:
        if not self._enable_read:
            return None
        try:
            response = await self.connection.read_text_file(
                path=path,
                session_id=self.session_id,
            )
        except Exception:
            return None
        return response.content

    async def _send_write_diff_update(
        self,
        *,
        tool_call_id: str | None,
        path: str,
        content: str,
        old_text: str | None,
    ) -> None:
        if not tool_call_id:
            return
        try:
            diff_content = tool_diff_content(
                path=path,
                new_text=content,
                old_text=old_text,
            )
            await self.connection.session_update(
                session_id=self.session_id,
                update=ToolCallProgress(
                    session_update="tool_call_update",
                    tool_call_id=tool_call_id,
                    content=[diff_content],
                ),
            )
        except Exception as e:
            self.logger.error(f"Error sending write diff update: {e}", exc_info=True)

    async def _notify_permission_denied(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        tool_use_id: str | None,
        tool_call_id: str | None,
        error_msg: str,
    ) -> None:
        if not tool_use_id or self._tool_handler is None:
            return
        try:
            if tool_call_id is None:
                await self._tool_handler.ensure_tool_call_exists(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    server_name=ACP_FILESYSTEM_TOOL_SOURCE,
                    arguments=arguments,
                )
            await self._tool_handler.on_tool_permission_denied(
                tool_name,
                ACP_FILESYSTEM_TOOL_SOURCE,
                tool_use_id,
                error_msg,
            )
        except Exception as e:
            self.logger.error(
                f"Error notifying file permission denial for {tool_name}: {e}",
                exc_info=True,
            )
