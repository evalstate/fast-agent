"""
ACPFilesystemRuntime - Read and write text files via ACP filesystem support.

This runtime allows FastAgent to read and write files through the ACP client's filesystem
capabilities when available (e.g., in Zed editor). This provides better integration and
security compared to direct file system access.
"""

from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, Tool

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.content_helpers import text_content

if TYPE_CHECKING:
    from acp import AgentSideConnection
    from acp.schema import ReadTextFileResponse

    from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
    from fast_agent.mcp.tool_permission_handler import ToolPermissionHandler

logger = get_logger(__name__)


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

        # Tool definition for reading text files
        self._read_tool = Tool(
            name="read_text_file",
            description="Read content from a text file. Returns the file contents as a string. ",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    },
                    "line": {
                        "type": "integer",
                        "description": "Optional line number to start reading from (1-based).",
                        "minimum": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum number of lines to read.",
                        "minimum": 1,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

        # Tool definition for writing text files
        self._write_tool = Tool(
            name="write_text_file",
            description="Write content to a text file. Creates or overwrites the file. ",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write to the file.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
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
        tools = []
        if self._enable_read:
            tools.append(self._read_tool)
        if self._enable_write:
            tools.append(self._write_tool)
        return tools

    async def read_text_file(
        self, arguments: dict[str, Any], tool_use_id: str | None = None
    ) -> CallToolResult:
        """
        Read a text file using ACP filesystem support.

        Args:
            arguments: Tool arguments containing 'path' and optionally 'line' and 'limit'
            tool_use_id: LLM's tool use ID (for matching with stream events)

        Returns:
            CallToolResult with file contents
        """
        # Validate arguments
        if not isinstance(arguments, dict):
            return CallToolResult(
                content=[text_content("Error: arguments must be a dict")],
                isError=True,
            )

        path = arguments.get("path")
        if not path or not isinstance(path, str):
            return CallToolResult(
                content=[text_content("Error: 'path' argument is required and must be a string")],
                isError=True,
            )

        self.logger.info(
            "Reading file via ACP filesystem",
            session_id=self.session_id,
            path=path,
        )

        # Check permission before execution
        if self._permission_handler:
            try:
                permission_result = await self._permission_handler.check_permission(
                    tool_name="read_text_file",
                    server_name="acp_filesystem",
                    arguments=arguments,
                    tool_use_id=tool_use_id,
                )
                if not permission_result.allowed:
                    error_msg = permission_result.error_message or (
                        f"Permission denied for reading file: {path}"
                    )
                    self.logger.info(
                        "File read denied by permission handler",
                        data={
                            "path": path,
                            "cancelled": permission_result.is_cancelled,
                        },
                    )
                    return CallToolResult(
                        content=[text_content(error_msg)],
                        isError=True,
                    )
            except Exception as e:
                self.logger.error(f"Error checking file read permission: {e}", exc_info=True)
                # Fail-safe: deny on permission check error
                return CallToolResult(
                    content=[text_content(f"Permission check failed: {e}")],
                    isError=True,
                )

        # Notify tool handler that execution is starting
        tool_call_id = None
        if self._tool_handler:
            try:
                tool_call_id = await self._tool_handler.on_tool_start(
                    "read_text_file", "acp_filesystem", arguments, tool_use_id
                )
            except Exception as e:
                self.logger.error(f"Error in tool start handler: {e}", exc_info=True)

        try:
            # Send request using the proper ACP method with flattened parameters
            response: ReadTextFileResponse = await self.connection.read_text_file(
                path=path,
                session_id=self.session_id,
                line=arguments.get("line"),
                limit=arguments.get("limit"),
            )
            content = response.content

            self.logger.info(
                "File read completed",
                session_id=self.session_id,
                path=path,
                content_length=len(content),
            )

            result = CallToolResult(
                content=[text_content(content)],
                isError=False,
            )

            # Notify tool handler of completion
            if self._tool_handler and tool_call_id:
                try:
                    await self._tool_handler.on_tool_complete(
                        tool_call_id, True, result.content, None
                    )
                except Exception as e:
                    self.logger.error(f"Error in tool complete handler: {e}", exc_info=True)

            return result

        except Exception as e:
            self.logger.error(
                f"Error reading file: {e}",
                session_id=self.session_id,
                path=path,
                exc_info=True,
            )

            # Notify tool handler of error
            if self._tool_handler and tool_call_id:
                try:
                    await self._tool_handler.on_tool_complete(tool_call_id, False, None, str(e))
                except Exception as handler_error:
                    self.logger.error(
                        f"Error in tool complete handler: {handler_error}", exc_info=True
                    )

            return CallToolResult(
                content=[text_content(f"Error reading file: {e}")],
                isError=True,
            )

    async def write_text_file(
        self, arguments: dict[str, Any], tool_use_id: str | None = None
    ) -> CallToolResult:
        """
        Write a text file using ACP filesystem support.

        Args:
            arguments: Tool arguments containing 'path' and 'content'
            tool_use_id: LLM's tool use ID (for matching with stream events)

        Returns:
            CallToolResult indicating success or failure
        """
        # Validate arguments
        if not isinstance(arguments, dict):
            return CallToolResult(
                content=[text_content("Error: arguments must be a dict")],
                isError=True,
            )

        path = arguments.get("path")
        if not path or not isinstance(path, str):
            return CallToolResult(
                content=[text_content("Error: 'path' argument is required and must be a string")],
                isError=True,
            )

        content = arguments.get("content")
        if content is None or not isinstance(content, str):
            return CallToolResult(
                content=[
                    text_content("Error: 'content' argument is required and must be a string")
                ],
                isError=True,
            )

        self.logger.info(
            "Writing file via ACP filesystem",
            session_id=self.session_id,
            path=path,
            content_length=len(content),
        )

        # Check permission before execution
        if self._permission_handler:
            try:
                permission_result = await self._permission_handler.check_permission(
                    tool_name="write_text_file",
                    server_name="acp_filesystem",
                    arguments=arguments,
                    tool_use_id=tool_use_id,
                )
                if not permission_result.allowed:
                    error_msg = permission_result.error_message or (
                        f"Permission denied for writing file: {path}"
                    )
                    self.logger.info(
                        "File write denied by permission handler",
                        data={
                            "path": path,
                            "cancelled": permission_result.is_cancelled,
                        },
                    )
                    return CallToolResult(
                        content=[text_content(error_msg)],
                        isError=True,
                    )
            except Exception as e:
                self.logger.error(f"Error checking file write permission: {e}", exc_info=True)
                # Fail-safe: deny on permission check error
                return CallToolResult(
                    content=[text_content(f"Permission check failed: {e}")],
                    isError=True,
                )

        # Notify tool handler that execution is starting
        tool_call_id = None
        if self._tool_handler:
            try:
                tool_call_id = await self._tool_handler.on_tool_start(
                    "write_text_file", "acp_filesystem", arguments, tool_use_id
                )
            except Exception as e:
                self.logger.error(f"Error in tool start handler: {e}", exc_info=True)

        try:
            # Send request using the proper ACP method with flattened parameters
            await self.connection.write_text_file(
                content=content,
                path=path,
                session_id=self.session_id,
            )

            self.logger.info(
                "File write completed",
                session_id=self.session_id,
                path=path,
            )

            result = CallToolResult(
                content=[text_content(f"Successfully wrote {len(content)} characters to {path}")],
                isError=False,
            )

            # Notify tool handler of completion
            if self._tool_handler and tool_call_id:
                try:
                    await self._tool_handler.on_tool_complete(
                        tool_call_id, True, result.content, None
                    )
                except Exception as e:
                    self.logger.error(f"Error in tool complete handler: {e}", exc_info=True)

            return result

        except Exception as e:
            self.logger.error(
                f"Error writing file: {e}",
                session_id=self.session_id,
                path=path,
                exc_info=True,
            )

            # Notify tool handler of error
            if self._tool_handler and tool_call_id:
                try:
                    await self._tool_handler.on_tool_complete(tool_call_id, False, None, str(e))
                except Exception as handler_error:
                    self.logger.error(
                        f"Error in tool complete handler: {handler_error}", exc_info=True
                    )

            return CallToolResult(
                content=[text_content(f"Error writing file: {e}")],
                isError=True,
            )

    def metadata(self) -> dict[str, Any]:
        """
        Get metadata about this runtime for display/logging.

        Returns:
            Dict with runtime information
        """
        enabled_tools = []
        if self._enable_read:
            enabled_tools.append("read_text_file")
        if self._enable_write:
            enabled_tools.append("write_text_file")

        return {
            "type": "acp_filesystem",
            "session_id": self.session_id,
            "activation_reason": self.activation_reason,
            "tools": enabled_tools,
        }
