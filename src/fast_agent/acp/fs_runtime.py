"""
ACPFilesystemRuntime - Read and write text files via ACP filesystem support.

This runtime allows FastAgent to access files through the ACP client's filesystem
capabilities when available (e.g., in Zed editor). This provides access to unsaved
editor state and allows clients to track file modifications.
"""

from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, Tool

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.content_helpers import text_content

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)


class ACPFilesystemRuntime:
    """
    Provides file access through ACP filesystem support.

    This runtime implements "read_text_file" and "write_text_file" tools by
    delegating to the ACP client's filesystem capabilities.

    The client (e.g., Zed editor) can provide access to unsaved editor state
    and track file modifications made during agent execution.
    """

    def __init__(
        self,
        connection: "AgentSideConnection",
        session_id: str,
        activation_reason: str,
        logger_instance=None,
    ):
        """
        Initialize the ACP filesystem runtime.

        Args:
            connection: The ACP connection to use for filesystem operations
            session_id: The ACP session ID for this runtime
            activation_reason: Human-readable reason for activation
            logger_instance: Optional logger instance
        """
        self.connection = connection
        self.session_id = session_id
        self.activation_reason = activation_reason
        self.logger = logger_instance or logger

        # Tool definitions for LLM
        self._read_tool = Tool(
            name="read_text_file",
            description="Read a text file from the client's filesystem. "
            "This can access unsaved editor state if the client supports it. "
            "Returns the file content as a string.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    },
                    "line": {
                        "type": "integer",
                        "description": "Optional line number to start reading from (1-based). "
                        "If provided, only reads from this line onwards.",
                        "minimum": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum number of lines to read. "
                        "If provided with 'line', reads up to this many lines starting from 'line'.",
                        "minimum": 1,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

        self._write_tool = Tool(
            name="write_text_file",
            description="Write a text file to the client's filesystem. "
            "Creates the file if it doesn't exist, or overwrites it if it does. "
            "The client may track this modification.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to write. "
                        "The file will be created if it doesn't exist.",
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
    def tools(self) -> list[Tool]:
        """Get the filesystem tool definitions."""
        return [self._read_tool, self._write_tool]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        """
        Execute a filesystem operation using ACP filesystem support.

        Args:
            tool_name: Name of the tool to execute ("read_text_file" or "write_text_file")
            arguments: Tool arguments

        Returns:
            CallToolResult with operation result
        """
        if tool_name == "read_text_file":
            return await self._read_text_file(arguments)
        elif tool_name == "write_text_file":
            return await self._write_text_file(arguments)
        else:
            return CallToolResult(
                content=[text_content(f"Error: Unknown filesystem tool '{tool_name}'")],
                isError=True,
            )

    async def _read_text_file(self, arguments: dict[str, Any]) -> CallToolResult:
        """
        Read a text file using ACP filesystem support.

        Args:
            arguments: Tool arguments containing 'path' key and optional 'line', 'limit'

        Returns:
            CallToolResult with file content
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
                content=[
                    text_content(
                        "Error: 'path' argument is required and must be a string"
                    )
                ],
                isError=True,
            )

        self.logger.info(
            "Reading file via ACP filesystem",
            session_id=self.session_id,
            path=path,
        )

        try:
            # Build request params per ACP spec
            request_params: dict[str, Any] = {
                "sessionId": self.session_id,
                "path": path,
            }

            # Add optional parameters if provided
            if line := arguments.get("line"):
                if not isinstance(line, int) or line < 1:
                    return CallToolResult(
                        content=[text_content("Error: 'line' must be a positive integer")],
                        isError=True,
                    )
                request_params["line"] = line

            if limit := arguments.get("limit"):
                if not isinstance(limit, int) or limit < 1:
                    return CallToolResult(
                        content=[text_content("Error: 'limit' must be a positive integer")],
                        isError=True,
                    )
                request_params["limit"] = limit

            # Call ACP client method
            result = await self.connection._conn.send_request(
                "fs/read_text_file", request_params
            )

            content = result.get("content", "")

            self.logger.info(
                "File read successfully",
                path=path,
                content_length=len(content),
            )

            return CallToolResult(
                content=[text_content(content)],
                isError=False,
            )

        except Exception as e:
            self.logger.error(
                f"Error reading file: {e}",
                path=path,
                exc_info=True,
            )
            return CallToolResult(
                content=[text_content(f"Error reading file: {e}")],
                isError=True,
            )

    async def _write_text_file(self, arguments: dict[str, Any]) -> CallToolResult:
        """
        Write a text file using ACP filesystem support.

        Args:
            arguments: Tool arguments containing 'path' and 'content' keys

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
                content=[
                    text_content(
                        "Error: 'path' argument is required and must be a string"
                    )
                ],
                isError=True,
            )

        content = arguments.get("content")
        if content is None or not isinstance(content, str):
            return CallToolResult(
                content=[
                    text_content(
                        "Error: 'content' argument is required and must be a string"
                    )
                ],
                isError=True,
            )

        self.logger.info(
            "Writing file via ACP filesystem",
            session_id=self.session_id,
            path=path,
            content_length=len(content),
        )

        try:
            # Build request params per ACP spec
            request_params: dict[str, Any] = {
                "sessionId": self.session_id,
                "path": path,
                "content": content,
            }

            # Call ACP client method
            await self.connection._conn.send_request(
                "fs/write_text_file", request_params
            )

            self.logger.info(
                "File written successfully",
                path=path,
            )

            return CallToolResult(
                content=[text_content(f"File written successfully to {path}")],
                isError=False,
            )

        except Exception as e:
            self.logger.error(
                f"Error writing file: {e}",
                path=path,
                exc_info=True,
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
        return {
            "type": "acp_filesystem",
            "session_id": self.session_id,
            "activation_reason": self.activation_reason,
        }
