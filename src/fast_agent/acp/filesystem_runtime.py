"""
ACPFilesystemRuntime - Read and write text files via ACP filesystem support.

This runtime allows FastAgent to read and write files through the ACP client's filesystem
capabilities when available (e.g., in Zed editor). This provides better integration and
security compared to direct file system access.
"""

from typing import TYPE_CHECKING, Any

from acp.schema import ReadTextFileRequest, ReadTextFileResponse, WriteTextFileRequest
from mcp.types import CallToolResult, Tool

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.content_helpers import text_content

if TYPE_CHECKING:
    from acp import AgentSideConnection

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
        """
        self.connection = connection
        self.session_id = session_id
        self.activation_reason = activation_reason
        self.logger = logger_instance or logger
        self._enable_read = enable_read
        self._enable_write = enable_write

        # Tool definition for reading text files
        self._read_tool = Tool(
            name="read_text_file",
            description="Read content from a text file. Returns the file contents as a string. "
            "The client handles file access and permissions.",
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
            description="Write content to a text file. Creates or overwrites the file. "
            "The client handles file access and permissions.",
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

    async def read_text_file(self, arguments: dict[str, Any]) -> CallToolResult:
        """
        Read a text file using ACP filesystem support.

        Args:
            arguments: Tool arguments containing 'path' and optionally 'line' and 'limit'

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
            # Build request using proper ACP schema
            request = ReadTextFileRequest(
                sessionId=self.session_id,
                path=path,
                line=arguments.get("line"),
                limit=arguments.get("limit"),
            )

            # Send request using the proper ACP method
            response: ReadTextFileResponse = await self.connection.readTextFile(request)
            content = response.content

            self.logger.info(
                "File read completed",
                session_id=self.session_id,
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
                session_id=self.session_id,
                path=path,
                exc_info=True,
            )
            return CallToolResult(
                content=[text_content(f"Error reading file: {e}")],
                isError=True,
            )

    async def write_text_file(self, arguments: dict[str, Any]) -> CallToolResult:
        """
        Write a text file using ACP filesystem support.

        Args:
            arguments: Tool arguments containing 'path' and 'content'

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
            # Build request using proper ACP schema
            request = WriteTextFileRequest(
                sessionId=self.session_id,
                path=path,
                content=content,
            )

            # Send request using the proper ACP method
            await self.connection.writeTextFile(request)

            self.logger.info(
                "File write completed",
                session_id=self.session_id,
                path=path,
            )

            return CallToolResult(
                content=[text_content(f"Successfully wrote {len(content)} characters to {path}")],
                isError=False,
            )

        except Exception as e:
            self.logger.error(
                f"Error writing file: {e}",
                session_id=self.session_id,
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
