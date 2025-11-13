"""
ACPFilesystemRuntime - Read and write files via ACP filesystem support.

This runtime allows FastAgent to read and write files through the ACP client's
filesystem capabilities when available (e.g., in Zed editor). This provides better
integration compared to local file operations and allows access to unsaved editor state.
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
    Provides file read/write operations through ACP filesystem support.

    This runtime implements the "read_text_file" and "write_text_file" tools by
    delegating to the ACP client's filesystem capabilities. The flow is:
    1. fs/read_text_file - Read file content from client filesystem
    2. fs/write_text_file - Write file content to client filesystem

    The client (e.g., Zed editor) handles file operations and can access
    unsaved editor state.
    """

    def __init__(
        self,
        connection: "AgentSideConnection",
        session_id: str,
        activation_reason: str,
        logger_instance=None,
        supports_read: bool = True,
        supports_write: bool = True,
    ):
        """
        Initialize the ACP filesystem runtime.

        Args:
            connection: The ACP connection to use for filesystem operations
            session_id: The ACP session ID for this runtime
            activation_reason: Human-readable reason for activation
            logger_instance: Optional logger instance
            supports_read: Whether the client supports reading files
            supports_write: Whether the client supports writing files
        """
        self.connection = connection
        self.session_id = session_id
        self.activation_reason = activation_reason
        self.logger = logger_instance or logger
        self.supports_read = supports_read
        self.supports_write = supports_write

        # Tool definitions for LLM
        self._tools: list[Tool] = []

        # Add read_text_file tool if supported
        if self.supports_read:
            self._tools.append(
                Tool(
                    name="read_text_file",
                    description="Read the contents of a text file from the client's filesystem. "
                    "Can access unsaved changes in the editor. Supports reading the entire "
                    "file or a specific range of lines.",
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
            )

        # Add write_text_file tool if supported
        if self.supports_write:
            self._tools.append(
                Tool(
                    name="write_text_file",
                    description="Write content to a text file in the client's filesystem. "
                    "Creates the file if it doesn't exist. Overwrites existing content.",
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
            )

        self.logger.info(
            "ACPFilesystemRuntime initialized",
            session_id=session_id,
            reason=activation_reason,
            supports_read=supports_read,
            supports_write=supports_write,
        )

    @property
    def tools(self) -> list[Tool]:
        """Get the list of filesystem tool definitions."""
        return self._tools

    async def read_text_file(self, arguments: dict[str, Any]) -> CallToolResult:
        """
        Read a text file using ACP filesystem support.

        Args:
            arguments: Tool arguments containing 'path' and optional 'line', 'limit'

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
            # Build request params per ACP spec
            read_params: dict[str, Any] = {
                "sessionId": self.session_id,
                "path": path,
            }

            # Add optional parameters if provided
            if line := arguments.get("line"):
                read_params["line"] = line
            if limit := arguments.get("limit"):
                read_params["limit"] = limit

            # Send request to client
            result = await self.connection._conn.send_request(
                "fs/read_text_file", read_params
            )
            content = result.get("content", "")

            self.logger.info(
                "File read successfully",
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
            CallToolResult with success/error message
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
            write_params: dict[str, Any] = {
                "sessionId": self.session_id,
                "path": path,
                "content": content,
            }

            # Send request to client
            await self.connection._conn.send_request("fs/write_text_file", write_params)

            self.logger.info(
                "File written successfully",
                session_id=self.session_id,
                path=path,
            )

            return CallToolResult(
                content=[text_content(f"Successfully wrote to {path}")],
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

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """
        Route tool calls to the appropriate handler.

        Args:
            name: Tool name ('read_text_file' or 'write_text_file')
            arguments: Tool arguments

        Returns:
            CallToolResult from the appropriate handler
        """
        if name == "read_text_file":
            return await self.read_text_file(arguments)
        elif name == "write_text_file":
            return await self.write_text_file(arguments)
        else:
            return CallToolResult(
                content=[text_content(f"Unknown tool: {name}")],
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
            "supports_read": self.supports_read,
            "supports_write": self.supports_write,
        }
