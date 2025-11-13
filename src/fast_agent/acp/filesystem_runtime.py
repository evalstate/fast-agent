"""
ACPFilesystemRuntime - File system operations via ACP filesystem support.

This runtime allows FastAgent to read and write text files through the ACP client's
filesystem capabilities when available. This provides better integration by allowing
access to unsaved editor state and tracking file modifications.
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
    Provides file system operations through ACP filesystem support.

    This runtime implements "fs_read_text_file" and "fs_write_text_file" tools
    by delegating to the ACP client's filesystem capabilities.

    The client handles file access, including unsaved editor state.
    """

    def __init__(
        self,
        connection: "AgentSideConnection",
        session_id: str,
        cwd: str | None = None,
        logger_instance=None,
    ):
        """
        Initialize the ACP filesystem runtime.

        Args:
            connection: The ACP connection to use for filesystem operations
            session_id: The ACP session ID for this runtime
            cwd: The current working directory (absolute path)
            logger_instance: Optional logger instance
        """
        self.connection = connection
        self.session_id = session_id
        self.cwd = cwd or "/"
        self.logger = logger_instance or logger

        # Tool definitions for LLM
        cwd_note = f" The current working directory is: {self.cwd}" if self.cwd else ""

        self._read_tool = Tool(
            name="fs_read_text_file",
            description=(
                "Read the contents of a text file from the client's filesystem. "
                "This includes unsaved changes in the editor. "
                f"You MUST provide an absolute path.{cwd_note}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": f"Absolute path to the file to read.{cwd_note}",
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

        self._write_tool = Tool(
            name="fs_write_text_file",
            description=(
                "Write text content to a file in the client's filesystem. "
                "Creates the file if it doesn't exist. "
                f"You MUST provide an absolute path.{cwd_note}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": f"Absolute path to the file to write.{cwd_note}",
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
            cwd=self.cwd,
        )

    @property
    def tools(self) -> list[Tool]:
        """Get the filesystem tool definitions."""
        return [self._read_tool, self._write_tool]

    async def read_text_file(self, arguments: dict[str, Any]) -> CallToolResult:
        """
        Read a text file using ACP filesystem support.

        Args:
            arguments: Tool arguments containing 'path' key and optional 'line', 'limit'

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
                    text_content("Error: 'path' argument is required and must be a string")
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
                request_params["line"] = line
            if limit := arguments.get("limit"):
                request_params["limit"] = limit

            result = await self.connection._conn.send_request(
                "fs/read_text_file", request_params
            )
            content = result.get("content", "")

            self.logger.debug(
                f"Successfully read file via ACP: {path} ({len(content)} chars)"
            )

            return CallToolResult(
                content=[text_content(content)],
                isError=False,
            )

        except Exception as e:
            error_msg = f"Error reading file '{path}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return CallToolResult(
                content=[text_content(error_msg)],
                isError=True,
            )

    async def write_text_file(self, arguments: dict[str, Any]) -> CallToolResult:
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
                    text_content("Error: 'path' argument is required and must be a string")
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

            await self.connection._conn.send_request(
                "fs/write_text_file", request_params
            )

            success_msg = f"Successfully wrote {len(content)} characters to {path}"
            self.logger.debug(success_msg)

            return CallToolResult(
                content=[text_content(success_msg)],
                isError=False,
            )

        except Exception as e:
            error_msg = f"Error writing file '{path}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return CallToolResult(
                content=[text_content(error_msg)],
                isError=True,
            )

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        """
        Execute a filesystem tool by name.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            CallToolResult from the appropriate tool method
        """
        if tool_name == "fs_read_text_file":
            return await self.read_text_file(arguments)
        elif tool_name == "fs_write_text_file":
            return await self.write_text_file(arguments)
        else:
            return CallToolResult(
                content=[text_content(f"Unknown filesystem tool: {tool_name}")],
                isError=True,
            )
