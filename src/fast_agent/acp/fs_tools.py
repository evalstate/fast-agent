"""
ACPFilesystemTools - Provide fs/read_text_file and fs/write_text_file tools via ACP.

This module provides filesystem tools that delegate to the ACP client's filesystem
capabilities when available. These tools allow the agent to read and write text files
in the client's filesystem.
"""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.tools.fastmcp_tool import FastMCPTool

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)


class ACPFilesystemTools:
    """
    Provides filesystem read/write tools through ACP fs support.

    This class creates FastMCPTool instances for fs/read_text_file and fs/write_text_file
    that delegate to the ACP client's filesystem capabilities.
    """

    def __init__(
        self,
        connection: "AgentSideConnection",
        session_id: str,
        cwd: str,
    ):
        """
        Initialize the ACP filesystem tools.

        Args:
            connection: The ACP connection to use for fs operations
            session_id: The ACP session ID for these tools
            cwd: The current working directory (absolute path) for this session
        """
        self.connection = connection
        self.session_id = session_id
        self.cwd = cwd

        logger.info(
            "ACPFilesystemTools initialized",
            session_id=session_id,
            cwd=cwd,
        )

    def get_read_text_file_tool(self) -> FastMCPTool:
        """
        Create a FastMCPTool for reading text files via ACP.

        Returns:
            FastMCPTool configured for fs/read_text_file
        """

        async def read_text_file(
            path: str = Field(
                description=f"Absolute path to the file to read. Current working directory: {self.cwd}"
            ),
            line: Optional[int] = Field(
                default=None,
                description="Line number to start reading from (1-based). If not specified, reads from the beginning.",
            ),
            limit: Optional[int] = Field(
                default=None,
                description="Maximum number of lines to read. If not specified, reads the entire file.",
            ),
        ) -> str:
            """
            Read content from a text file in the client's file system.

            This tool uses the ACP fs/read_text_file capability to read files from the
            client's filesystem. The path must be an absolute path.
            """
            try:
                logger.info(
                    "Reading text file via ACP",
                    session_id=self.session_id,
                    path=path,
                    line=line,
                    limit=limit,
                )

                # Build request params per ACP spec
                params: dict[str, Any] = {
                    "sessionId": self.session_id,
                    "path": path,
                }

                # Add optional parameters if provided
                if line is not None:
                    params["line"] = line
                if limit is not None:
                    params["limit"] = limit

                # Call ACP fs/read_text_file
                result = await self.connection._conn.send_request("fs/read_text_file", params)
                content = result.get("content", "")

                logger.info(
                    "Text file read successfully",
                    session_id=self.session_id,
                    path=path,
                    content_length=len(content),
                )

                return content

            except Exception as e:
                error_msg = f"Error reading file '{path}': {e}"
                logger.error(
                    error_msg,
                    session_id=self.session_id,
                    path=path,
                    exc_info=True,
                )
                raise RuntimeError(error_msg) from e

        tool = FastMCPTool.from_function(read_text_file)
        tool.name = "read_text_file"
        tool.description = (
            f"Read content from a text file in the client's file system. "
            f"The path must be an absolute path. "
            f"Current working directory: {self.cwd}. "
            f"Optionally specify 'line' (1-based) to start reading from a specific line, "
            f"and 'limit' to read only a certain number of lines."
        )

        return tool

    def get_write_text_file_tool(self) -> FastMCPTool:
        """
        Create a FastMCPTool for writing text files via ACP.

        Returns:
            FastMCPTool configured for fs/write_text_file
        """

        async def write_text_file(
            path: str = Field(
                description=f"Absolute path to the file to write. Current working directory: {self.cwd}"
            ),
            content: str = Field(description="The text content to write to the file."),
        ) -> str:
            """
            Write content to a text file in the client's file system.

            This tool uses the ACP fs/write_text_file capability to write files to the
            client's filesystem. The path must be an absolute path. This will create or
            overwrite the file.
            """
            try:
                logger.info(
                    "Writing text file via ACP",
                    session_id=self.session_id,
                    path=path,
                    content_length=len(content),
                )

                # Build request params per ACP spec
                params: dict[str, Any] = {
                    "sessionId": self.session_id,
                    "path": path,
                    "content": content,
                }

                # Call ACP fs/write_text_file
                await self.connection._conn.send_request("fs/write_text_file", params)

                success_msg = f"Successfully wrote {len(content)} characters to '{path}'"
                logger.info(
                    "Text file written successfully",
                    session_id=self.session_id,
                    path=path,
                    content_length=len(content),
                )

                return success_msg

            except Exception as e:
                error_msg = f"Error writing file '{path}': {e}"
                logger.error(
                    error_msg,
                    session_id=self.session_id,
                    path=path,
                    exc_info=True,
                )
                raise RuntimeError(error_msg) from e

        tool = FastMCPTool.from_function(write_text_file)
        tool.name = "write_text_file"
        tool.description = (
            f"Write content to a text file in the client's file system. "
            f"The path must be an absolute path. "
            f"Current working directory: {self.cwd}. "
            f"This will create the file if it doesn't exist, or overwrite it if it does."
        )

        return tool

    def get_tools(self) -> list[FastMCPTool]:
        """
        Get all filesystem tools.

        Returns:
            List of FastMCPTool instances for all filesystem operations
        """
        return [
            self.get_read_text_file_tool(),
            self.get_write_text_file_tool(),
        ]
