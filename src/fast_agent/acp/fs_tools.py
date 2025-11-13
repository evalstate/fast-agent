"""
ACP Filesystem Tools - Provides fs/read_text_file and fs/write_text_file tools.

These tools allow the agent to read and write files on the client's filesystem
via the ACP protocol, when the client indicates support for filesystem capabilities.
"""

from typing import Annotated

from acp import AgentSideConnection
from acp.schema import (
    ReadTextFileRequest,
    ReadTextFileResponse,
    WriteTextFileRequest,
)
from mcp.server.fastmcp.tools.base import Tool as FastMCPTool

from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


def create_acp_read_text_file_tool(
    connection: AgentSideConnection,
    session_id: str,
    cwd: str,
) -> FastMCPTool:
    """
    Create an ACP read_text_file tool that forwards requests to the client.

    Args:
        connection: The ACP connection to the client
        session_id: The session ID for logging
        cwd: The current working directory (absolute path)

    Returns:
        FastMCPTool that can read text files via ACP
    """

    async def fs__read_text_file(
        path: Annotated[
            str,
            f"Absolute path to the text file to read. The current working directory is: {cwd}",
        ],
    ) -> str:
        """
        Read a text file from the client's filesystem.

        This tool uses the ACP protocol to read files on the client's machine,
        not the agent's machine. Always use absolute paths.
        """
        logger.info(
            "ACP fs/read_text_file request",
            name="acp_fs_read",
            session_id=session_id,
            path=path,
        )

        try:
            request = ReadTextFileRequest(path=path)
            response: ReadTextFileResponse = await connection.readTextFile(request)

            logger.info(
                "ACP fs/read_text_file success",
                name="acp_fs_read_success",
                session_id=session_id,
                path=path,
                content_length=len(response.content),
            )

            return response.content

        except Exception as e:
            error_msg = f"Failed to read file '{path}': {e}"
            logger.error(
                "ACP fs/read_text_file error",
                name="acp_fs_read_error",
                session_id=session_id,
                path=path,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(error_msg) from e

    return FastMCPTool.from_function(fs__read_text_file)


def create_acp_write_text_file_tool(
    connection: AgentSideConnection,
    session_id: str,
    cwd: str,
) -> FastMCPTool:
    """
    Create an ACP write_text_file tool that forwards requests to the client.

    Args:
        connection: The ACP connection to the client
        session_id: The session ID for logging
        cwd: The current working directory (absolute path)

    Returns:
        FastMCPTool that can write text files via ACP
    """

    async def fs__write_text_file(
        path: Annotated[
            str,
            f"Absolute path to the text file to write. The current working directory is: {cwd}",
        ],
        content: Annotated[str, "The text content to write to the file"],
    ) -> str:
        """
        Write a text file to the client's filesystem.

        This tool uses the ACP protocol to write files on the client's machine,
        not the agent's machine. Always use absolute paths.
        """
        logger.info(
            "ACP fs/write_text_file request",
            name="acp_fs_write",
            session_id=session_id,
            path=path,
            content_length=len(content),
        )

        try:
            request = WriteTextFileRequest(path=path, content=content)
            await connection.writeTextFile(request)

            logger.info(
                "ACP fs/write_text_file success",
                name="acp_fs_write_success",
                session_id=session_id,
                path=path,
                content_length=len(content),
            )

            return f"Successfully wrote {len(content)} characters to {path}"

        except Exception as e:
            error_msg = f"Failed to write file '{path}': {e}"
            logger.error(
                "ACP fs/write_text_file error",
                name="acp_fs_write_error",
                session_id=session_id,
                path=path,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(error_msg) from e

    return FastMCPTool.from_function(fs__write_text_file)
