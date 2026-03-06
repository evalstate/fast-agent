"""Local filesystem runtime for shell-enabled agents.

Provides ACP-compatible ``read_text_file`` and ``write_text_file`` tool
implementations for non-ACP environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, TextContent, Tool


class LocalFilesystemRuntime:
    """Expose local filesystem tools with ACP-compatible signatures."""

    def __init__(
        self,
        logger,
        working_directory: Path | None = None,
        *,
        enable_read: bool = True,
        enable_write: bool = True,
    ) -> None:
        self._logger = logger
        self._working_directory = working_directory
        self._enable_read = enable_read
        self._enable_write = enable_write

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

    @property
    def tools(self) -> list[Tool]:
        """Return locally supported filesystem tools."""
        tools: list[Tool] = []
        if self._enable_read:
            tools.append(self._read_tool)
        if self._enable_write:
            tools.append(self._write_tool)
        return tools

    def set_enabled_tools(self, *, enable_read: bool, enable_write: bool) -> None:
        """Update enabled filesystem tool flags."""
        self._enable_read = enable_read
        self._enable_write = enable_write

    def set_working_directory(self, working_directory: Path | None) -> None:
        """Update the base directory used for relative file paths."""
        self._working_directory = working_directory

    def _base_directory(self) -> Path:
        if self._working_directory is None:
            return Path.cwd()
        if self._working_directory.is_absolute():
            return self._working_directory.resolve()
        return (Path.cwd() / self._working_directory).resolve()

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self._base_directory() / candidate).resolve()

    @staticmethod
    def _coerce_positive_int(value: Any, field: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"Error: '{field}' argument must be an integer greater than or equal to 1"
            )
        return value

    async def read_text_file(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        """Read a local text file, optionally slicing by line and limit."""
        del tool_use_id

        if not isinstance(arguments, dict):
            return CallToolResult(
                content=[TextContent(type="text", text="Error: arguments must be a dict")],
                isError=True,
            )

        path_value = arguments.get("path")
        if not path_value or not isinstance(path_value, str):
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="Error: 'path' argument is required and must be a string",
                    )
                ],
                isError=True,
            )

        try:
            line = self._coerce_positive_int(arguments.get("line"), "line")
            limit = self._coerce_positive_int(arguments.get("limit"), "limit")
        except ValueError as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc))],
                isError=True,
            )

        resolved_path = self._resolve_path(path_value.strip())

        try:
            content = resolved_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            self._logger.error(f"Error reading file: {exc}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error reading file: {exc}")],
                isError=True,
            )

        if line is not None or limit is not None:
            lines = content.splitlines()
            start_index = (line - 1) if line is not None else 0
            end_index = start_index + limit if limit is not None else None
            content = "\n".join(lines[start_index:end_index])

        self._logger.debug(f"Read local file: {resolved_path} ({len(content)} chars)")
        return CallToolResult(
            content=[TextContent(type="text", text=content)],
            isError=False,
        )

    async def write_text_file(
        self, arguments: dict[str, Any] | None = None, tool_use_id: str | None = None
    ) -> CallToolResult:
        """Write a local text file, creating parent directories as needed."""
        del tool_use_id

        if not isinstance(arguments, dict):
            return CallToolResult(
                content=[TextContent(type="text", text="Error: arguments must be a dict")],
                isError=True,
            )

        path_value = arguments.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="Error: 'path' argument is required and must be a string",
                    )
                ],
                isError=True,
            )

        content_value = arguments.get("content")
        if not isinstance(content_value, str):
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="Error: 'content' argument is required and must be a string",
                    )
                ],
                isError=True,
            )

        resolved_path = self._resolve_path(path_value.strip())
        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(content_value, encoding="utf-8", errors="replace")
        except Exception as exc:
            self._logger.error(f"Error writing file: {exc}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error writing file: {exc}")],
                isError=True,
            )

        self._logger.debug(f"Wrote local file: {resolved_path} ({len(content_value)} chars)")
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"Successfully wrote {len(content_value)} characters to {path_value.strip()}"
                    ),
                )
            ],
            isError=False,
        )

    def metadata(self) -> dict[str, Any]:
        """Expose runtime metadata for tool displays and diagnostics."""
        tools: list[str] = []
        if self._enable_read:
            tools.append("read_text_file")
        if self._enable_write:
            tools.append("write_text_file")

        return {
            "type": "local_filesystem",
            "tools": tools,
            "working_directory": str(self._base_directory()),
        }
