"""Shared ACP-compatible filesystem tool definitions."""

from __future__ import annotations

from mcp.types import Tool


def build_read_text_file_tool() -> Tool:
    """Return the shared ``read_text_file`` tool definition."""
    return Tool(
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


def build_write_text_file_tool() -> Tool:
    """Return the shared ``write_text_file`` tool definition."""
    return Tool(
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
