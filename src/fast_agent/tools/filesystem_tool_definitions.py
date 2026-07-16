"""Shared ACP-compatible filesystem tool definitions."""

from __future__ import annotations

from typing import Final

from mcp.types import Tool

ATTACH_MEDIA_TOOL_NAME: Final = "attach_media"
READ_TEXT_FILE_TOOL_NAME: Final = "read_text_file"
WRITE_TEXT_FILE_TOOL_NAME: Final = "write_text_file"


def build_attach_media_tool(
    supported_mime_types: list[str] | None = None,
    *,
    is_google: bool = False,
    max_bytes: int | None = None,
) -> Tool:
    """Return the shared ``attach_media`` tool definition."""
    capability_text = "No embedded media MIME types are advertised for the current model."
    if supported_mime_types:
        capability_text = (
            "Supported MIME types for the current model: "
            + ", ".join(sorted(set(supported_mime_types)))
            + "."
        )

    size_text = ""
    if max_bytes is not None:
        size_text = f" Local embedded files must be no larger than {max_bytes} bytes."
    youtube_text = " Gemini YouTube links are also supported." if is_google else ""
    return Tool(
        name=ATTACH_MEDIA_TOOL_NAME,
        description=(
            "Stage a file from the active filesystem, file:// URI, or provider-fetchable "
            "media/document URL as multimodal user input for the next model call. "
            f"{capability_text}{size_text}{youtube_text} Convert unsupported image formats to "
            "a supported format and unsupported video to image frames. Do not use this for "
            "internal:// or MCP resource URIs; use get_resource for those. Use read_text_file "
            "for plain text/code files."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": (
                        "Active filesystem path, file:// URI, or provider-fetchable remote URI/URL to attach."
                    ),
                },
                "mime_type": {
                    "type": "string",
                    "description": "Optional MIME type override. If omitted, inferred from extension/URL.",
                },
                "name": {
                    "type": "string",
                    "description": "Optional display name for linked resources.",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Optional short context label for linked resources. Ignored for most "
                        "local embedded media."
                    ),
                },
            },
            "required": ["source"],
            "additionalProperties": False,
        },
    )


def build_read_text_file_tool() -> Tool:
    """Return the shared ``read_text_file`` tool definition."""
    return Tool(
        name=READ_TEXT_FILE_TOOL_NAME,
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
        name=WRITE_TEXT_FILE_TOOL_NAME,
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
