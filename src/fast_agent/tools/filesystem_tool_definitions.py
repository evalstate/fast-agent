"""Shared ACP-compatible filesystem tool definitions."""

from __future__ import annotations

from mcp.types import Tool


def build_attach_media_tool(
    supported_mime_types: list[str] | None = None,
    *,
    is_google: bool = False,
) -> Tool:
    """Return the shared ``attach_media`` tool definition."""
    supported = ""
    if supported_mime_types:
        supported = (
            " Supported MIME types for the current model include: "
            + ", ".join(sorted(set(supported_mime_types)))
            + "."
        )

    youtube_suffix = ", and Gemini YouTube links" if is_google else ""
    return Tool(
        name="attach_media",
        description=(
            "Stage a local file, file:// URI, or provider-fetchable media/document URL as "
            "multimodal user input for the next model call. Use this for images, PDFs, audio, "
            f"and video{youtube_suffix}. Do not use this for internal:// or MCP resource "
            "URIs; use get_resource for those. Use read_text_file for plain text/code files."
            + supported
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": (
                        "Local path, file:// URI, or provider-fetchable remote URI/URL to attach."
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


def build_attach_resource_tool(
    supported_mime_types: list[str] | None = None,
    *,
    is_google: bool = False,
) -> Tool:
    """Deprecated compatibility alias for ``attach_media`` tool schema."""
    return build_attach_media_tool(supported_mime_types, is_google=is_google)


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
