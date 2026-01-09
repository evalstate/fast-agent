"""
Content block conversion from ACP to MCP format.

This module handles conversion of content blocks from the Agent Client Protocol (ACP)
to Model Context Protocol (MCP) format for processing by fast-agent.
"""

import re
from typing import cast

import acp.schema as acp_schema
import mcp.types as mcp_types
from acp.helpers import ContentBlock as ACPContentBlock
from mcp.types import ContentBlock as MCPContentBlock
from pydantic import AnyUrl


def convert_acp_content_to_mcp(acp_content: ACPContentBlock) -> MCPContentBlock | None:
    """
    Convert an ACP content block to MCP format.

    Args:
        acp_content: Content block from ACP (Agent Client Protocol)

    Returns:
        Corresponding MCP content block, or None if conversion is not supported

    Supported conversions:
        - TextContentBlock -> TextContent
        - ImageContentBlock -> ImageContent
        - EmbeddedResourceContentBlock -> EmbeddedResource
    """
    match acp_content:
        case acp_schema.TextContentBlock():
            return _convert_text_content(acp_content)
        case acp_schema.ImageContentBlock():
            return _convert_image_content(acp_content)
        case acp_schema.EmbeddedResourceContentBlock():
            return _convert_embedded_resource(acp_content)
        case _:
            # Unsupported content types (audio, resource links, etc.)
            return None


def _convert_text_content(
    acp_text: acp_schema.TextContentBlock,
) -> mcp_types.TextContent:
    """Convert ACP TextContentBlock to MCP TextContent."""
    return mcp_types.TextContent(
        type="text",
        text=acp_text.text,
        annotations=_convert_annotations(acp_text.annotations),
    )


def _convert_image_content(
    acp_image: acp_schema.ImageContentBlock,
) -> mcp_types.ImageContent:
    """Convert ACP ImageContentBlock to MCP ImageContent."""
    return mcp_types.ImageContent(
        type="image",
        data=acp_image.data,
        mimeType=acp_image.mimeType,
        annotations=_convert_annotations(acp_image.annotations),
    )


def _convert_embedded_resource(
    acp_resource: acp_schema.EmbeddedResourceContentBlock,
) -> mcp_types.EmbeddedResource:
    """Convert ACP EmbeddedResourceContentBlock to MCP EmbeddedResource."""
    return mcp_types.EmbeddedResource(
        type="resource",
        resource=_convert_resource_contents(acp_resource.resource),
        annotations=_convert_annotations(acp_resource.annotations),
    )


def _convert_resource_contents(
    acp_resource: acp_schema.TextResourceContents | acp_schema.BlobResourceContents,
) -> mcp_types.TextResourceContents | mcp_types.BlobResourceContents:
    """Convert ACP resource contents to MCP resource contents."""
    match acp_resource:
        case acp_schema.TextResourceContents():
            return mcp_types.TextResourceContents(
                uri=AnyUrl(acp_resource.uri),
                mimeType=acp_resource.mimeType or None,
                text=acp_resource.text,
            )
        case acp_schema.BlobResourceContents():
            return mcp_types.BlobResourceContents(
                uri=AnyUrl(acp_resource.uri),
                mimeType=acp_resource.mimeType or None,
                blob=acp_resource.blob,
            )
        case _:
            raise ValueError(f"Unsupported resource type: {type(acp_resource)}")


def _convert_annotations(
    acp_annotations: acp_schema.Annotations | None,
) -> mcp_types.Annotations | None:
    """Convert ACP annotations to MCP annotations."""
    if not acp_annotations:
        return None

    audience = (
        cast("list[mcp_types.Role]", list(acp_annotations.audience))
        if acp_annotations.audience
        else None
    )
    return mcp_types.Annotations(
        audience=audience,
        priority=getattr(acp_annotations, "priority", None),
    )


def _file_uri_to_path(uri: str) -> str:
    """
    Convert a file:// URI to a local filesystem path.

    Examples:
        file:///home/user/foo.txt -> /home/user/foo.txt
        file:///C:/Users/test/foo.txt -> C:/Users/test/foo.txt
        /already/a/path.txt -> /already/a/path.txt (unchanged)
    """
    if uri.startswith("file:///"):
        path = uri[7:]  # Remove "file://"
        # On Windows, file:///C:/path -> C:/path (remove leading /)
        # On Unix, file:///home/user -> /home/user (keep leading /)
        if len(path) > 2 and path[0] == "/" and path[2] == ":":
            # Windows path like /C:/Users/...
            path = path[1:]
        return path
    elif uri.startswith("file://"):
        # file://host/path or malformed - just strip file://
        return uri[7:]
    return uri


def inline_resources_for_slash_command(
    acp_prompt: list[ACPContentBlock],
) -> list[ACPContentBlock]:
    """
    If the prompt starts with a slash command, inline resource paths into the text.

    When ACP clients attach files via "@" syntax, they come as separate resource
    blocks. This function detects slash commands and converts file:// URIs to
    local paths so slash command handlers receive usable file paths.

    Handles two client behaviors:
    1. Text has "@filename" references that match resource URIs by filename:
       Input:  [TextBlock("/card @foo.txt"), ResourceBlock(uri="file:///path/foo.txt")]
       Output: [TextBlock("/card /path/foo.txt")]

    2. Text ends with trailing space and resources follow without "@" references:
       Input:  [TextBlock("/card "), ResourceBlock(uri="file:///foo.txt")]
       Output: [TextBlock("/card /foo.txt")]

    Args:
        acp_prompt: List of ACP content blocks

    Returns:
        Modified prompt with resource paths inlined if it's a slash command,
        otherwise returns the original prompt unchanged.
    """
    if not acp_prompt:
        return acp_prompt

    first_block = acp_prompt[0]

    # Only process if first block is text starting with "/"
    if not isinstance(first_block, acp_schema.TextContentBlock):
        return acp_prompt

    text = first_block.text
    if not text.lstrip().startswith("/"):
        return acp_prompt

    # Collect paths from resource blocks, indexed by filename
    # Convert file:// URIs to local paths for slash command handlers
    path_by_filename: dict[str, str] = {}
    paths: list[str] = []
    for block in acp_prompt[1:]:
        if isinstance(block, acp_schema.EmbeddedResourceContentBlock):
            uri = block.resource.uri
            if uri:
                uri_str = str(uri)
                # Convert file:// URI to local path
                path = _file_uri_to_path(uri_str)
                paths.append(path)
                # Extract filename for matching @references
                # e.g., "/path/to/foo.txt" -> "foo.txt"
                filename = path.rsplit("/", 1)[-1]
                # Also handle Windows paths with backslashes
                if "\\" in filename:
                    filename = filename.rsplit("\\", 1)[-1]
                if filename:
                    path_by_filename[filename] = path

    if not paths:
        return acp_prompt

    # Check if text contains @filename references to replace
    at_pattern = re.compile(r"@(\S+)")
    matches = at_pattern.findall(text)

    if matches:
        # Replace @filename with corresponding path
        inlined_text = text
        for match in matches:
            if match in path_by_filename:
                inlined_text = inlined_text.replace(f"@{match}", path_by_filename[match])
    else:
        # No @references found, append paths to the end
        inlined_text = text.rstrip() + " " + " ".join(paths)

    return [acp_schema.TextContentBlock(type="text", text=inlined_text)]


def convert_acp_prompt_to_mcp_content_blocks(
    acp_prompt: list[ACPContentBlock],
) -> list[MCPContentBlock]:
    """
    Convert a list of ACP content blocks to MCP content blocks.

    Args:
        acp_prompt: List of content blocks from ACP prompt

    Returns:
        List of MCP content blocks (only supported types are converted)
    """
    mcp_blocks = []

    for acp_block in acp_prompt:
        mcp_block = convert_acp_content_to_mcp(acp_block)
        if mcp_block is not None:
            mcp_blocks.append(mcp_block)

    return mcp_blocks
