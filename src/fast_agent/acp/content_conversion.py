"""
Content block conversion from ACP to MCP format.

This module handles conversion of content blocks from the Agent Client Protocol (ACP)
to Model Context Protocol (MCP) format for processing by fast-agent.
"""

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
