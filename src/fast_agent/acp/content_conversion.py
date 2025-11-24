"""
Content block conversion from ACP to MCP format.

This module handles conversion of content blocks from the Agent Client Protocol (ACP)
to Model Context Protocol (MCP) format for processing by fast-agent.
"""

from typing import Union

import acp.schema as acp_schema
import mcp.types as mcp_types

# Type aliases for clarity
ACPContentBlock = Union[
    acp_schema.TextContentBlock,
    acp_schema.ImageContentBlock,
    acp_schema.EmbeddedResourceContentBlock,
    acp_schema.ResourceContentBlock,
    acp_schema.AudioContentBlock,
]

MCPContentBlock = Union[
    mcp_types.TextContent,
    mcp_types.ImageContent,
    mcp_types.EmbeddedResource,
]


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
    content_type = getattr(acp_content, "type", None)

    if content_type == "text":
        return _convert_text_content(acp_content)
    elif content_type == "image":
        return _convert_image_content(acp_content)
    elif content_type == "resource":
        return _convert_embedded_resource(acp_content)
    else:
        # Unsupported content types (audio, resource links, etc.)
        return None


def _convert_text_content(
    acp_text: acp_schema.TextContentBlock,
) -> mcp_types.TextContent:
    """Convert ACP TextContentBlock to MCP TextContent."""
    return mcp_types.TextContent(
        type="text",
        text=acp_text.text,
        annotations=_convert_annotations(acp_text.annotations)
        if hasattr(acp_text, "annotations") and acp_text.annotations
        else None,
    )


def _convert_image_content(
    acp_image: acp_schema.ImageContentBlock,
) -> mcp_types.ImageContent:
    """Convert ACP ImageContentBlock to MCP ImageContent."""
    return mcp_types.ImageContent(
        type="image",
        data=acp_image.data,
        mimeType=acp_image.mimeType,
        annotations=_convert_annotations(acp_image.annotations)
        if hasattr(acp_image, "annotations") and acp_image.annotations
        else None,
    )


def _convert_embedded_resource(
    acp_resource: acp_schema.EmbeddedResourceContentBlock,
) -> mcp_types.EmbeddedResource:
    """Convert ACP EmbeddedResourceContentBlock to MCP EmbeddedResource."""
    # Convert the nested resource contents
    mcp_resource_contents = _convert_resource_contents(acp_resource.resource)

    return mcp_types.EmbeddedResource(
        type="resource",
        resource=mcp_resource_contents,
        annotations=_convert_annotations(acp_resource.annotations)
        if hasattr(acp_resource, "annotations") and acp_resource.annotations
        else None,
    )


def _convert_resource_contents(
    acp_resource: Union[acp_schema.TextResourceContents, acp_schema.BlobResourceContents],
) -> Union[mcp_types.TextResourceContents, mcp_types.BlobResourceContents]:
    """Convert ACP resource contents to MCP resource contents."""
    if isinstance(acp_resource, acp_schema.TextResourceContents):
        return mcp_types.TextResourceContents(
            uri=acp_resource.uri,
            mimeType=acp_resource.mimeType if acp_resource.mimeType else None,
            text=acp_resource.text,
        )
    elif isinstance(acp_resource, acp_schema.BlobResourceContents):
        return mcp_types.BlobResourceContents(
            uri=acp_resource.uri,
            mimeType=acp_resource.mimeType if acp_resource.mimeType else None,
            blob=acp_resource.blob,
        )
    else:
        raise ValueError(f"Unsupported resource type: {type(acp_resource)}")


def _convert_annotations(
    acp_annotations: acp_schema.Annotations | None,
) -> mcp_types.Annotations | None:
    """Convert ACP annotations to MCP annotations."""
    if not acp_annotations:
        return None

    # Both ACP and MCP Annotations have the same structure
    # Copy over all available fields
    return mcp_types.Annotations(
        audience=acp_annotations.audience
        if hasattr(acp_annotations, "audience")
        else None,
        priority=acp_annotations.priority
        if hasattr(acp_annotations, "priority")
        else None,
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
