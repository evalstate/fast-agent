"""
Convert ACP ContentBlocks to MCP ContentBlocks.

This module provides utilities to convert content blocks from the Agent Client Protocol (ACP)
format to the Model Context Protocol (MCP) format used internally by fast-agent.
"""

from typing import List, Union

from acp.schema import (
    AudioContentBlock,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
)
from acp.schema import (
    BlobResourceContents as ACPBlobResourceContents,
)
from acp.schema import (
    TextResourceContents as ACPTextResourceContents,
)
from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from mcp.types import (
    ContentBlock as MCPContentBlock,
)
from pydantic import AnyUrl

from fast_agent.core.logging.logger import get_logger

# Type alias for ACP ContentBlock (union of all content block types)
ACPContentBlock = Union[
    TextContentBlock,
    ImageContentBlock,
    AudioContentBlock,
    ResourceContentBlock,
    EmbeddedResourceContentBlock,
]

logger = get_logger(__name__)


def convert_acp_content_to_mcp(acp_content: ACPContentBlock) -> MCPContentBlock:
    """
    Convert a single ACP ContentBlock to an MCP ContentBlock.

    Args:
        acp_content: ACP ContentBlock to convert

    Returns:
        Corresponding MCP ContentBlock

    Raises:
        ValueError: If the content type is not supported
    """
    if isinstance(acp_content, TextContentBlock):
        return TextContent(
            type="text",
            text=acp_content.text,
            annotations=acp_content.annotations,
        )

    elif isinstance(acp_content, ImageContentBlock):
        return ImageContent(
            type="image",
            data=acp_content.data,
            mimeType=acp_content.mimeType,
            annotations=acp_content.annotations,
        )

    elif isinstance(acp_content, EmbeddedResourceContentBlock):
        # Convert the embedded resource
        resource = acp_content.resource

        if isinstance(resource, ACPTextResourceContents):
            mcp_resource = TextResourceContents(
                uri=AnyUrl(str(resource.uri)),
                text=resource.text,
                mimeType=resource.mimeType,
            )
        elif isinstance(resource, ACPBlobResourceContents):
            mcp_resource = BlobResourceContents(
                uri=AnyUrl(str(resource.uri)),
                blob=resource.blob,
                mimeType=resource.mimeType,
            )
        else:
            raise ValueError(f"Unsupported ACP resource type: {type(resource)}")

        return EmbeddedResource(
            type="resource",
            resource=mcp_resource,
            annotations=acp_content.annotations,
        )

    elif isinstance(acp_content, AudioContentBlock):
        # Audio content is not directly supported by MCP, but we can represent it
        # as an embedded resource with blob data
        logger.info(
            "Converting AudioContentBlock to MCP EmbeddedResource (blob)",
            name="acp_audio_conversion",
            mime_type=acp_content.mimeType,
        )

        # Create a synthetic URI for audio content
        uri = f"data:{acp_content.mimeType};base64,{acp_content.data[:50]}..."

        mcp_resource = BlobResourceContents(
            uri=AnyUrl(uri),
            blob=acp_content.data,
            mimeType=acp_content.mimeType,
        )

        return EmbeddedResource(
            type="resource",
            resource=mcp_resource,
            annotations=acp_content.annotations,
        )

    elif isinstance(acp_content, ResourceContentBlock):
        # ResourceContentBlock (resource_link) doesn't have a direct MCP equivalent
        # We'll convert it to a TextContent with formatted information
        logger.info(
            "Converting ResourceContentBlock to MCP TextContent",
            name="acp_resource_link_conversion",
            uri=str(acp_content.uri),
        )

        text_parts = [
            f"Resource: {acp_content.name}",
            f"URI: {acp_content.uri}",
        ]
        if acp_content.description:
            text_parts.append(f"Description: {acp_content.description}")
        if acp_content.mimeType:
            text_parts.append(f"MIME Type: {acp_content.mimeType}")
        if acp_content.size:
            text_parts.append(f"Size: {acp_content.size} bytes")

        return TextContent(
            type="text",
            text="\n".join(text_parts),
            annotations=acp_content.annotations,
        )

    else:
        raise ValueError(f"Unsupported ACP content block type: {type(acp_content)}")


def convert_acp_content_list_to_mcp(acp_content_list: List[ACPContentBlock]) -> List[MCPContentBlock]:
    """
    Convert a list of ACP ContentBlocks to MCP ContentBlocks.

    Args:
        acp_content_list: List of ACP ContentBlocks to convert

    Returns:
        List of corresponding MCP ContentBlocks
    """
    mcp_content_list = []

    for acp_content in acp_content_list:
        try:
            mcp_content = convert_acp_content_to_mcp(acp_content)
            mcp_content_list.append(mcp_content)
        except ValueError as e:
            logger.warning(
                f"Skipping unsupported ACP content block: {e}",
                name="acp_content_conversion_skip",
                content_type=type(acp_content).__name__,
            )

    return mcp_content_list
