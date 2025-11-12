"""
Convert ACP content blocks to MCP content blocks.

This module handles the translation of content from the Agent Client Protocol (ACP)
format to the Model Context Protocol (MCP) format used internally by fast-agent.
"""

from typing import List

from acp.schema import (  # noqa: I001
    AudioContentBlock as ACPAudioContent,
    EmbeddedResourceContentBlock as ACPEmbeddedResource,
    ImageContentBlock as ACPImageContent,
    TextContentBlock as ACPTextContent,
)
from mcp.types import (
    AudioContent,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    Role,
    TextContent,
)

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

logger = get_logger(__name__)


def convert_acp_content_to_mcp(acp_content: object) -> ContentBlock | None:
    """
    Convert a single ACP content block to MCP ContentBlock format.

    Args:
        acp_content: An ACP content block (TextContentBlock, ImageContentBlock, etc.)

    Returns:
        Corresponding MCP ContentBlock or None if conversion not supported

    Raises:
        ValueError: If the content type is unrecognized
    """
    # Handle TextContentBlock
    if isinstance(acp_content, ACPTextContent):
        return TextContent(
            type="text",
            text=acp_content.text,
            annotations=acp_content.annotations,
        )

    # Handle ImageContentBlock
    elif isinstance(acp_content, ACPImageContent):
        # Note: ACP has an optional 'uri' field that MCP doesn't support
        # We only use the data and mimeType fields
        return ImageContent(
            type="image",
            data=acp_content.data,
            mimeType=acp_content.mimeType,
            annotations=acp_content.annotations,
        )

    # Handle EmbeddedResourceContentBlock
    elif isinstance(acp_content, ACPEmbeddedResource):
        return EmbeddedResource(
            type="resource",
            resource=acp_content.resource,
            annotations=acp_content.annotations,
        )

    # Handle AudioContentBlock
    elif isinstance(acp_content, ACPAudioContent):
        return AudioContent(
            type="audio",
            data=acp_content.data,
            mimeType=acp_content.mimeType,
            annotations=acp_content.annotations,
        )

    # Unknown content type
    else:
        logger.warning(
            f"Unsupported ACP content type: {type(acp_content).__name__}",
            name="acp_content_unsupported",
            content_type=type(acp_content).__name__,
        )
        return None


def convert_acp_prompt_to_extended(
    acp_prompt: List[object], role: Role = "user"
) -> PromptMessageExtended:
    """
    Convert a list of ACP content blocks to a PromptMessageExtended.

    Args:
        acp_prompt: List of ACP content blocks from PromptRequest
        role: The role for the message (default: "user")

    Returns:
        PromptMessageExtended with converted content blocks

    Example:
        >>> from acp.schema import TextContentBlock, ImageContentBlock
        >>> acp_prompt = [
        ...     TextContentBlock(type="text", text="What's in this image?"),
        ...     ImageContentBlock(type="image", data="base64...", mimeType="image/png")
        ... ]
        >>> message = convert_acp_prompt_to_extended(acp_prompt)
        >>> len(message.content)
        2
    """
    content_blocks: List[ContentBlock] = []

    for acp_content in acp_prompt:
        mcp_content = convert_acp_content_to_mcp(acp_content)
        if mcp_content is not None:
            content_blocks.append(mcp_content)

    logger.info(
        f"Converted {len(acp_prompt)} ACP content blocks to {len(content_blocks)} MCP blocks",
        name="acp_content_converted",
        acp_count=len(acp_prompt),
        mcp_count=len(content_blocks),
    )

    return PromptMessageExtended(role=role, content=content_blocks)
