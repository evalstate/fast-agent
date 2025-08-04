"""
Helper functions for working with content objects.

These utilities simplify extracting content from content structures
without repetitive type checking.
"""

from typing import Optional, Union

from mcp.types import (
    BlobResourceContents,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ReadResourceResult,
    ResourceLink,
    TextContent,
    TextResourceContents,
)


def get_text(content: ContentBlock) -> Optional[str]:
    """
    Extract text content from a content object if available.

    Args:
        content: A content object ContentBlock

    Returns:
        The text content as a string or None if not a text content
    """
    if isinstance(content, TextContent):
        return content.text

    if isinstance(content, TextResourceContents):
        return content.text

    if isinstance(content, EmbeddedResource):
        if isinstance(content.resource, TextResourceContents):
            return content.resource.text

    return None


def get_image_data(content: ContentBlock) -> Optional[str]:
    """
    Extract image data from a content object if available.

    Args:
        content: A content object ContentBlock

    Returns:
        The image data as a base64 string or None if not an image content
    """
    if isinstance(content, ImageContent):
        return content.data

    if isinstance(content, EmbeddedResource):
        if isinstance(content.resource, BlobResourceContents):
            # This assumes the blob might be an image, which isn't always true
            # Consider checking the mimeType if needed
            return content.resource.blob

    return None


def get_resource_uri(content: ContentBlock) -> Optional[str]:
    """
    Extract resource URI from an EmbeddedResource if available.

    Args:
        content: A content object ContentBlock

    Returns:
        The resource URI as a string or None if not an embedded resource
    """
    if isinstance(content, EmbeddedResource):
        return str(content.resource.uri)

    return None


def is_text_content(content: ContentBlock) -> bool:
    """
    Check if the content is text content.

    Args:
        content: A content object ContentBlock

    Returns:
        True if the content is TextContent, False otherwise
    """
    return isinstance(content, TextContent) or isinstance(content, TextResourceContents)


def is_image_content(content: Union[TextContent, ImageContent, EmbeddedResource]) -> bool:
    """
    Check if the content is image content.

    Args:
        content: A content object ContentBlock

    Returns:
        True if the content is ImageContent, False otherwise
    """
    return isinstance(content, ImageContent)


def is_resource_content(content: ContentBlock) -> bool:
    """
    Check if the content is an embedded resource.

    Args:
        content: A content object ContentBlock

    Returns:
        True if the content is EmbeddedResource, False otherwise
    """
    return isinstance(content, EmbeddedResource)


def is_resource_link(content: ContentBlock) -> bool:
    """
    Check if the content is an embedded resource.

    Args:
        content: A ContentBlock object

    Returns:
        True if the content is ResourceLink, False otherwise
    """
    return isinstance(content, ResourceLink)


def get_resource_text(result: ReadResourceResult, index: int = 0) -> Optional[str]:
    """
    Extract text content from a ReadResourceResult at the specified index.

    Args:
        result: A ReadResourceResult from an MCP resource read operation
        index: Index of the content item to extract text from (default: 0)

    Returns:
        The text content as a string or None if not available or not text content

    Raises:
        IndexError: If the index is out of bounds for the contents list
    """
    if index >= len(result.contents):
        raise IndexError(
            f"Index {index} out of bounds for contents list of length {len(result.contents)}"
        )

    content = result.contents[index]
    if isinstance(content, TextResourceContents):
        return content.text

    return None


def split_thinking_content(message: str) -> tuple[Optional[str], str]:
    """
    Split a message into thinking and content parts.

    Extracts content between <thinking> tags and returns it along with the remaining content.

    Args:
        message: A string that may contain a <thinking>...</thinking> block followed by content

    Returns:
        A tuple of (thinking_content, main_content) where:
        - thinking_content: The content inside <thinking> tags, or None if not found/parsing fails
        - main_content: The content after the thinking block, or the entire message if no thinking block
    """
    import re

    # Pattern to match <thinking>...</thinking> at the start of the message
    pattern = r"^<think>(.*?)</think>\s*(.*)$"
    match = re.match(pattern, message, re.DOTALL)

    if match:
        thinking_content = match.group(1).strip()
        main_content = match.group(2).strip()
        return (thinking_content, main_content)
    else:
        # No thinking block found or parsing failed
        return (None, message)
