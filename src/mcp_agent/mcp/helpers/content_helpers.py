"""
Helper functions for working with content objects.

These utilities simplify extracting content from content structures
without repetitive type checking.
"""

from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

from mcp.types import (
    BlobResourceContents,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
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


def ensure_multipart_messages(
    messages: List[Union["PromptMessageMultipart", PromptMessage]],
) -> List["PromptMessageMultipart"]:
    """
    Ensure all messages in a list are PromptMessageMultipart objects.

    This function handles mixed-type lists where some messages may be PromptMessage
    and others may be PromptMessageMultipart. Each PromptMessage is converted to
    PromptMessageMultipart individually, preserving any existing PromptMessageMultipart
    objects.

    Args:
        messages: List containing either PromptMessage or PromptMessageMultipart objects

    Returns:
        List of PromptMessageMultipart objects
    """
    # Import here to avoid circular dependency
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

    if not messages:
        return []

    result = []
    for message in messages:
        if isinstance(message, PromptMessage):
            # Convert single PromptMessage to PromptMessageMultipart
            result.append(PromptMessageMultipart(role=message.role, content=[message.content]))
        else:
            # Already a PromptMessageMultipart, keep as-is
            result.append(message)

    return result


def normalize_to_multipart_list(
    messages: Union[
        str,
        PromptMessage,
        "PromptMessageMultipart",
        List[Union[str, PromptMessage, "PromptMessageMultipart"]],
    ],
) -> List["PromptMessageMultipart"]:
    """
    Normalize various input types to a list of PromptMessageMultipart objects.

    This function provides a unified way to handle all possible message input types:
    - Single string → converts to list with one user PromptMessageMultipart
    - Single PromptMessage → converts to list with one PromptMessageMultipart
    - Single PromptMessageMultipart → wraps in a list
    - List of mixed types → converts each element appropriately

    Args:
        messages: Input in various formats (string, PromptMessage, PromptMessageMultipart, or list of these)

    Returns:
        List of PromptMessageMultipart objects
    """
    # Import here to avoid circular dependency
    from mcp_agent.core.prompt import Prompt
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

    # Handle single values by converting to list
    if not isinstance(messages, list):
        if isinstance(messages, str):
            # Convert string to user PromptMessageMultipart
            return [Prompt.user(messages)]
        elif isinstance(messages, PromptMessage):
            # Convert PromptMessage to PromptMessageMultipart
            return [PromptMessageMultipart(role=messages.role, content=[messages.content])]
        elif isinstance(messages, PromptMessageMultipart):
            # Wrap single PromptMessageMultipart in list
            return [messages]
        else:
            # Try to convert to string as fallback
            return [Prompt.user(str(messages))]

    # Handle list of messages
    result = []
    for message in messages:
        if isinstance(message, str):
            # Convert string to user PromptMessageMultipart
            result.append(Prompt.user(message))
        elif isinstance(message, PromptMessage):
            # Convert PromptMessage to PromptMessageMultipart
            result.append(PromptMessageMultipart(role=message.role, content=[message.content]))
        elif isinstance(message, PromptMessageMultipart):
            # Already correct type
            result.append(message)
        else:
            # Try to convert to string as fallback
            result.append(Prompt.user(str(message)))

    return result


def text_content(content: str) -> TextContent:
    return TextContent(type="text", text=content)
