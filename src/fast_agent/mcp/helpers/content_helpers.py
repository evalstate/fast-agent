"""
Helper functions for working with content objects (Fast Agent namespace).

"""

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

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
    """Extract text content from a content object if available."""
    if isinstance(content, TextContent):
        return content.text

    if isinstance(content, TextResourceContents):
        return content.text

    if isinstance(content, EmbeddedResource):
        if isinstance(content.resource, TextResourceContents):
            return content.resource.text

    if isinstance(content, ResourceLink):
        name = content.name or "unknown"
        uri_str = str(content.uri)
        mime_type = content.mimeType or "unknown"
        description = content.description or "No description"

        return (
            f"Linked Resource ${name} MIME type {mime_type}>\n"
            f"Resource Link: {uri_str}\n"
            f"${description}\n"
        )

    return None


def get_image_data(content: ContentBlock) -> Optional[str]:
    """Extract image data from a content object if available."""
    if isinstance(content, ImageContent):
        return content.data

    if isinstance(content, EmbeddedResource):
        if isinstance(content.resource, BlobResourceContents):
            return content.resource.blob

    return None


def get_resource_uri(content: ContentBlock) -> Optional[str]:
    """Extract resource URI from an EmbeddedResource if available."""
    if isinstance(content, EmbeddedResource):
        return str(content.resource.uri)
    return None


def is_text_content(content: ContentBlock) -> bool:
    """Check if the content is text content."""
    return isinstance(content, TextContent) or isinstance(content, TextResourceContents)


def is_image_content(content: Union[TextContent, ImageContent, EmbeddedResource]) -> bool:
    """Check if the content is image content."""
    return isinstance(content, ImageContent)


def is_resource_content(content: ContentBlock) -> bool:
    """Check if the content is an embedded resource."""
    return isinstance(content, EmbeddedResource)


def is_resource_link(content: ContentBlock) -> bool:
    """Check if the content is a resource link."""
    return isinstance(content, ResourceLink)


def get_resource_text(result: ReadResourceResult, index: int = 0) -> Optional[str]:
    """Extract text content from a ReadResourceResult at the specified index."""
    if index >= len(result.contents):
        raise IndexError(
            f"Index {index} out of bounds for contents list of length {len(result.contents)}"
        )
    content = result.contents[index]
    if isinstance(content, TextResourceContents):
        return content.text
    return None


def split_thinking_content(message: str) -> tuple[Optional[str], str]:
    """Split a message into thinking and content parts."""
    import re

    pattern = r"^<think>(.*?)</think>\s*(.*)$"
    match = re.match(pattern, message, re.DOTALL)

    if match:
        thinking_content = match.group(1).strip()
        main_content = match.group(2).strip()
        if main_content.startswith("<think>"):
            nested_thinking, remaining = split_thinking_content(main_content)
            if nested_thinking is not None:
                thinking_content = "\n".join(
                    part for part in [thinking_content, nested_thinking] if part
                )
                main_content = remaining
        return (thinking_content, main_content)
    else:
        return (None, message)


def text_content(text: str) -> TextContent:
    """Convenience to create a TextContent block from a string."""
    return TextContent(type="text", text=text)


def ensure_multipart_messages(
    messages: List[Union["PromptMessageExtended", PromptMessage]],
) -> List["PromptMessageExtended"]:
    """Ensure all messages in a list are PromptMessageExtended objects."""
    # Import here to avoid circular dependency
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

    if not messages:
        return []

    result = []
    for message in messages:
        if isinstance(message, PromptMessage):
            result.append(PromptMessageExtended(role=message.role, content=[message.content]))
        else:
            result.append(message)

    return result


def normalize_to_extended_list(
    messages: Union[
        str,
        PromptMessage,
        "PromptMessageExtended",
        Sequence[Union[str, PromptMessage, "PromptMessageExtended"]],
    ],
) -> List["PromptMessageExtended"]:
    """Normalize various input types to a list of PromptMessageExtended objects."""
    # Import here to avoid circular dependency
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

    if messages is None:
        return []

    # Single string → convert to user PromptMessageExtended
    if isinstance(messages, str):
        return [
            PromptMessageExtended(role="user", content=[TextContent(type="text", text=messages)])
        ]

    # Single PromptMessage → convert to PromptMessageExtended
    if isinstance(messages, PromptMessage):
        return [PromptMessageExtended(role=messages.role, content=[messages.content])]

    # Single PromptMessageExtended → wrap in a list
    if isinstance(messages, PromptMessageExtended):
        return [messages]

    # List of mixed types → convert each element
    result: List[PromptMessageExtended] = []
    for item in messages:
        if isinstance(item, str):
            result.append(
                PromptMessageExtended(role="user", content=[TextContent(type="text", text=item)])
            )
        elif isinstance(item, PromptMessage):
            result.append(PromptMessageExtended(role=item.role, content=[item.content]))
        else:
            result.append(item)

    return result
