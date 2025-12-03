"""
Helper functions for working with content objects (Fast Agent namespace).

"""

from typing import TYPE_CHECKING, Sequence, Union

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


def get_text(content: ContentBlock) -> str | None:
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
        description = content.description or ""

        lines = [
            f"[ResourceLink: {name} ({mime_type})]",
            f"URI: {uri_str}",
        ]
        if description:
            lines.append(description)

        return "\n".join(lines)

    return None


def get_image_data(content: ContentBlock) -> str | None:
    """Extract image data from a content object if available."""
    if isinstance(content, ImageContent):
        return content.data

    if isinstance(content, EmbeddedResource):
        if isinstance(content.resource, BlobResourceContents):
            return content.resource.blob

    return None


def get_resource_uri(content: ContentBlock) -> str | None:
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


def get_resource_text(result: ReadResourceResult, index: int = 0) -> str | None:
    """Extract text content from a ReadResourceResult at the specified index."""
    if index >= len(result.contents):
        raise IndexError(
            f"Index {index} out of bounds for contents list of length {len(result.contents)}"
        )
    content = result.contents[index]
    if isinstance(content, TextResourceContents):
        return content.text
    return None


def split_thinking_content(message: str) -> tuple[str | None, str]:
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


def _infer_mime_type(url: str, default: str = "application/octet-stream") -> str:
    """Infer MIME type from URL using the mimetypes database."""
    from urllib.parse import urlparse

    from fast_agent.mcp.mime_utils import guess_mime_type

    # Special case: YouTube URLs (Google has native support)
    parsed = urlparse(url.lower())
    youtube_hosts = ("youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com")
    if parsed.netloc in youtube_hosts:
        return "video/mp4"

    mime = guess_mime_type(url)
    # guess_mime_type returns "application/octet-stream" for unknown types
    if mime == "application/octet-stream":
        return default
    return mime


def _extract_name_from_url(url: str) -> str:
    """Extract a reasonable name from a URL."""
    from urllib.parse import unquote, urlparse

    path = urlparse(url).path
    if path:
        # Get the last path segment
        name = unquote(path.rstrip("/").split("/")[-1])
        if name:
            return name
    # Fallback to domain
    return urlparse(url).netloc or "resource"


def resource_link(
    url: str,
    *,
    name: str | None = None,
    mime_type: str | None = None,
    description: str | None = None,
) -> ResourceLink:
    """
    Create a ResourceLink from a URL with automatic MIME type inference.

    Args:
        url: The URL to the resource
        name: Optional name (defaults to filename from URL)
        mime_type: Optional MIME type (inferred from extension if not provided)
        description: Optional description

    Returns:
        A ResourceLink object
    """
    from pydantic import AnyUrl

    return ResourceLink(
        type="resource_link",
        uri=AnyUrl(url),
        name=name or _extract_name_from_url(url),
        mimeType=mime_type or _infer_mime_type(url),
        description=description,
    )


def image_link(
    url: str,
    *,
    name: str | None = None,
    mime_type: str | None = None,
    description: str | None = None,
) -> ResourceLink:
    """
    Create a ResourceLink for an image URL.

    Args:
        url: The URL to the image
        name: Optional name (defaults to filename from URL)
        mime_type: Optional MIME type (inferred from extension, defaults to image/jpeg)
        description: Optional description

    Returns:
        A ResourceLink object with image MIME type
    """
    inferred = _infer_mime_type(url, default="image/jpeg")
    # Ensure it's an image type
    if not inferred.startswith("image/"):
        inferred = "image/jpeg"

    return resource_link(
        url,
        name=name,
        mime_type=mime_type or inferred,
        description=description,
    )


def video_link(
    url: str,
    *,
    name: str | None = None,
    mime_type: str | None = None,
    description: str | None = None,
) -> ResourceLink:
    """
    Create a ResourceLink for a video URL.

    Args:
        url: The URL to the video
        name: Optional name (defaults to filename from URL)
        mime_type: Optional MIME type (inferred from extension, defaults to video/mp4)
        description: Optional description

    Returns:
        A ResourceLink object with video MIME type
    """
    inferred = _infer_mime_type(url, default="video/mp4")
    # Ensure it's a video type
    if not inferred.startswith("video/"):
        inferred = "video/mp4"

    return resource_link(
        url,
        name=name,
        mime_type=mime_type or inferred,
        description=description,
    )


def audio_link(
    url: str,
    *,
    name: str | None = None,
    mime_type: str | None = None,
    description: str | None = None,
) -> ResourceLink:
    """
    Create a ResourceLink for an audio URL.

    Args:
        url: The URL to the audio file
        name: Optional name (defaults to filename from URL)
        mime_type: Optional MIME type (inferred from extension, defaults to audio/mpeg)
        description: Optional description

    Returns:
        A ResourceLink object with audio MIME type
    """
    inferred = _infer_mime_type(url, default="audio/mpeg")
    # Ensure it's an audio type
    if not inferred.startswith("audio/"):
        inferred = "audio/mpeg"

    return resource_link(
        url,
        name=name,
        mime_type=mime_type or inferred,
        description=description,
    )


def ensure_multipart_messages(
    messages: list[Union["PromptMessageExtended", PromptMessage]],
) -> list["PromptMessageExtended"]:
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
) -> list["PromptMessageExtended"]:
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
    result: list[PromptMessageExtended] = []
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
