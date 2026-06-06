"""
Helper functions for creating MCP content types with minimal code.

This module provides simple functions to create TextContent, ImageContent,
EmbeddedResource, and other MCP content types with minimal boilerplate.
"""

import base64
from pathlib import Path
from typing import Any, Protocol, Union, runtime_checkable

from mcp.types import (
    Annotations,
    BlobResourceContents,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    ReadResourceResult,
    ResourceContents,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl

from fast_agent.mcp.message_roles import MessageRole
from fast_agent.mcp.mime_utils import (
    guess_mime_type,
    is_binary_content,
    is_image_mime_type,
)
from fast_agent.types import PromptMessageExtended


@runtime_checkable
class _EmbeddedResourceLike(Protocol):
    type: str
    resource: object


def MCPText(
    text: str,
    role: MessageRole = "user",
    annotations: Annotations | None = None,
) -> dict:
    """
    Create a message with text content.

    Args:
        text: The text content
        role: Role of the message, defaults to "user"
        annotations: Optional annotations

    Returns:
        A dictionary with role and content that can be used in a prompt
    """
    return {
        "role": role,
        "content": TextContent(type="text", text=text, annotations=annotations),
    }


def MCPImage(
    path: str | Path | None = None,
    data: bytes | None = None,
    mime_type: str | None = None,
    role: MessageRole = "user",
    annotations: Annotations | None = None,
) -> dict:
    """
    Create a message with image content.

    Args:
        path: Path to the image file
        data: Raw image data bytes (alternative to path)
        mime_type: Optional mime type, will be guessed from path if not provided
        role: Role of the message, defaults to "user"
        annotations: Optional annotations

    Returns:
        A dictionary with role and content that can be used in a prompt
    """
    if path is None and data is None:
        raise ValueError("Either path or data must be provided")

    if path is not None and data is not None:
        raise ValueError("Only one of path or data can be provided")

    if path is not None:
        path = Path(path)
        if not mime_type:
            mime_type = guess_mime_type(str(path))
        with path.open("rb") as f:
            data = f.read()

    if not mime_type:
        mime_type = "image/png"  # Default

    if data is None:
        raise ValueError("Image data is missing after path resolution")

    b64_data = base64.b64encode(data).decode("ascii")

    return {
        "role": role,
        "content": ImageContent(
            type="image", data=b64_data, mimeType=mime_type, annotations=annotations
        ),
    }


def MCPFile(
    path: Union[str, Path],
    mime_type: str | None = None,
    role: MessageRole = "user",
    annotations: Annotations | None = None,
) -> dict:
    """
    Create a message with an embedded resource from a file.

    Args:
        path: Path to the resource file
        mime_type: Optional mime type, will be guessed from path if not provided
        role: Role of the message, defaults to "user"
        annotations: Optional annotations

    Returns:
        A dictionary with role and content that can be used in a prompt
    """
    path = Path(path)
    uri = f"file://{path.absolute()}"

    if not mime_type:
        mime_type = guess_mime_type(str(path))

    # Determine if this is text or binary content
    is_binary = is_binary_content(mime_type)

    if is_binary:
        # Read as binary
        binary_data = path.read_bytes()
        b64_data = base64.b64encode(binary_data).decode("ascii")

        resource = BlobResourceContents(uri=AnyUrl(uri), blob=b64_data, mimeType=mime_type)
    else:
        # Read as text
        try:
            text_data = path.read_text(encoding="utf-8")
            resource = TextResourceContents(uri=AnyUrl(uri), text=text_data, mimeType=mime_type)
        except UnicodeDecodeError:
            # Fallback to binary if text read fails
            binary_data = path.read_bytes()
            b64_data = base64.b64encode(binary_data).decode("ascii")
            resource = BlobResourceContents(
                uri=AnyUrl(uri), blob=b64_data, mimeType=mime_type or "application/octet-stream"
            )

    return {
        "role": role,
        "content": EmbeddedResource(type="resource", resource=resource, annotations=annotations),
    }


def MCPPrompt(
    *content_items: Union[
        dict,
        str,
        Path,
        bytes,
        ContentBlock,
        _EmbeddedResourceLike,
        ResourceContents,
        ReadResourceResult,
        PromptMessage,
        PromptMessageExtended,
    ],
    role: MessageRole = "user",
) -> list[dict]:
    """
    Create one or more prompt messages with various content types.

    This function intelligently creates different content types:
    - Strings become TextContent
    - File paths with image mime types become ImageContent
    - File paths with text mime types or other mime types become EmbeddedResource
    - Dicts with role and content are passed through unchanged
    - Raw bytes become ImageContent
    - TextContent objects are used directly
    - ImageContent objects are used directly
    - EmbeddedResource objects are used directly
    - ResourceContent objects are wrapped in EmbeddedResource
    - ReadResourceResult objects are expanded into multiple messages

    Args:
        *content_items: Content items of various types
        role: Role for all items (user or assistant)

    Returns:
        List of messages that can be used in a prompt
    """
    result: list[dict] = []
    for item in content_items:
        result.extend(_prompt_messages_for_item(item, role=role))
    return result


def _prompt_messages_for_item(
    item: object,
    *,
    role: MessageRole,
) -> list[dict]:
    messages = _prompt_messages_from_message_like(item, role=role)
    if messages is not None:
        return messages
    messages = _prompt_messages_from_file_like(item, role=role)
    if messages is not None:
        return messages
    messages = _prompt_messages_from_resource_like(item, role=role)
    if messages is not None:
        return messages
    return [MCPText(str(item), role=role)]


def _prompt_messages_from_message_like(
    item: object,
    *,
    role: MessageRole,
) -> list[dict] | None:
    if isinstance(item, dict) and "role" in item and "content" in item:
        return [item]
    if isinstance(item, PromptMessage):
        return [{"role": item.role, "content": item.content}]
    if isinstance(item, PromptMessageExtended):
        return [{"role": msg.role, "content": msg.content} for msg in item.from_multipart()]
    if isinstance(item, ContentBlock):
        return [{"role": role, "content": item}]
    if isinstance(item, str):
        return [MCPText(item, role=role)]
    return None


def _prompt_messages_from_file_like(
    item: object,
    *,
    role: MessageRole,
) -> list[dict] | None:
    if isinstance(item, Path):
        return [_prompt_message_from_path(item, role=role)]
    if isinstance(item, bytes):
        return [MCPImage(data=item, role=role)]
    return None


def _prompt_message_from_path(
    path: Path,
    *,
    role: MessageRole,
) -> dict:
    mime_type = guess_mime_type(str(path))
    if is_image_mime_type(mime_type):
        return MCPImage(path=path, role=role)
    return MCPFile(path=path, role=role)


def _embedded_resource_message(
    resource: TextResourceContents | BlobResourceContents,
    *,
    role: MessageRole,
) -> dict:
    return {"role": role, "content": EmbeddedResource(type="resource", resource=resource)}


def _prompt_messages_from_resource_like(
    item: object,
    *,
    role: MessageRole,
) -> list[dict] | None:
    if isinstance(item, _EmbeddedResourceLike) and item.type == "resource":
        return [_prompt_message_from_embedded_resource_like(item, role=role)]
    if isinstance(item, (TextResourceContents, BlobResourceContents)):
        return [_embedded_resource_message(item, role=role)]
    if isinstance(item, ResourceContents):
        return [MCPText(str(item), role=role)]
    if isinstance(item, ReadResourceResult):
        return [
            _embedded_resource_message(resource_content, role=role)
            for resource_content in item.contents
        ]
    return None


def _prompt_message_from_embedded_resource_like(
    item: _EmbeddedResourceLike,
    *,
    role: MessageRole,
) -> dict:
    resource = item.resource
    if isinstance(resource, (TextResourceContents, BlobResourceContents)):
        return _embedded_resource_message(resource, role=role)
    return MCPText(str(item), role=role)


def User(
    *content_items: Union[
        dict,
        str,
        Path,
        bytes,
        ContentBlock,
        ResourceContents,
        ReadResourceResult,
        PromptMessage,
        PromptMessageExtended,
    ],
) -> list[dict]:
    """Create user message(s) with various content types."""
    return MCPPrompt(*content_items, role="user")


def Assistant(
    *content_items: Union[
        dict,
        str,
        Path,
        bytes,
        ContentBlock,
        ResourceContents,
        ReadResourceResult,
        PromptMessage,
        PromptMessageExtended,
    ],
) -> list[dict]:
    """Create assistant message(s) with various content types."""
    return MCPPrompt(*content_items, role="assistant")


def create_message(content: Any, role: MessageRole = "user") -> dict:
    """
    Create a single prompt message from content of various types.

    Args:
        content: Content of various types (str, Path, bytes, etc.)
        role: Role of the message

    Returns:
        A dictionary with role and content that can be used in a prompt
    """
    messages = MCPPrompt(content, role=role)
    return messages[0] if messages else {}
