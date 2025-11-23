"""
Utilities for rendering PromptMessageExtended objects for display.
"""

from typing import cast

from mcp.types import BlobResourceContents, EmbeddedResource, ImageContent, TextResourceContents

from fast_agent.mcp.helpers.content_helpers import (
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_text_content,
)
from fast_agent.types import PromptMessageExtended


def render_multipart_message(message: PromptMessageExtended) -> str:
    """
    Render a multipart message for display purposes.

    This function formats the message content for user-friendly display,
    handling different content types appropriately.

    Args:
        message: A PromptMessageExtended object to render

    Returns:
        A string representation of the message's content
    """
    rendered_parts: list[str] = []

    for content in message.content:
        if is_text_content(content):
            # Handle text content
            text = get_text(content)
            if text:
                rendered_parts.append(text)

        elif is_image_content(content):
            # Format details about the image
            image = cast("ImageContent", content)
            image_data = image.data
            data_size = len(image_data) if image_data else 0
            mime_type = image.mimeType
            image_info = f"[IMAGE: {mime_type}, {data_size} bytes]"
            rendered_parts.append(image_info)

        elif is_resource_content(content):
            # Handle embedded resources
            uri = get_resource_uri(content)
            embedded = cast("EmbeddedResource", content)
            resource = embedded.resource

            if isinstance(resource, TextResourceContents):
                # Handle text resources
                text = resource.text
                text_length = len(text)
                mime_type = resource.mimeType or "text/plain"

                # Preview with truncation for long content
                preview = text[:300] + ("..." if text_length > 300 else "")
                resource_info = (
                    f"[EMBEDDED TEXT RESOURCE: {mime_type}, {uri}, {text_length} chars]\n{preview}"
                )
                rendered_parts.append(resource_info)

            elif isinstance(resource, BlobResourceContents):
                # Handle blob resources (binary data)
                blob = resource.blob
                blob_length = len(blob) if blob else 0
                mime_type = resource.mimeType or "application/octet-stream"

                resource_info = f"[EMBEDDED BLOB RESOURCE: {mime_type}, {uri}, {blob_length} bytes]"
                rendered_parts.append(resource_info)

        else:
            # Fallback for other content types
            text = get_text(content)
            if text is not None:
                rendered_parts.append(text)

    return "\n".join(rendered_parts)
