"""
Utilities for rendering PromptMessageExtended objects for display.
"""

from collections.abc import Sequence

from mcp.types import BlobResourceContents, ContentBlock, TextResourceContents

from fast_agent.mcp.helpers.content_helpers import (
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_text_content,
)


def render_content_blocks(content_blocks: Sequence[ContentBlock]) -> str:
    """Render MCP content blocks for display without dumping binary payloads."""
    rendered_parts: list[str] = []

    for content in content_blocks:
        if is_text_content(content):
            # Handle text content
            text = get_text(content)
            if text:
                rendered_parts.append(text)

        elif is_image_content(content):
            # Format details about the image
            image = content
            image_data = image.data
            data_size = len(image_data) if image_data else 0
            mime_type = image.mimeType
            image_info = f"[IMAGE: {mime_type}, {data_size} bytes]"
            rendered_parts.append(image_info)

        elif is_resource_content(content):
            # Handle embedded resources
            uri = get_resource_uri(content)
            embedded = content
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
