"""
Utilities for rendering PromptMessageMultipart objects for display.
"""

from typing import List

from mcp.types import BlobResourceContents, TextResourceContents

from mcp_agent.mcp.helpers.content_helpers import (
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_text_content,
)
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


def render_multipart_message(message: PromptMessageMultipart) -> str:
    """
    Render a multipart message for display purposes.
    
    This function formats the message content for user-friendly display,
    handling different content types appropriately.
    
    Args:
        message: A PromptMessageMultipart object to render
        
    Returns:
        A string representation of the message's content
    """
    rendered_parts: List[str] = []
    
    for content in message.content:
        if is_text_content(content):
            # Handle text content
            text = get_text(content)
            if text:
                rendered_parts.append(text)
            
        elif is_image_content(content):
            # Format details about the image
            image_data = getattr(content, "data", "")
            data_size = len(image_data) if image_data else 0
            mime_type = getattr(content, "mimeType", "unknown")
            image_info = f"[IMAGE: {mime_type}, {data_size} bytes]"
            rendered_parts.append(image_info)
            
        elif is_resource_content(content):
            # Handle embedded resources
            uri = get_resource_uri(content)
            resource = getattr(content, "resource", None)
            
            if resource and isinstance(resource, TextResourceContents):
                # Handle text resources
                text = resource.text
                text_length = len(text)
                mime_type = getattr(resource, "mimeType", "text/plain")
                
                # Preview with truncation for long content
                preview = text[:300] + ("..." if text_length > 300 else "")
                resource_info = f"[EMBEDDED TEXT RESOURCE: {mime_type}, {uri}, {text_length} chars]\n{preview}"
                rendered_parts.append(resource_info)
                
            elif resource and isinstance(resource, BlobResourceContents):
                # Handle blob resources (binary data)
                blob = getattr(resource, "blob", "")
                blob_length = len(blob) if blob else 0
                mime_type = getattr(resource, "mimeType", "application/octet-stream")
                
                resource_info = f"[EMBEDDED BLOB RESOURCE: {mime_type}, {uri}, {blob_length} bytes]"
                rendered_parts.append(resource_info)
                
        else:
            # Fallback for other content types
            text = get_text(content)
            if text is not None:
                rendered_parts.append(text)
    
    return "\n".join(rendered_parts)