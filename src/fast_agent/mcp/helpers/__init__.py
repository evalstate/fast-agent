"""
Helper modules for working with MCP content (Fast Agent namespace).

This mirrors the legacy fast_agent.mcp.helpers API to provide a stable,
cycle-free import path now that PromptMessageExtended lives in fast_agent.mcp.
"""

from .content_helpers import (
    ensure_multipart_messages,
    get_image_data,
    get_resource_text,
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_resource_link,
    is_text_content,
    normalize_to_extended_list,
    split_thinking_content,
    text_content,
)

__all__ = [
    "get_text",
    "get_image_data",
    "get_resource_uri",
    "is_text_content",
    "is_image_content",
    "is_resource_content",
    "is_resource_link",
    "get_resource_text",
    "ensure_multipart_messages",
    "normalize_to_extended_list",
    "split_thinking_content",
    "text_content",
]
