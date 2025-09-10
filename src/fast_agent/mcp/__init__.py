"""
MCP utilities and types for Fast Agent.

Public API:
- `PromptMessageExtended`: canonical message container used internally by providers.
- Helpers from `fast_agent.mcp.helpers` (re-exported for convenience).

Note: Backward compatibility for legacy `PromptMessageMultipart` imports is handled
via `mcp_agent.mcp.prompt_message_multipart`, which subclasses `PromptMessageExtended`.
"""

from .helpers import (
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
from .prompt_message_extended import PromptMessageExtended

__all__ = [
    "PromptMessageExtended",
    # Helpers
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
