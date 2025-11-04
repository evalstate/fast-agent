"""
MCP utilities and types for fast-agent.

Public API:
- `Prompt`: helper for constructing MCP prompts/messages.
- `PromptMessageExtended`: canonical message container used internally by providers.
- Helpers from `fast_agent.mcp.helpers` (re-exported for convenience).

Note: Backward compatibility for legacy `PromptMessageMultipart` imports is handled
via `fast_agent.mcp.prompt_message_multipart`, which subclasses `PromptMessageExtended`.
"""

from .common import SEP
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

__all__ = [
    "Prompt",
    # Common
    "SEP",
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


def __getattr__(name: str):
    # Lazily import to avoid circular imports with fast_agent.types
    if name == "Prompt":
        from .prompt import Prompt  # local import

        return Prompt

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
