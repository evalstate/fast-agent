"""
MCP utilities and types for fast-agent.

Public API:
- `FastAgent`: main application facade (compatibility re-export).
- `Prompt`: helper for constructing MCP prompts/messages.
- `PromptMessageExtended`: canonical message container used internally by providers.
- Helpers from `fast_agent.mcp.helpers` (re-exported for convenience).

Note: Backward compatibility for legacy `PromptMessageMultipart` imports is handled
via `fast_agent.mcp.prompt_message_multipart`, which subclasses `PromptMessageExtended`.
"""

from importlib import import_module
from typing import Any

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
    # Common
    "SEP",
    "FastAgent",
    "Prompt",
    # Helpers
    "ensure_multipart_messages",
    "get_image_data",
    "get_resource_text",
    "get_resource_uri",
    "get_text",
    "is_image_content",
    "is_resource_content",
    "is_resource_link",
    "is_text_content",
    "normalize_to_extended_list",
    "split_thinking_content",
    "text_content",
]


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Prompt": ("fast_agent.mcp.prompt", "Prompt"),
    "FastAgent": ("fast_agent.core.fastagent", "FastAgent"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    return getattr(import_module(module_name), attr_name)
