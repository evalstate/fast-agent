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
from typing import TYPE_CHECKING, Any

from .common import SEP

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
    "ensure_multipart_messages": (
        "fast_agent.mcp.helpers",
        "ensure_multipart_messages",
    ),
    "get_image_data": ("fast_agent.mcp.helpers", "get_image_data"),
    "get_resource_text": ("fast_agent.mcp.helpers", "get_resource_text"),
    "get_resource_uri": ("fast_agent.mcp.helpers", "get_resource_uri"),
    "get_text": ("fast_agent.mcp.helpers", "get_text"),
    "is_image_content": ("fast_agent.mcp.helpers", "is_image_content"),
    "is_resource_content": ("fast_agent.mcp.helpers", "is_resource_content"),
    "is_resource_link": ("fast_agent.mcp.helpers", "is_resource_link"),
    "is_text_content": ("fast_agent.mcp.helpers", "is_text_content"),
    "normalize_to_extended_list": (
        "fast_agent.mcp.helpers",
        "normalize_to_extended_list",
    ),
    "split_thinking_content": ("fast_agent.mcp.helpers", "split_thinking_content"),
    "text_content": ("fast_agent.mcp.helpers", "text_content"),
}

if TYPE_CHECKING:
    from fast_agent.core.fastagent import FastAgent as FastAgent
    from fast_agent.mcp.helpers import ensure_multipart_messages as ensure_multipart_messages
    from fast_agent.mcp.helpers import get_image_data as get_image_data
    from fast_agent.mcp.helpers import get_resource_text as get_resource_text
    from fast_agent.mcp.helpers import get_resource_uri as get_resource_uri
    from fast_agent.mcp.helpers import get_text as get_text
    from fast_agent.mcp.helpers import is_image_content as is_image_content
    from fast_agent.mcp.helpers import is_resource_content as is_resource_content
    from fast_agent.mcp.helpers import is_resource_link as is_resource_link
    from fast_agent.mcp.helpers import is_text_content as is_text_content
    from fast_agent.mcp.helpers import normalize_to_extended_list as normalize_to_extended_list
    from fast_agent.mcp.helpers import split_thinking_content as split_thinking_content
    from fast_agent.mcp.helpers import text_content as text_content
    from fast_agent.mcp.prompt import Prompt as Prompt


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
