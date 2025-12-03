"""Shared type definitions and helpers for fast-agent.

Goals:
- Provide a stable import path for commonly used public types and helpers
- Keep dependencies minimal to reduce import-time cycles
"""

# Re-export common enums/types
# Public request parameters used to configure LLM calls
# Re-export ResourceLink from MCP for convenience
from mcp.types import ResourceLink

from fast_agent.llm.request_params import RequestParams

# Content helpers commonly used by users to build messages
from fast_agent.mcp.helpers.content_helpers import (
    audio_link,
    ensure_multipart_messages,
    image_link,
    normalize_to_extended_list,
    resource_link,
    text_content,
    video_link,
)

# Public message model used across providers and MCP integration
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

# Conversation analysis utilities
from .conversation_summary import ConversationSummary

# Stop reason enum - imported directly to avoid circular dependency
from .llm_stop_reason import LlmStopReason

# Message search utilities
from .message_search import extract_first, extract_last, find_matches, search_messages

__all__ = [
    # Enums / types
    "LlmStopReason",
    "PromptMessageExtended",
    "RequestParams",
    "ResourceLink",
    # Content helpers
    "text_content",
    "resource_link",
    "image_link",
    "video_link",
    "audio_link",
    "ensure_multipart_messages",
    "normalize_to_extended_list",
    # Analysis utilities
    "ConversationSummary",
    # Search utilities
    "search_messages",
    "find_matches",
    "extract_first",
    "extract_last",
]
