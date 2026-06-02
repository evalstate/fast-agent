"""Display-only metadata helpers for MCP tool results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.types import CallToolResult, ContentBlock

_MEDIA_PREVIEW_ATTR = "_fast_agent_media_preview_content"


def set_tool_result_media_preview(
    result: CallToolResult,
    content: Sequence[ContentBlock],
) -> None:
    """Attach media preview blocks without changing provider-facing result content."""
    setattr(result, _MEDIA_PREVIEW_ATTR, list(content))


def get_tool_result_media_preview(
    result: CallToolResult,
) -> Sequence[ContentBlock] | None:
    """Return display-only media preview blocks attached to a tool result."""
    value = getattr(result, _MEDIA_PREVIEW_ATTR, None)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    return value
