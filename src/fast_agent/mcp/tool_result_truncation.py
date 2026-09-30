"""Bound MCP tool results before they enter model history."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp_types import CallToolResult, ContentBlock, TextContent

from fast_agent.mcp.helpers.content_helpers import (
    canonicalize_tool_result_content_for_llm,
    get_text,
)
from fast_agent.tools.output_truncation import truncate_text_output

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_TOOL_RESULT_TRUNCATION_GUIDANCE = (
    "Use a narrower query or request a smaller result to retain the relevant content."
)


async def bound_tool_result_for_llm(
    result: CallToolResult,
    *,
    byte_limit: int,
    retain: Callable[[], Awaitable[str | None]],
) -> CallToolResult:
    """Return a bounded copy when the canonical textual result exceeds the limit.

    ``retain`` is awaited only when truncating; it may keep the complete result elsewhere
    and return guidance pointing at it, replacing the default advice.
    """

    canonical = canonicalize_tool_result_content_for_llm(result)
    text = "\n".join(text for block in canonical if (text := get_text(block)) is not None)
    if len(text.encode("utf-8")) <= byte_limit:
        return result
    truncated = truncate_text_output(
        text,
        byte_limit=byte_limit,
        label="Tool result",
        guidance=await retain() or _TOOL_RESULT_TRUNCATION_GUIDANCE,
    )
    assert truncated is not None

    content: list[ContentBlock] = [TextContent(type="text", text=truncated.text)]
    content.extend(block for block in canonical if get_text(block) is None)
    return result.model_copy(
        update={
            "content": content,
            "structured_content": None,
        }
    )
