"""Bound MCP tool results before they enter model history."""

from __future__ import annotations

from mcp.types import CallToolResult, ContentBlock, TextContent

from fast_agent.mcp.helpers.content_helpers import (
    canonicalize_tool_result_content_for_llm,
    get_text,
)
from fast_agent.tools.output_truncation import truncate_text_output

_TOOL_RESULT_TRUNCATION_GUIDANCE = (
    "Use a narrower query or request a smaller result to retain the relevant content."
)


def truncate_tool_result_for_llm(
    result: CallToolResult,
    *,
    byte_limit: int,
) -> CallToolResult:
    """Return a bounded copy when the canonical textual result exceeds the limit."""

    canonical = canonicalize_tool_result_content_for_llm(result)
    text = "\n".join(
        text for block in canonical if (text := get_text(block)) is not None
    )
    truncated = truncate_text_output(
        text,
        byte_limit=byte_limit,
        label="Tool result",
        guidance=_TOOL_RESULT_TRUNCATION_GUIDANCE,
    )
    if truncated is None:
        return result

    content: list[ContentBlock] = [TextContent(type="text", text=truncated.text)]
    content.extend(block for block in canonical if get_text(block) is None)
    return result.model_copy(
        update={
            "content": content,
            "structuredContent": None,
        }
    )
