"""Bound MCP tool results before they enter model history."""

from __future__ import annotations

from mcp_types import CallToolResult, ContentBlock, TextContent

from fast_agent.mcp.helpers.content_helpers import (
    canonicalize_tool_result_content_for_llm,
    get_text,
)
from fast_agent.tools.output_truncation import truncate_text_output

_TOOL_RESULT_TRUNCATION_GUIDANCE = (
    "Use a narrower query or request a smaller result to retain the relevant content."
)


def _canonical_text(canonical: list[ContentBlock]) -> str:
    return "\n".join(text for block in canonical if (text := get_text(block)) is not None)


def tool_result_exceeds_byte_limit(result: CallToolResult, *, byte_limit: int) -> bool:
    """Return whether the canonical textual result would be truncated for the model."""

    text = _canonical_text(canonicalize_tool_result_content_for_llm(result))
    return len(text.encode("utf-8")) > byte_limit


def truncate_tool_result_for_llm(
    result: CallToolResult,
    *,
    byte_limit: int,
    guidance: str | None = None,
) -> CallToolResult:
    """Return a bounded copy when the canonical textual result exceeds the limit.

    ``guidance`` replaces the default advice, for example with the location of a
    retained copy of the complete result.
    """

    canonical = canonicalize_tool_result_content_for_llm(result)
    truncated = truncate_text_output(
        _canonical_text(canonical),
        byte_limit=byte_limit,
        label="Tool result",
        guidance=guidance or _TOOL_RESULT_TRUNCATION_GUIDANCE,
    )
    if truncated is None:
        return result

    content: list[ContentBlock] = [TextContent(type="text", text=truncated.text)]
    content.extend(block for block in canonical if get_text(block) is None)
    return result.model_copy(
        update={
            "content": content,
            "structured_content": None,
        }
    )
