"""Bound MCP tool results before they enter model history."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp_types import CallToolResult, ContentBlock, TextContent

from fast_agent.mcp.helpers.content_helpers import (
    canonicalize_tool_result_content_for_llm,
    get_text,
)
from fast_agent.tools.output_truncation import truncate_text_output
from fast_agent.utils.numeric import positive_int_or_none

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

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


def page_read_text_file_result(
    result: CallToolResult,
    *,
    tool_args: Mapping[str, Any],
    byte_limit: int,
) -> CallToolResult | None:
    """Cut an oversized file read at a line boundary and say where to continue.

    A head/tail cut of file content hides the middle without saying which lines are
    missing; a page of whole lines can be continued exactly with ``line``. Returns None
    when the result fits, is not a single text block, or its first line alone is too long.
    """

    if result.is_error or result.structured_content is not None or len(result.content) != 1:
        return None
    block = result.content[0]
    if not isinstance(block, TextContent):
        return None
    total_bytes = len(block.text.encode("utf-8"))
    if total_bytes <= byte_limit:
        return None

    lines = block.text.split("\n")
    if lines[-1] == "":
        lines.pop()
    kept: list[str] = []
    used_bytes = 0
    for line in lines:
        line_bytes = len(line.encode("utf-8")) + 1
        if used_bytes + line_bytes > byte_limit:
            break
        kept.append(line)
        used_bytes += line_bytes
    if not kept:
        return None

    first_line = positive_int_or_none(tool_args.get("line")) or 1
    last_shown = first_line + len(kept) - 1
    last_requested = first_line + len(lines) - 1
    notice = (
        f"[File read truncated: showing lines {first_line}-{last_shown} of "
        f"{first_line}-{last_requested} ({used_bytes} of {total_bytes} bytes). "
        f"Continue with line={last_shown + 1}, or use line and limit for a narrower range.]"
    )
    text = "\n".join(kept) + "\n" + notice
    return result.model_copy(update={"content": [TextContent(type="text", text=text)]})
