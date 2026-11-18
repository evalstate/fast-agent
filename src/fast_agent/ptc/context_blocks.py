"""Context blocks fast-agent adds to user messages for the model, not typed by the user."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp_types import TextContent

from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.ptc.catalog import TOOL_SOURCES_FINGERPRINT_META
from fast_agent.ptc.mentions import ORIGINAL_TEXT_META, TOOL_DEFINITIONS_META

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mcp_types import ContentBlock


def is_generated_ptc_context_block(block: ContentBlock) -> bool:
    """Whether ``block`` is a generated tool-source roster or ``^tool:`` definitions block."""
    meta = block.meta
    return bool(meta) and (TOOL_SOURCES_FINGERPRINT_META in meta or TOOL_DEFINITIONS_META in meta)


def user_authored_content(
    blocks: Iterable[ContentBlock], *, as_typed: bool = False
) -> list[ContentBlock]:
    """Skip generated blocks, optionally restoring typed text on a copy for display."""
    content: list[ContentBlock] = []
    for block in blocks:
        if is_generated_ptc_context_block(block):
            continue
        if as_typed and isinstance(block, TextContent):
            original = (block.meta or {}).get(ORIGINAL_TEXT_META)
            if isinstance(original, str) and original != block.text:
                block = block.model_copy(update={"text": original})
        content.append(block)
    return content


def user_authored_text(blocks: Iterable[ContentBlock]) -> str:
    """Newline-joined text as typed, for user-facing previews."""
    return "\n".join(
        text
        for block in user_authored_content(blocks, as_typed=True)
        if (text := get_text(block)) is not None
    )
