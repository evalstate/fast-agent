"""Context blocks fast-agent adds to user messages for the model, not typed by the user."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.ptc.catalog import TOOL_SOURCES_FINGERPRINT_META
from fast_agent.ptc.mentions import TOOL_DEFINITIONS_META

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mcp_types import ContentBlock


def is_generated_ptc_context_block(block: ContentBlock) -> bool:
    """Whether ``block`` is a generated tool-source roster or ``^tool:`` definitions block."""
    meta = block.meta
    return bool(meta) and (TOOL_SOURCES_FINGERPRINT_META in meta or TOOL_DEFINITIONS_META in meta)


def user_authored_content(blocks: Iterable[ContentBlock]) -> list[ContentBlock]:
    """``blocks`` without generated PTC context blocks."""
    return [block for block in blocks if not is_generated_ptc_context_block(block)]


def user_authored_text(blocks: Iterable[ContentBlock]) -> str:
    """Newline-joined text of ``blocks``, skipping generated PTC context blocks."""
    return "\n".join(
        text for block in user_authored_content(blocks) if (text := get_text(block)) is not None
    )
