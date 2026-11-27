"""Context blocks fast-agent adds to user messages for the model, not typed by the user."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from mcp_types import TextContent

from fast_agent.constants import (
    FAST_AGENT_COMPACTION_CHANNEL,
    FAST_AGENT_PROVIDER_CHECKPOINT,
    MCP_EVENTS_BLOCK_META,
    OPENAI_COMPACTION_ITEMS,
)
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.ptc.catalog import TOOL_SOURCES_FINGERPRINT_META
from fast_agent.ptc.mentions import ORIGINAL_TEXT_META, TOOL_DEFINITIONS_META
from fast_agent.ptc.vocabulary import PTC_VOCABULARY_META

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from mcp_types import ContentBlock

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


_CHECKPOINT_CHANNELS: Final = frozenset(
    {FAST_AGENT_COMPACTION_CHANNEL, FAST_AGENT_PROVIDER_CHECKPOINT, OPENAI_COMPACTION_ITEMS}
)
_SIGNATURE_PREFIX = "async def "


def recent_ptc_messages(
    messages: Sequence[PromptMessageExtended],
) -> Iterator[PromptMessageExtended]:
    """Walk backwards through messages still available in full after compaction."""
    for message in reversed(messages):
        if not _CHECKPOINT_CHANNELS.isdisjoint(message.channels or {}):
            break
        yield message


def seen_tool_signatures(messages: Sequence[PromptMessageExtended]) -> frozenset[str]:
    """Read signatures from sent mentions and visible sandbox results, not nested call logs.

    A search whose result was discarded or truncated away does not reveal a signature.
    """
    signatures: set[str] = set()

    def collect(block: TextContent) -> None:
        signatures.update(
            line.removeprefix(_SIGNATURE_PREFIX).removesuffix(":")
            for line in block.text.splitlines()
            if line.startswith(_SIGNATURE_PREFIX)
        )

    for message in recent_ptc_messages(messages):
        for block in message.content:
            if isinstance(block, TextContent) and TOOL_DEFINITIONS_META in (block.meta or {}):
                collect(block)
        for result in (message.tool_results or {}).values():
            if PTC_VOCABULARY_META in (result.meta or {}):
                for block in result.content:
                    if isinstance(block, TextContent):
                        collect(block)
    return frozenset(signatures)


def is_generated_ptc_context_block(block: ContentBlock) -> bool:
    """Whether ``block`` is a generated context block (PTC roster/definitions or MCP events)."""
    meta = block.meta
    return bool(meta) and (
        TOOL_SOURCES_FINGERPRINT_META in meta
        or TOOL_DEFINITIONS_META in meta
        or MCP_EVENTS_BLOCK_META in meta
    )


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
