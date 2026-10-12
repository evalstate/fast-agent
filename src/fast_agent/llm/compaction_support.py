"""How an LLM can compact conversation history.

The harness owns compaction policy (when to compact, which turns to keep,
archiving). Providers declare how the summarization step can be performed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from mcp_types import TextContent

from fast_agent.constants import FAST_AGENT_COMPACTION_CHANNEL

if TYPE_CHECKING:
    from mcp_types import ContentBlock

    from fast_agent.config import CompactionSettings
    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.types import PromptMessageExtended


class CompactionStrategy(StrEnum):
    """Checkpoint strategies serialized in history metadata and ATIF."""

    SUMMARY = "summary"
    OPENAI_COMPACTION = "openai_compaction"


CompactionMode = Literal["native", "client"]
"""Model-string ``compaction=`` override: provider-native compaction, or fast-agent's
own summary. Unset means the route's default (``native_compaction`` setting)."""


class CompactionSupport(StrEnum):
    CLIENT = "client"
    """fast-agent summarizes the compact region with a side-channel LLM call."""

    ON_DEMAND = "on_demand"
    """fast-agent decides when; the provider produces a native compaction checkpoint."""

    IN_REQUEST = "in_request"
    """The provider compacts during normal requests; harness auto-compaction stands aside."""


@dataclass(frozen=True, slots=True)
class NativeCompaction:
    """A provider-produced checkpoint replacing the compacted region of history."""

    strategy: CompactionStrategy
    """Recorded in checkpoint metadata and ATIF, e.g. ``openai_compaction``."""

    request: str
    """Audit description of the compaction request (it has no prompt text)."""

    response: PromptMessageExtended
    """The provider response, carrying usage and timing channels."""

    checkpoint_channels: dict[str, list[ContentBlock]]
    """Provider channels the checkpoint message carries for replay."""

    description: str
    """Human-readable stand-in for the opaque checkpoint."""

    opaque_bytes: int
    """Size of the encrypted checkpoint payload, for display."""


@dataclass(frozen=True, slots=True)
class CompactionMetadata:
    """Display fields read from persisted checkpoint metadata.

    Keep strategy as a string: unknown provider strategies remain native checkpoints.
    """

    strategy: str | None
    checkpoint_bytes: int | None
    archive_file: str | None

    @property
    def is_native(self) -> bool:
        return self.strategy is not None and self.strategy != CompactionStrategy.SUMMARY.value


def read_compaction_metadata(message: PromptMessageExtended) -> CompactionMetadata | None:
    """Read display metadata, returning None for absent or malformed JSON.

    Reconstruction validates the full archive metadata separately and raises on errors.
    """
    blocks = (message.channels or {}).get(FAST_AGENT_COMPACTION_CHANNEL) or []
    if not blocks or not isinstance(blocks[0], TextContent):
        return None
    try:
        metadata = json.loads(blocks[0].text)
    except json.JSONDecodeError:
        return None
    if not isinstance(metadata, dict):
        return None
    strategy = metadata.get("strategy")
    checkpoint_bytes = metadata.get("checkpoint_bytes")
    archive_file = metadata.get("archive_file")
    return CompactionMetadata(
        strategy=strategy if isinstance(strategy, str) else None,
        checkpoint_bytes=checkpoint_bytes if isinstance(checkpoint_bytes, int) else None,
        archive_file=archive_file if isinstance(archive_file, str) else None,
    )


def provider_checkpoint_marker(provider: str) -> list[ContentBlock]:
    """Blocks for ``FAST_AGENT_PROVIDER_CHECKPOINT`` on a response carrying an in-band
    checkpoint. History rewrites (e.g. process-poll folding) must keep such messages.
    """
    return [TextContent(type="text", text=provider)]


def resolve_compaction_support(
    llm: FastAgentLLMProtocol | None,
    settings: CompactionSettings,
) -> CompactionSupport:
    """Resolve how compaction runs for this LLM under the configured strategy."""
    if llm is None or settings.strategy == "client":
        return CompactionSupport.CLIENT
    return llm.compaction_support


# The API rejects compaction triggers below this many input tokens.
COMPACTION_MIN_TRIGGER_TOKENS = 50_000


def in_request_compaction_trigger(
    llm: FastAgentLLMProtocol | None,
    settings: CompactionSettings,
) -> int | None:
    """Effective provider trigger, or None when harness compaction should run."""
    if llm is None or llm.compaction_support is not CompactionSupport.IN_REQUEST:
        return None
    usage = llm.usage_accumulator
    context_window = usage.context_window_size if usage else None
    if (
        not settings.auto
        or settings.strategy == "client"
        or settings.has_custom_prompt
        or context_window is None
        or context_window <= 0
    ):
        return None
    return max(int(context_window * settings.threshold), COMPACTION_MIN_TRIGGER_TOKENS)
