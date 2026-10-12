"""How an LLM can compact conversation history.

The harness owns compaction policy (when to compact, which turns to keep,
archiving). Providers declare how the summarization step can be performed.
"""

from __future__ import annotations

import json
import math
import re
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


DEFAULT_COMPACTION_THRESHOLD = 0.85
"""Auto-compaction threshold when neither the model string, config nor model sets one."""

_COMPACT_AT_TOKENS = re.compile(r"(\d+)([km]?)")
COMPACT_AT_MIN_TOKENS = 1_000


@dataclass(frozen=True, slots=True)
class CompactAt:
    """When auto-compaction fires: at an input-token count, or a context-window fraction."""

    tokens: int | None = None
    fraction: float | None = None

    def __post_init__(self) -> None:
        if (self.tokens is None) == (self.fraction is None):
            raise ValueError("CompactAt takes exactly one of tokens or fraction")

    @classmethod
    def parse(cls, value: str) -> CompactAt:
        """``190k``, ``1m`` or ``190000`` tokens; ``0.4`` or ``40%`` of the window."""
        cleaned = value.strip().lower().replace("_", "")
        if match := _COMPACT_AT_TOKENS.fullmatch(cleaned):
            tokens = int(match.group(1)) * {"": 1, "k": 1_000, "m": 1_000_000}[match.group(2)]
            if tokens >= COMPACT_AT_MIN_TOKENS:
                return cls(tokens=tokens)
        else:
            try:
                fraction = float(cleaned[:-1]) / 100 if cleaned.endswith("%") else float(cleaned)
            except ValueError:
                fraction = 0.0
            if 0 < fraction <= 1:
                return cls(fraction=fraction)
        raise ValueError(
            f"Invalid compact_at value '{value}': use tokens (190k, 1m, at least "
            f"{COMPACT_AT_MIN_TOKENS:,}) or a context-window fraction (0.4, 40%)"
        )

    def trigger_tokens(self, context_window: int | None) -> int | None:
        """Input tokens that trigger compaction; None when a fraction has no window."""
        if self.tokens is not None:
            return self.tokens
        assert self.fraction is not None
        if context_window is None or context_window <= 0:
            return None
        # Smallest count whose share of the window reaches the fraction (no float drift).
        tokens = math.ceil(context_window * self.fraction)
        return tokens - 1 if (tokens - 1) / context_window >= self.fraction else tokens

    def describe(self, context_window: int | None) -> str:
        """E.g. ``190,000 tokens (38% of context)`` or ``85% of context``."""
        tokens = self.trigger_tokens(context_window)
        window = context_window if context_window is not None and context_window > 0 else None
        if self.fraction is not None:
            suffix = f" ({tokens:,} tokens)" if tokens is not None else ""
            return f"{self.fraction * 100:.0f}% of context{suffix}"
        suffix = f" ({tokens / window * 100:.0f}% of context)" if tokens and window else ""
        return f"{tokens:,} tokens{suffix}"


def resolve_compact_at(
    llm: FastAgentLLMProtocol | None,
    settings: CompactionSettings,
) -> CompactAt:
    """The model string's ``compact_at``, else ``compaction.threshold``, else the
    model's default, else 85% of the context window."""
    if llm is not None and llm.compact_at_override is not None:
        return llm.compact_at_override
    if settings.threshold is not None:
        return CompactAt(fraction=settings.threshold)
    if llm is not None and llm.default_compact_at is not None:
        return llm.default_compact_at
    return CompactAt(fraction=DEFAULT_COMPACTION_THRESHOLD)


def compaction_trigger_tokens(
    llm: FastAgentLLMProtocol | None,
    settings: CompactionSettings,
    context_window: int | None,
) -> int | None:
    """Input tokens at which auto-compaction fires, or None when unknown."""
    return resolve_compact_at(llm, settings).trigger_tokens(context_window)


# The API rejects compaction triggers below this many input tokens.
COMPACTION_MIN_TRIGGER_TOKENS = 50_000


def in_request_compaction_trigger(
    llm: FastAgentLLMProtocol | None,
    settings: CompactionSettings,
) -> int | None:
    """Effective provider trigger, or None when harness compaction should run."""
    if llm is None or llm.compaction_support is not CompactionSupport.IN_REQUEST:
        return None
    if not settings.auto or settings.strategy == "client" or settings.has_custom_prompt:
        return None
    usage = llm.usage_accumulator
    trigger = compaction_trigger_tokens(llm, settings, usage.context_window_size if usage else None)
    return max(trigger, COMPACTION_MIN_TRIGGER_TOKENS) if trigger is not None else None
