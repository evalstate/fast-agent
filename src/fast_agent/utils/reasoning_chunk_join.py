from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

_SENTENCE_PUNCTUATION = ".!?;:"
_MARKDOWN_PREFIXES = "\"`*["
ReasoningDeltaNormalizer = Callable[[str | None, str], str]


def _looks_like_markdown_block_heading(incoming: str) -> bool:
    if not incoming.startswith("**"):
        return False
    closing_index = incoming.find("**", 2)
    if closing_index <= 2:
        return False
    suffix = incoming[closing_index + 2 :]
    return suffix.startswith("\n")


def _looks_like_sentence_chunk(incoming: str) -> bool:
    if not incoming:
        return False
    if " " not in incoming:
        return False
    first = incoming[0]
    return first.isupper() or first in _MARKDOWN_PREFIXES


def identity_reasoning_delta(last_char: str | None, incoming: str) -> str:
    del last_char
    return incoming


def normalize_reasoning_delta(last_char: str | None, incoming: str) -> str:
    """Normalize one reasoning delta without rebuilding the full accumulated text.

    Keep the Codex-style append-only flow, but patch the specific broken case where
    providers split natural-language reasoning into sentence chunks without a
    separating space, e.g. "approach." + "Specifying session retrieval format".
    """
    if not incoming:
        return ""
    if not last_char or last_char.isspace() or incoming[0].isspace():
        return incoming
    if _looks_like_markdown_block_heading(incoming):
        return f"\n\n{incoming}"
    if last_char in _SENTENCE_PUNCTUATION and _looks_like_sentence_chunk(incoming):
        return f" {incoming}"
    if last_char.islower() and _looks_like_sentence_chunk(incoming):
        return f" {incoming}"
    return incoming


@dataclass(slots=True)
class ReasoningTextAccumulator:
    normalizer: ReasoningDeltaNormalizer = identity_reasoning_delta
    _parts: list[str] = field(default_factory=list)
    _last_char: str | None = None

    def append(self, incoming: str) -> str:
        normalized = self.normalizer(self._last_char, incoming)
        if normalized:
            self._parts.append(normalized)
            self._last_char = normalized[-1]
        return normalized

    def extend(self, incoming_parts: Sequence[str]) -> None:
        for incoming in incoming_parts:
            self.append(incoming)

    def text(self) -> str:
        return "".join(self._parts)

    def parts(self) -> list[str]:
        return list(self._parts)


def join_reasoning_segments(
    parts: Sequence[str],
    *,
    normalizer: ReasoningDeltaNormalizer = normalize_reasoning_delta,
) -> str:
    accumulator = ReasoningTextAccumulator(normalizer=normalizer)
    accumulator.extend(parts)
    return accumulator.text()
