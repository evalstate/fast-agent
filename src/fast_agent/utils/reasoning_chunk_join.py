from __future__ import annotations

_SENTENCE_PUNCTUATION = ".!?;:"
_MARKDOWN_PREFIXES = "\"`*["


def _looks_like_sentence_chunk(incoming: str) -> bool:
    if not incoming:
        return False
    if " " not in incoming:
        return False
    first = incoming[0]
    return first.isupper() or first in _MARKDOWN_PREFIXES


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
    if last_char in _SENTENCE_PUNCTUATION and _looks_like_sentence_chunk(incoming):
        return f" {incoming}"
    if last_char.islower() and _looks_like_sentence_chunk(incoming):
        return f" {incoming}"
    return incoming
