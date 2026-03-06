from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


_SENTENCE_PUNCTUATION = ".!?;:"
_MARKDOWN_OR_QUOTE_PREFIXES = "\"'`*_["
_CLOSING_DELIMITERS = ")]}\"'"


def _needs_reasoning_separator(existing: str, incoming: str) -> bool:
    if not existing or not incoming:
        return False

    prev = existing[-1]
    nxt = incoming[0]

    if prev.isspace() or nxt.isspace():
        return False

    if prev.islower() and nxt.isupper():
        return True

    if prev.isdigit() and nxt.isupper():
        return True

    if prev in _SENTENCE_PUNCTUATION and (nxt.isupper() or nxt in _MARKDOWN_OR_QUOTE_PREFIXES):
        return True

    if prev in _CLOSING_DELIMITERS and nxt.isupper():
        return True

    if nxt in _MARKDOWN_OR_QUOTE_PREFIXES and (prev.isalnum() or prev in _SENTENCE_PUNCTUATION):
        return True

    return False


def append_reasoning_chunk(existing: str, incoming: str) -> str:
    if not existing:
        return incoming
    if not incoming:
        return existing
    if _needs_reasoning_separator(existing, incoming):
        return f"{existing} {incoming}"
    return existing + incoming


def join_reasoning_chunks(chunks: Sequence[str]) -> str:
    combined = ""
    for chunk in chunks:
        if not chunk:
            continue
        combined = append_reasoning_chunk(combined, chunk)
    return combined


def normalize_reasoning_delta(existing: str, incoming: str) -> str:
    if not incoming:
        return ""
    combined = append_reasoning_chunk(existing, incoming)
    return combined[len(existing) :]
