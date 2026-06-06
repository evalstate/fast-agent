"""Shared selector helpers for marketplace-style lists."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

from fast_agent.utils.action_normalization import normalize_action_token

T = TypeVar("T")

SelectorNames = Callable[[T], Iterable[str]]


def _one_based_index(selector: str, item_count: int) -> int | None:
    if not selector.isdigit():
        return None
    index = int(selector)
    if 1 <= index <= item_count:
        return index - 1
    return None


def _normalize_selector(value: str) -> str:
    return normalize_action_token(value)


def select_one_by_name_or_index(
    entries: Iterable[T],
    selector: str,
    *,
    names: SelectorNames[T],
) -> T | None:
    selector_clean = selector.strip()
    if not selector_clean:
        return None

    entries_list = list(entries)
    index = _one_based_index(selector_clean, len(entries_list))
    if index is not None:
        return entries_list[index]

    selector_lower = _normalize_selector(selector_clean)
    matches = [
        entry
        for entry in entries_list
        if any(_normalize_selector(name) == selector_lower for name in names(entry))
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def select_updates_by_name_or_index(
    updates: Sequence[T],
    selector: str,
    *,
    names: SelectorNames[T],
) -> list[T]:
    selector_clean = selector.strip()
    if not selector_clean:
        return []
    if _normalize_selector(selector_clean) == "all":
        return list(updates)

    selected = select_one_by_name_or_index(updates, selector_clean, names=names)
    return [selected] if selected is not None else []
