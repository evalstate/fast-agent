"""Small collection helpers."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import TypeVar, overload

H = TypeVar("H", bound=Hashable)
T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


@overload
def unique_preserve_order(values: Iterable[H]) -> list[H]: ...


@overload
def unique_preserve_order(
    values: Iterable[T],
    *,
    key: Callable[[T], K],
) -> list[T]: ...


def unique_preserve_order(
    values: Iterable[T],
    *,
    key: Callable[[T], K] | None = None,
) -> list[T]:
    if key is None:
        return list(dict.fromkeys(values))

    unique: list[T] = []
    seen: set[K] = set()
    for value in values:
        marker = key(value)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(value)
    return unique


def cycle_next(
    current: T | None,
    candidates: Sequence[T],
    *,
    default: T | None = None,
) -> T | None:
    if not candidates:
        return None

    effective_current = current if current is not None else default
    if effective_current is None:
        return candidates[0]

    try:
        current_index = candidates.index(effective_current)
    except ValueError:
        return candidates[0]
    return candidates[(current_index + 1) % len(candidates)]
