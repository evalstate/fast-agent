from __future__ import annotations

from dataclasses import dataclass

from fast_agent.utils.collections import cycle_next, unique_preserve_order


def test_unique_preserve_order_keeps_first_occurrence_order() -> None:
    assert unique_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_unique_preserve_order_accepts_iterables() -> None:
    assert unique_preserve_order(value for value in (1, 2, 1, 3)) == [1, 2, 3]


def test_unique_preserve_order_accepts_key_function() -> None:
    assert unique_preserve_order(["Finder", "finder", "Other"], key=str.casefold) == [
        "Finder",
        "Other",
    ]


def test_unique_preserve_order_accepts_unhashable_values_with_key() -> None:
    @dataclass
    class Item:
        key: str
        value: int

    first = Item("same", 1)
    duplicate = Item("same", 2)
    other = Item("other", 3)

    assert unique_preserve_order([first, duplicate, other], key=lambda item: item.key) == [
        first,
        other,
    ]


def test_cycle_next_uses_default_and_wraps() -> None:
    assert cycle_next(None, ["low", "medium", "high"], default="medium") == "high"
    assert cycle_next("high", ["low", "medium", "high"]) == "low"


def test_cycle_next_falls_back_to_first_candidate() -> None:
    assert cycle_next(None, ["fast", "flex"]) == "fast"
    assert cycle_next("unsupported", ["fast", "flex"]) == "fast"
    assert cycle_next("fast", []) is None
