"""Small numeric normalization helpers shared by display surfaces."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def finite_number_or_none(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and math.isfinite(value):
        return value
    return None


def positive_number_or_none(value: object) -> int | float | None:
    number = finite_number_or_none(value)
    if number is not None and number > 0:
        return number
    return None


def nonnegative_number_or_none(value: object) -> int | float | None:
    number = finite_number_or_none(value)
    if number is not None and number >= 0:
        return number
    return None


def nonnegative_int_or_none(value: object) -> int | None:
    number = int_or_none(value)
    if number is not None and number >= 0:
        return number
    return None


def int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def positive_int_or_none(value: object) -> int | None:
    number = nonnegative_int_or_none(value)
    if number is not None and number > 0:
        return number
    return None


def sorted_unique_positive_ints(values: Iterable[object]) -> list[int]:
    return sorted(
        {parsed for value in values if (parsed := positive_int_or_none(value)) is not None}
    )
