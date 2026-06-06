from __future__ import annotations

import math

from fast_agent.utils.numeric import (
    finite_number_or_none,
    int_or_none,
    nonnegative_int_or_none,
    nonnegative_number_or_none,
    positive_int_or_none,
    positive_number_or_none,
    sorted_unique_positive_ints,
)


def test_finite_number_or_none_accepts_ints_and_floats() -> None:
    assert finite_number_or_none(0) == 0
    assert finite_number_or_none(1.5) == 1.5


def test_finite_number_or_none_rejects_bool_and_non_finite_values() -> None:
    assert finite_number_or_none(True) is None
    assert finite_number_or_none(False) is None
    assert finite_number_or_none(math.inf) is None
    assert finite_number_or_none(math.nan) is None
    assert finite_number_or_none("1") is None


def test_positive_number_or_none_requires_value_above_zero() -> None:
    assert positive_number_or_none(0) is None
    assert positive_number_or_none(-1) is None
    assert positive_number_or_none(0.5) == 0.5


def test_nonnegative_number_or_none_allows_zero() -> None:
    assert nonnegative_number_or_none(-0.1) is None
    assert nonnegative_number_or_none(0) == 0
    assert nonnegative_number_or_none(2) == 2


def test_nonnegative_int_or_none_requires_non_bool_ints() -> None:
    assert nonnegative_int_or_none(True) is None
    assert nonnegative_int_or_none(1.5) is None
    assert nonnegative_int_or_none(-1) is None
    assert nonnegative_int_or_none(0) == 0
    assert nonnegative_int_or_none(2) == 2


def test_int_or_none_accepts_negative_zero_and_positive_non_bool_ints() -> None:
    assert int_or_none(True) is None
    assert int_or_none(1.5) is None
    assert int_or_none("1") is None
    assert int_or_none(-1) == -1
    assert int_or_none(0) == 0
    assert int_or_none(2) == 2


def test_positive_int_or_none_requires_positive_non_bool_ints() -> None:
    assert positive_int_or_none(False) is None
    assert positive_int_or_none(1.5) is None
    assert positive_int_or_none(0) is None
    assert positive_int_or_none(3) == 3


def test_sorted_unique_positive_ints_normalizes_budget_like_values() -> None:
    assert sorted_unique_positive_ints([20_000, 0, True, 1_000, 1_000, -5]) == [
        1_000,
        20_000,
    ]
