"""Tests for the time utility helpers."""

import pytest

from fast_agent.utils.time import format_compact_duration, format_duration, format_two_unit_duration


def test_format_duration_does_not_crash_and_returns_strings() -> None:
    inputs = [-5.0, 0.0, 0.4, 59.9, 60.0, 1234.5, 3600.0, 86_401.0]
    for value in inputs:
        result = format_duration(value)
        assert isinstance(result, str)
        assert result  # non-empty string


def test_format_duration_thresholds() -> None:
    assert format_duration(0.5) == "0.50s"
    assert format_duration(59.99) == "59.99s"
    assert format_duration(60.0) == "1m 00s"
    assert format_duration(3599.0) == "59m 59s"
    assert format_duration(3600.0) == "1h 00m"
    assert format_duration(86_400.0) == "1d 0h 0m"
    assert format_duration(90061.0) == "1d 1h 1m"


def test_format_duration_treats_non_finite_values_as_zero() -> None:
    assert format_duration(float("nan")) == "0.00s"
    assert format_duration(float("inf")) == "0.00s"
    assert format_duration(float("-inf")) == "0.00s"


def test_format_duration_treats_boolean_values_as_zero() -> None:
    assert format_duration(True) == "0.00s"
    assert format_duration(False) == "0.00s"


def test_format_compact_duration_omits_missing_and_non_finite_values() -> None:
    assert format_compact_duration(None) is None
    assert format_compact_duration(float("nan")) is None
    assert format_compact_duration(float("inf")) is None
    assert format_compact_duration(True) is None


def test_format_compact_duration_uses_dense_two_unit_display() -> None:
    assert format_compact_duration(0.5) == "<1s"
    assert format_compact_duration(65) == "1m05s"
    assert format_compact_duration(3700) == "1h01m"


@pytest.mark.parametrize(
    ("total_seconds", "expected"),
    [
        (0, "0s"),
        (-5, "0s"),
        (5, "5s"),
        (60, "1m"),
        (65, "1m05s"),
        (3600, "1h"),
        (3660, "1h01m"),
        (86400, "1d"),
        (90000, "1d1h"),
        (86400 + 59 * 60, "1d"),
    ],
)
def test_format_two_unit_duration_uses_largest_two_units(
    total_seconds: int,
    expected: str,
) -> None:
    assert format_two_unit_duration(total_seconds) == expected
