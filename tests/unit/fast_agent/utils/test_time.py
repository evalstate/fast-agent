"""Tests for the time utility helpers."""

from fast_agent.utils.time import format_duration


def test_format_duration_does_not_crash_and_returns_strings() -> None:
    inputs = [-5.0, 0.0, 0.4, 59.9, 60.0, 1234.5, 3600.0, 86_401.0]
    for value in inputs:
        result = format_duration(value)
        assert isinstance(result, str)
        assert result  # non-empty string


def test_format_duration_thresholds() -> None:
    assert format_duration(0.5) == "0.50s"
    assert format_duration(60.0) == "1m 00s"
    assert format_duration(3599.0) == "59m 59s"
    assert format_duration(3600.0) == "1h 00m"
    assert format_duration(90061.0) == "1d 1h 1m"
