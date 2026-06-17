"""Shared timing display helpers."""

from __future__ import annotations

from fast_agent.utils.numeric import nonnegative_number_or_none

MISSING_TIMING_DISPLAY = "-"


def format_duration_ms(value: object) -> str:
    value = nonnegative_number_or_none(value)
    if value is None:
        return MISSING_TIMING_DISPLAY
    rounded_ms = round(value)
    if rounded_ms < 1000:
        return f"{rounded_ms}ms"
    return f"{value / 1000:.1f}s"


def format_rate_per_second(value: object) -> str:
    value = nonnegative_number_or_none(value)
    if value is None:
        return MISSING_TIMING_DISPLAY
    return f"{value:.1f}"
