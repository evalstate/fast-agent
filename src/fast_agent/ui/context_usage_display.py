"""Shared compact context-usage helpers for UI surfaces."""

from __future__ import annotations

import math
from typing import Protocol

from fast_agent.utils.numeric import (
    finite_number_or_none,
    nonnegative_number_or_none,
    positive_number_or_none,
)


class ContextUsageAccumulator(Protocol):
    @property
    def context_window_size(self) -> int | float | None: ...

    @property
    def current_context_tokens(self) -> int | float | None: ...


def normalize_context_usage_percent(value: object) -> float | None:
    number = finite_number_or_none(value)
    if number is None:
        return None
    return float(number)


def _positive_number_or_fallback(
    value: object,
    fallback: int | float | None,
) -> int | float | None:
    number = positive_number_or_none(value)
    return number if number is not None else positive_number_or_none(fallback)


def resolve_context_usage_percent(
    *,
    context_pct: float | None,
    usage_accumulator: ContextUsageAccumulator | None,
    fallback_window_size: int | float | None = None,
) -> float | None:
    """Resolve context usage percent from an accumulator when needed."""
    if context_pct is not None or usage_accumulator is None:
        return context_pct

    try:
        raw_window_size = usage_accumulator.context_window_size
        raw_current_context_tokens = usage_accumulator.current_context_tokens
    except AttributeError:
        return None

    window_size = _positive_number_or_fallback(raw_window_size, fallback_window_size)
    if window_size is None:
        return None

    current_context_tokens = nonnegative_number_or_none(raw_current_context_tokens)
    if current_context_tokens is None:
        return None
    return (current_context_tokens / window_size) * 100


def format_compact_context_usage_percent(pct: float | None) -> str | None:
    """Format context usage with stable width for compact displays."""
    if pct is None or not math.isfinite(pct):
        return None

    safe_pct = max(pct, 0.0)
    if safe_pct >= 100.0:
        return "100%+"
    if safe_pct < 10.0:
        return f"{min(safe_pct, 9.99):.2f}%"
    return f"{min(safe_pct, 99.9):.1f}%"
