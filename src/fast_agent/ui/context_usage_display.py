"""Shared compact context-usage helpers for UI surfaces."""

from __future__ import annotations


def resolve_context_usage_percent(
    *,
    context_pct: float | None,
    usage_accumulator: object | None,
    fallback_window_size: int | float | None = None,
) -> float | None:
    """Resolve context usage percent from an accumulator when needed."""
    if context_pct is not None or usage_accumulator is None:
        return context_pct

    try:
        window_size = getattr(usage_accumulator, "context_window_size", None)
        if not isinstance(window_size, (int, float)) or window_size <= 0:
            window_size = fallback_window_size
        if not isinstance(window_size, (int, float)) or window_size <= 0:
            return None

        current_context_tokens = getattr(usage_accumulator, "current_context_tokens", None)
        if not isinstance(current_context_tokens, (int, float)):
            return None
        return (current_context_tokens / window_size) * 100
    except Exception:
        return None


def format_compact_context_usage_percent(pct: float | None) -> str | None:
    """Format context usage with stable width for compact displays."""
    if pct is None:
        return None

    safe_pct = max(pct, 0.0)
    if safe_pct >= 100.0:
        return "100%+"
    if safe_pct < 10.0:
        return f"{min(safe_pct, 9.99):.2f}%"
    return f"{min(safe_pct, 99.9):.1f}%"
