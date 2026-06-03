"""Common time and duration helpers."""

from __future__ import annotations

import math

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR

_TWO_UNIT_DURATION_FORMATS: tuple[tuple[int, str, int, str], ...] = (
    (SECONDS_PER_DAY, "d", SECONDS_PER_HOUR, "h"),
    (SECONDS_PER_HOUR, "h", SECONDS_PER_MINUTE, "m"),
    (SECONDS_PER_MINUTE, "m", 1, "s"),
)


def _valid_duration_seconds(seconds: float | None) -> float | None:
    if seconds is None or isinstance(seconds, bool) or not math.isfinite(seconds):
        return None
    return max(seconds, 0.0)


def _normalized_duration_seconds(seconds: float) -> float:
    return _valid_duration_seconds(seconds) or 0.0


def format_duration(seconds: float) -> str:
    """Return a concise, human-friendly duration string."""
    seconds = _normalized_duration_seconds(seconds)
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds:.2f}s"

    total_seconds = round(seconds)
    minutes, sec = divmod(total_seconds, SECONDS_PER_MINUTE)
    if total_seconds < SECONDS_PER_HOUR:
        return f"{minutes}m {sec:02d}s"

    hours, minutes = divmod(minutes, 60)
    if total_seconds < SECONDS_PER_DAY:
        return f"{hours}h {minutes:02d}m"

    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h {minutes}m"


def format_compact_duration(seconds: float | None) -> str | None:
    """Return a compact duration such as ``1m05s`` for dense status displays."""
    value = _valid_duration_seconds(seconds)
    if value is None:
        return None

    total = int(value)
    if total < 1:
        return "<1s"
    return format_two_unit_duration(total)


def format_two_unit_duration(total_seconds: int) -> str:
    """Return a compact label using the largest one or two time units."""
    total = max(0, int(total_seconds))
    if total == 0:
        return "0s"

    for primary_seconds, primary_suffix, secondary_seconds, secondary_suffix in _TWO_UNIT_DURATION_FORMATS:
        label = _format_two_unit_label(
            total,
            primary_seconds=primary_seconds,
            primary_suffix=primary_suffix,
            secondary_seconds=secondary_seconds,
            secondary_suffix=secondary_suffix,
            pad_secondary=secondary_seconds < 3600,
        )
        if label is not None:
            return label

    return f"{total}s"


def _format_two_unit_label(
    total_seconds: int,
    *,
    primary_seconds: int,
    primary_suffix: str,
    secondary_seconds: int,
    secondary_suffix: str,
    pad_secondary: bool,
) -> str | None:
    primary, remainder = divmod(total_seconds, primary_seconds)
    if primary == 0:
        return None

    secondary = remainder // secondary_seconds
    if secondary == 0:
        return f"{primary}{primary_suffix}"

    secondary_text = f"{secondary:02d}" if pad_secondary else str(secondary)
    return f"{primary}{primary_suffix}{secondary_text}{secondary_suffix}"
