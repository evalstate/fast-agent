"""Common time and duration helpers."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Return a concise, human-friendly duration string."""
    seconds = max(seconds, 0.0)
    if seconds < 60:
        return f"{seconds:.2f}s"

    total_seconds = int(round(seconds))
    minutes, sec = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"

    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"

    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h {minutes}m"
