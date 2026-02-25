"""Formatting helpers shared across marketplace managers."""

from __future__ import annotations

from datetime import UTC, datetime


def format_revision_short(revision: str | None) -> str:
    if revision is None:
        return "?"
    trimmed = revision.strip()
    if not trimmed:
        return "?"
    normalized = trimmed.lower()
    if len(normalized) >= 8 and all(ch in "0123456789abcdef" for ch in normalized):
        return trimmed[:7]
    return trimmed


def format_installed_at_display(installed_at: str | None) -> str:
    if not installed_at:
        return "unknown"
    normalized = installed_at.strip()
    if not normalized:
        return "unknown"
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return installed_at
    return parsed.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
