"""Shared count display helpers."""

from __future__ import annotations


def plural_label(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or f"{singular}s")


def format_count(count: int, singular: str, plural: str | None = None) -> str:
    count_text, label = format_count_parts(count, singular, plural)
    return f"{count_text} {label}"


def format_count_parts(
    count: int,
    singular: str,
    plural: str | None = None,
) -> tuple[str, str]:
    return f"{count:,}", plural_label(count, singular, plural)


def format_count_breakdown(label: str, total: int, **parts: int) -> str:
    if not parts:
        return f"{label}: {total:,}"
    breakdown = ", ".join(f"{name}: {value:,}" for name, value in parts.items())
    return f"{label}: {total:,} ({breakdown})"
