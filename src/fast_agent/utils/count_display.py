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


def format_compact_count(count: int) -> str:
    """Group counts below one million and abbreviate larger values to three significant digits."""

    magnitude = abs(count)
    if magnitude < 1_000_000:
        return f"{count:,}"

    units = ((1_000_000, "M"), (1_000_000_000, "B"), (1_000_000_000_000, "T"))
    unit_index = 0
    for index, (candidate_divisor, _) in enumerate(units[1:], start=1):
        if magnitude < candidate_divisor:
            break
        unit_index = index

    divisor, suffix = units[unit_index]
    scaled = count / divisor
    decimals = 2 if abs(scaled) < 10 else 1 if abs(scaled) < 100 else 0
    if round(abs(scaled), decimals) >= 1_000 and unit_index < len(units) - 1:
        divisor, suffix = units[unit_index + 1]
        scaled = count / divisor
        decimals = 2
    return f"{scaled:.{decimals}f}{suffix}"


def format_count_breakdown(label: str, total: int, **parts: int) -> str:
    if not parts:
        return f"{label}: {total:,}"
    breakdown = ", ".join(f"{name}: {value:,}" for name, value in parts.items())
    return f"{label}: {total:,} ({breakdown})"
