"""Task-budget parsing and formatting helpers."""

from __future__ import annotations

import re

TASK_BUDGET_MIN_TOKENS = 20_000
_DISABLED_TASK_BUDGET_VALUES = frozenset({"off", "none", "default", "unset"})
_TASK_BUDGET_PATTERN = re.compile(r"^\s*(\d+)\s*([km]?)\s*$", re.IGNORECASE)


def parse_task_budget_tokens(value: str | int | None) -> int | None:
    """Parse task-budget input into a token count.

    Supported forms:
    - ``None`` -> ``None``
    - integers -> unchanged
    - ``20k`` / ``128k`` / ``1m``
    - ``off`` / ``none`` / ``default`` / ``unset`` -> ``None``
    """

    if value is None:
        return None
    if isinstance(value, int):
        return value

    cleaned = value.strip().lower()
    if not cleaned or cleaned in _DISABLED_TASK_BUDGET_VALUES:
        return None

    match = _TASK_BUDGET_PATTERN.fullmatch(cleaned)
    if match is None:
        raise ValueError(
            "Task budget must be an integer token count or shorthand like 20k/128k/1m."
        )

    amount = int(match.group(1))
    suffix = match.group(2).lower()
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000}[suffix]
    return amount * multiplier


def validate_task_budget_tokens(value: int | None) -> int | None:
    """Validate Anthropic task-budget limits."""

    if value is None:
        return None
    if value < TASK_BUDGET_MIN_TOKENS:
        raise ValueError(f"Task budget must be at least {TASK_BUDGET_MIN_TOKENS:,} tokens.")
    return value


def format_task_budget_tokens(value: int | None) -> str:
    """Return a compact display form for task-budget tokens."""

    if value is None:
        return "default"
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)

