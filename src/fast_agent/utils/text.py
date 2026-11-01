"""Shared text normalization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def split_lines(text: str, *, keepends: bool = False) -> list[str]:
    """Split text into lines at ``\\n`` (and ``\\r\\n``) only, like files, JSONL and shells.

    ``str.splitlines()`` also breaks at ``\\v``, ``\\f``, ``\\x1c``-``\\x1e``, ``\\x85``,
    ``\\u2028`` and ``\\u2029``, which can legitimately appear inside a line.
    """
    *lines, last = text.split("\n")
    split = (
        [f"{line}\n" for line in lines] if keepends else [line.removesuffix("\r") for line in lines]
    )
    if last:
        split.append(last)
    return split


def collapse_whitespace(value: str | None) -> str:
    return "" if not value else " ".join(value.split())


def summarize_command(command: str, *, limit: int = 64) -> str:
    """Collapse and truncate a shell command for dense status displays."""
    single_line = collapse_whitespace(command)
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[: limit - 1]}…"


def strip_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def strip_str_to_none(value: object) -> str | None:
    return strip_to_none(value) if isinstance(value, str) else None


def strip_casefold(value: str) -> str:
    return value.strip().casefold()


def casefold_text(value: str) -> str:
    return value.casefold()


def starts_with_casefold(value: str, prefix: str) -> bool:
    return value.casefold().startswith(prefix.casefold())


def format_english_list(items: Sequence[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])}, and {items[-1]}"
