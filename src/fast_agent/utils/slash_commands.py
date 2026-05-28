"""Helpers for slash-command routing."""

from __future__ import annotations


def split_subcommand_and_remainder(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if not stripped:
        return "", ""

    parts = stripped.split(maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]
