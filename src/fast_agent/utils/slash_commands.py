"""Helpers for slash-command routing."""

from __future__ import annotations

from fast_agent.utils.action_normalization import split_first_token
from fast_agent.utils.text import strip_to_none


def parse_slash_command_line(text: str) -> tuple[str, str] | None:
    """Return slash command name and arguments, or None for non-slash input."""

    stripped = strip_to_none(text)
    if stripped is None or not stripped.startswith("/"):
        return None
    command_name, remainder = split_first_token(stripped[1:], default_token="")
    return command_name or "", remainder


def split_subcommand_and_remainder(text: str) -> tuple[str, str]:
    subcommand, remainder = split_first_token(text, default_token="")
    return subcommand or "", remainder
