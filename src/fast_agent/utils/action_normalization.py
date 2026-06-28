"""Shared helpers for user-facing action aliases."""

from __future__ import annotations

from typing import Literal

from fast_agent.utils.text import strip_casefold, strip_to_none

HELP_ACTION_ALIASES = frozenset({"help", "--help", "-h"})
CANCEL_ACTION_ALIASES = frozenset({"q", "quit", "exit"})
TRUE_WORD_BOOLEAN_ALIASES = frozenset({"true", "on", "yes", "enable", "enabled"})
FALSE_WORD_BOOLEAN_ALIASES = frozenset({"false", "off", "no", "disable", "disabled"})
TRUE_ACTION_ALIASES = frozenset((*TRUE_WORD_BOOLEAN_ALIASES, "1"))
FALSE_ACTION_ALIASES = frozenset((*FALSE_WORD_BOOLEAN_ALIASES, "0"))


def normalize_action_token(value: str | None) -> str:
    return strip_casefold(value) if value is not None else ""


def is_help_flag(value: str | None) -> bool:
    return normalize_action_token(value) in HELP_ACTION_ALIASES


def is_cancel_action(value: str | None) -> bool:
    return normalize_action_token(value) in CANCEL_ACTION_ALIASES


def parse_boolean_alias(value: str, *, numeric: bool = True) -> bool | None:
    normalized = normalize_action_token(value)
    true_aliases = TRUE_ACTION_ALIASES if numeric else TRUE_WORD_BOOLEAN_ALIASES
    false_aliases = FALSE_ACTION_ALIASES if numeric else FALSE_WORD_BOOLEAN_ALIASES
    if normalized in true_aliases:
        return True
    if normalized in false_aliases:
        return False
    return None


def on_off_label(value: bool) -> Literal["on", "off"]:
    return "on" if value else "off"


def enabled_disabled_label(value: bool) -> Literal["enabled", "disabled"]:
    return "enabled" if value else "disabled"


def split_first_token(
    text: str | None,
    *,
    default_token: str | None = None,
) -> tuple[str | None, str]:
    stripped = strip_to_none(text)
    if stripped is None:
        return default_token, ""
    parts = stripped.split(maxsplit=1)
    token = parts[0]
    remainder = parts[1] if len(parts) > 1 else ""
    return token, strip_to_none(remainder) or ""


def split_action_arguments(
    arguments: str | None,
    *,
    default_action: str | None = None,
) -> tuple[str | None, str]:
    return split_first_token(arguments, default_token=default_action)
