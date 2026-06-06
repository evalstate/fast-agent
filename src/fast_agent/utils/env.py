"""Environment variable parsing helpers."""

from __future__ import annotations

import os

from fast_agent.utils.action_normalization import normalize_action_token

TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def is_truthy_env_value(value: str | None) -> bool:
    return normalize_action_token(value) in TRUTHY_ENV_VALUES


def _optional_env_flag_value(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    return is_truthy_env_value(value)


def env_flag(name: str, *, default: bool = False) -> bool:
    value = _optional_env_flag_value(name)
    return default if value is None else value


def optional_env_flag(name: str) -> bool | None:
    return _optional_env_flag_value(name)
