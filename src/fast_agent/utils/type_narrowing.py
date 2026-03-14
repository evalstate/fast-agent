"""Helpers for narrowing dynamic values to concrete typed containers."""

from __future__ import annotations

from typing import TypeGuard


def is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    """Return ``True`` when ``value`` is a dict with string keys."""

    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in value)
