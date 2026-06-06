"""Small normalization helpers shared by command summary builders."""

from __future__ import annotations

from collections.abc import Mapping

from fast_agent.utils.text import strip_to_none

type JsonObject = dict[str, object]


def json_object(value: object) -> JsonObject:
    if not isinstance(value, Mapping):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return strip_to_none(value)
