from __future__ import annotations

from typing import Literal, TypeGuard

from fast_agent.utils.text import strip_casefold

StreamingMode = Literal["markdown", "plain", "none"]

STREAMING_MODES: tuple[StreamingMode, ...] = ("markdown", "plain", "none")
STREAMING_MODE_HELP = "|".join(STREAMING_MODES)
DEFAULT_STREAMING_MODE: StreamingMode = "markdown"


def is_streaming_mode(value: object) -> TypeGuard[StreamingMode]:
    return value in STREAMING_MODES


def normalize_streaming_mode(value: object) -> StreamingMode:
    if isinstance(value, str):
        candidate = strip_casefold(value)
        if is_streaming_mode(candidate):
            return candidate
    return DEFAULT_STREAMING_MODE
