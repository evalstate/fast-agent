from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fast_agent.types.streaming import StreamingMode, normalize_streaming_mode


@dataclass(frozen=True, slots=True)
class StreamingPreferences:
    enabled: bool
    mode: StreamingMode


def resolve_streaming_preferences(logger_settings: object) -> StreamingPreferences:
    streaming_mode = normalize_streaming_mode(getattr(logger_settings, "streaming", None))

    if streaming_mode == "markdown" and _legacy_bool_setting(
        logger_settings,
        "streaming_plain_text",
        default=False,
    ):
        streaming_mode = "plain"

    enabled = (
        _legacy_bool_setting(logger_settings, "show_chat", default=True)
        and _legacy_bool_setting(logger_settings, "streaming_display", default=True)
        and streaming_mode != "none"
    )
    return StreamingPreferences(enabled=enabled, mode=streaming_mode)


def _legacy_bool_setting(
    logger_settings: object,
    name: str,
    *,
    default: bool,
) -> bool:
    value: Any = getattr(logger_settings, name, default)
    return value if isinstance(value, bool) else default
