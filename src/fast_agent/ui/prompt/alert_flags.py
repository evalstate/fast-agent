"""Alert flag extraction helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp.types import ContentBlock, TextContent

from fast_agent.constants import (
    FAST_AGENT_ALERT_CHANNEL,
    FAST_AGENT_ERROR_CHANNEL,
    FAST_AGENT_REMOVED_METADATA_CHANNEL,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from fast_agent.types import PromptMessageExtended

type JsonObject = dict[str, object]

_ALERT_FLAGS = frozenset({"T", "D", "V"})
_CATEGORY_ALERT_FLAGS = {
    "text": "T",
    "document": "D",
    "vision": "V",
}


def _json_text_payload(block: object) -> JsonObject | None:
    if not isinstance(block, TextContent) or not block.text:
        return None
    try:
        payload = json.loads(block.text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return {key: value for key, value in payload.items() if isinstance(key, str)}


def _string_field(payload: JsonObject, key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _string_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _category_alert_flag(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return _CATEGORY_ALERT_FLAGS.get(value)


def _iter_json_payloads_of_type(
    blocks: Sequence[ContentBlock] | None,
    payload_type: str,
) -> Iterator[JsonObject]:
    for block in blocks or []:
        payload = _json_text_payload(block)
        if payload is None:
            continue
        if _string_field(payload, "type") == payload_type:
            yield payload


def _extract_alert_flags_from_alert(blocks: Sequence[ContentBlock] | None) -> set[str]:
    flags: set[str] = set()
    for payload in _iter_json_payloads_of_type(blocks, "unsupported_content_removed"):
        flags.update(
            raw_flag for raw_flag in _string_values(payload.get("flags")) if raw_flag in _ALERT_FLAGS
        )
        flags.update(
            category_flag
            for category in _string_values(payload.get("categories"))
            if (category_flag := _category_alert_flag(category)) is not None
        )

    return flags


def _extract_alert_flags_from_meta(blocks: Sequence[ContentBlock] | None) -> set[str]:
    flags: set[str] = set()
    for payload in _iter_json_payloads_of_type(blocks, "fast-agent-removed"):
        category_flag = _category_alert_flag(_string_field(payload, "category"))
        if category_flag is not None:
            flags.add(category_flag)
    return flags


def _resolve_alert_flags_from_history(
    message_history: "Sequence[PromptMessageExtended]",
) -> set[str]:
    """Resolve TDV alert flags from persisted conversation history."""
    alert_flags: set[str] = set()
    error_seen = False

    for message in message_history:
        channels = message.channels or {}
        if channels.get(FAST_AGENT_ERROR_CHANNEL):
            error_seen = True

        if message.role != "user":
            continue

        alert_blocks = channels.get(FAST_AGENT_ALERT_CHANNEL, [])
        message_flags = _extract_alert_flags_from_alert(alert_blocks)
        if not message_flags:
            meta_blocks = channels.get(FAST_AGENT_REMOVED_METADATA_CHANNEL, [])
            message_flags = _extract_alert_flags_from_meta(meta_blocks)
        alert_flags.update(message_flags)

    if error_seen and not alert_flags:
        alert_flags.add("T")

    return alert_flags
