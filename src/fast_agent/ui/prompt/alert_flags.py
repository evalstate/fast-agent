"""Alert flag extraction helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fast_agent.constants import (
    FAST_AGENT_ALERT_CHANNEL,
    FAST_AGENT_ERROR_CHANNEL,
    FAST_AGENT_REMOVED_METADATA_CHANNEL,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.types import PromptMessageExtended


def _category_to_alert_flag(category: str) -> str | None:
    match category:
        case "text":
            return "T"
        case "document":
            return "D"
        case "vision":
            return "V"
    return None


def _extract_alert_flags_from_alert(blocks) -> set[str]:
    flags: set[str] = set()
    for block in blocks or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            continue

        if payload.get("type") != "unsupported_content_removed":
            continue

        raw_flags = payload.get("flags")
        if isinstance(raw_flags, str):
            raw_flags = [raw_flags]
        if isinstance(raw_flags, list):
            for raw_flag in raw_flags:
                if raw_flag in {"T", "D", "V"}:
                    flags.add(raw_flag)

        categories = payload.get("categories")
        if isinstance(categories, str):
            categories = [categories]
        if isinstance(categories, list):
            for category in categories:
                if not isinstance(category, str):
                    continue
                category_flag = _category_to_alert_flag(category)
                if category_flag is not None:
                    flags.add(category_flag)

    return flags


def _extract_alert_flags_from_meta(blocks) -> set[str]:
    flags: set[str] = set()
    for block in blocks or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            continue
        if payload.get("type") != "fast-agent-removed":
            continue
        category = payload.get("category")
        if not isinstance(category, str):
            continue
        category_flag = _category_to_alert_flag(category)
        if category_flag is not None:
            flags.add(category_flag)
    return flags


def _resolve_alert_flags_from_history(
    message_history: "Sequence[PromptMessageExtended]",
) -> set[str]:
    """Resolve TDV alert flags from persisted conversation history."""
    alert_flags: set[str] = set()
    legacy_alert_flags: set[str] = set()
    error_seen = False

    for message in message_history:
        channels = message.channels or {}
        if channels.get(FAST_AGENT_ERROR_CHANNEL):
            error_seen = True

        if message.role != "user":
            continue

        alert_blocks = channels.get(FAST_AGENT_ALERT_CHANNEL, [])
        alert_flags.update(_extract_alert_flags_from_alert(alert_blocks))

        meta_blocks = channels.get(FAST_AGENT_REMOVED_METADATA_CHANNEL, [])
        legacy_alert_flags.update(_extract_alert_flags_from_meta(meta_blocks))

    if not alert_flags:
        alert_flags.update(legacy_alert_flags)

    if error_seen and not alert_flags:
        alert_flags.add("T")

    return alert_flags
