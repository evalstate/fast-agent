from __future__ import annotations

from typing import Literal, Mapping, NotRequired, TypedDict

from fast_agent.core.logging.json_serializer import snapshot_json_value

REASONING_REPLAY_SCHEMA = "fast-agent.openai-responses.reasoning-replay"
REASONING_REPLAY_VERSION = 1
_REASONING_REPLAY_FIELDS = frozenset({"schema", "version", "item"})
_REASONING_ITEM_FIELDS = frozenset({"type", "id", "summary", "encrypted_content", "content"})


class ReasoningSummaryPart(TypedDict):
    type: Literal["summary_text"]
    text: str


class ReasoningContentPart(TypedDict):
    type: Literal["reasoning_text"]
    text: str


class CanonicalReasoningItem(TypedDict):
    type: Literal["reasoning"]
    id: str
    summary: list[ReasoningSummaryPart]
    encrypted_content: str
    content: NotRequired[list[ReasoningContentPart]]


class ReasoningReplayEnvelope(TypedDict):
    schema: Literal["fast-agent.openai-responses.reasoning-replay"]
    version: Literal[1]
    item: CanonicalReasoningItem


def capture_reasoning_replay(output_item: object) -> ReasoningReplayEnvelope | None:
    """Capture the provider-supported reasoning input fields from a completed output item."""
    payload = snapshot_json_value(output_item)
    if not isinstance(payload, dict):
        return None
    if payload.get("status") not in {None, "completed"}:
        return None

    item = _canonical_reasoning_item(payload)
    if item is None:
        return None

    return {
        "schema": REASONING_REPLAY_SCHEMA,
        "version": REASONING_REPLAY_VERSION,
        "item": item,
    }


def parse_reasoning_replay(payload: Mapping[str, object]) -> CanonicalReasoningItem | None:
    """Parse the canonical history envelope into the exact provider input shape."""
    if (
        frozenset(payload) != _REASONING_REPLAY_FIELDS
        or payload.get("schema") != REASONING_REPLAY_SCHEMA
        or payload.get("version") != REASONING_REPLAY_VERSION
    ):
        return None
    raw_item = _string_keyed_mapping(payload.get("item"))
    if raw_item is None or not frozenset(raw_item).issubset(_REASONING_ITEM_FIELDS):
        return None
    return _canonical_reasoning_item(raw_item)


def _canonical_reasoning_item(
    payload: Mapping[str, object],
) -> CanonicalReasoningItem | None:
    item_type = payload.get("type")
    if item_type != "reasoning":
        return None

    item_id = payload.get("id")
    encrypted_content = payload.get("encrypted_content")
    if not isinstance(item_id, str) or not item_id:
        return None
    if not isinstance(encrypted_content, str) or not encrypted_content:
        return None

    summary_value = payload.get("summary")
    summary = _summary_parts(summary_value)
    if summary is None:
        return None

    item: CanonicalReasoningItem = {
        "type": "reasoning",
        "id": item_id,
        "summary": summary,
        "encrypted_content": encrypted_content,
    }

    content_value = payload.get("content")
    if content_value is not None:
        content = _content_parts(content_value)
        if content is None:
            return None
        if content:
            item["content"] = content
    return item


def _summary_parts(value: object) -> list[ReasoningSummaryPart] | None:
    if not isinstance(value, list):
        return None
    parts: list[ReasoningSummaryPart] = []
    for part in value:
        if not isinstance(part, Mapping):
            return None
        text = part.get("text")
        if part.get("type") != "summary_text" or not isinstance(text, str):
            return None
        parts.append({"type": "summary_text", "text": text})
    return parts


def _content_parts(value: object) -> list[ReasoningContentPart] | None:
    if not isinstance(value, list):
        return None
    parts: list[ReasoningContentPart] = []
    for part in value:
        if not isinstance(part, Mapping):
            return None
        text = part.get("text")
        if part.get("type") != "reasoning_text" or not isinstance(text, str):
            return None
        parts.append({"type": "reasoning_text", "text": text})
    return parts


def _string_keyed_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result
