"""Capture and replay of opaque OpenAI Responses compaction items."""

from __future__ import annotations

from typing import Literal, Mapping, NotRequired, TypedDict

from fast_agent.core.logging.json_serializer import snapshot_json_value

COMPACTION_REPLAY_SCHEMA = "fast-agent.openai-responses.compaction-replay"
COMPACTION_REPLAY_VERSION = 1
_COMPACTION_REPLAY_FIELDS = frozenset({"schema", "version", "item"})
# ``origin`` (the route that produced the item) is optional: earlier envelopes lack it.
_OPTIONAL_COMPACTION_REPLAY_FIELDS = frozenset({"origin"})


class CompactionItem(TypedDict):
    """The exact input shape the API accepts back: type, id and encrypted_content."""

    type: Literal["compaction"]
    id: str
    encrypted_content: str


class CompactionReplayEnvelope(TypedDict):
    schema: Literal["fast-agent.openai-responses.compaction-replay"]
    version: Literal[1]
    item: CompactionItem
    origin: NotRequired[str]
    """Config name of the route that produced the item; some routes' items are only
    readable on that route (see ``ResponsesLLM._accepts_checkpoint_origin``)."""


def capture_compaction_replay(
    output_item: object, *, origin: str
) -> CompactionReplayEnvelope | None:
    """Capture a completed ``compaction`` output item for history replay."""
    payload = snapshot_json_value(output_item)
    if not isinstance(payload, dict):
        return None
    item = _compaction_item(payload)
    if item is None:
        return None
    return {
        "schema": COMPACTION_REPLAY_SCHEMA,
        "version": COMPACTION_REPLAY_VERSION,
        "item": item,
        "origin": origin,
    }


def compaction_replay_origin(payload: Mapping[str, object]) -> str | None:
    """The producing route recorded in an envelope, or None for earlier envelopes."""
    origin = payload.get("origin")
    return origin if isinstance(origin, str) and origin else None


def parse_compaction_replay(payload: Mapping[str, object]) -> CompactionItem | None:
    """Parse the canonical history envelope into the exact provider input shape."""
    fields = frozenset(payload)
    if (
        not _COMPACTION_REPLAY_FIELDS
        <= fields
        <= _COMPACTION_REPLAY_FIELDS | _OPTIONAL_COMPACTION_REPLAY_FIELDS
        or payload.get("schema") != COMPACTION_REPLAY_SCHEMA
        or payload.get("version") != COMPACTION_REPLAY_VERSION
    ):
        return None
    item = payload.get("item")
    if not isinstance(item, dict):
        return None
    return _compaction_item({str(key): value for key, value in item.items()})


def _compaction_item(payload: Mapping[str, object]) -> CompactionItem | None:
    item_id = payload.get("id")
    encrypted_content = payload.get("encrypted_content")
    if payload.get("type") != "compaction":
        return None
    if not isinstance(item_id, str) or not item_id:
        return None
    if not isinstance(encrypted_content, str) or not encrypted_content:
        return None
    return {"type": "compaction", "id": item_id, "encrypted_content": encrypted_content}
