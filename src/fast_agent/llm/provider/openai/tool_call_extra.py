"""Keep OpenAI-compatible tool-call ``extra_content`` for replay.

Gemini 3 puts a thought signature on ``tool_calls[].extra_content`` and rejects
the next tool-loop request when that field is missing. The value is opaque
provider data, stored on the assistant message and keyed by tool call id.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from mcp_types import ContentBlock, TextContent
from pydantic import BaseModel

from fast_agent.constants import OPENAI_TOOL_CALL_EXTRA_CONTENT
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.content_helpers import get_text

_logger = get_logger(__name__)


def remember_tool_call_extra(accumulated: dict[str, Any], delta: object) -> None:
    """Merge one streamed tool-call delta's ``extra_content`` into ``accumulated``."""
    incoming = read_tool_call_extra_content(delta)
    if incoming is None:
        return
    current = accumulated.get("extra_content")
    merged = merge_tool_call_extra_content(
        current if isinstance(current, dict) else None,
        incoming,
    )
    if merged:
        accumulated["extra_content"] = merged


def remember_chunk_tool_call_extra(
    chunk: object,
    extra_by_index: dict[int, dict[str, Any]],
) -> None:
    """Record ``extra_content`` from a chat-completions stream chunk, keyed by index."""
    if not isinstance(chunk, BaseModel):
        return
    choices = chunk.__dict__.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, BaseModel):
            continue
        delta = choice.__dict__.get("delta")
        if not isinstance(delta, BaseModel):
            continue
        tool_calls = delta.__dict__.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            index = _tool_call_index(tool_call)
            incoming = read_tool_call_extra_content(tool_call)
            if index is None or incoming is None:
                continue
            extra_by_index[index] = (
                merge_tool_call_extra_content(extra_by_index.get(index), incoming) or incoming
            )


def retain_completion_tool_call_extra(
    completion: object,
    extra_by_index: Mapping[int, Mapping[str, Any]],
) -> None:
    """Copy streamed ``extra_content`` onto the final tool calls when the snapshot dropped it."""
    if not extra_by_index or not isinstance(completion, BaseModel):
        return
    choices = completion.__dict__.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0]
    if not isinstance(choice, BaseModel):
        return
    message = choice.__dict__.get("message")
    if not isinstance(message, BaseModel):
        return
    tool_calls = message.__dict__.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    for index, tool_call in enumerate(tool_calls):
        incoming = extra_by_index.get(index)
        if not isinstance(incoming, Mapping) or not isinstance(tool_call, BaseModel):
            continue
        current = read_tool_call_extra_content(tool_call)
        merged = merge_tool_call_extra_content(current, incoming)
        if merged is None or merged == current:
            continue
        write_tool_call_extra_content(tool_call, merged)


def write_tool_call_extra_content(tool_call: BaseModel, extra: Mapping[str, Any]) -> None:
    """Attach ``extra_content`` to an OpenAI tool call that has no field for it."""
    extra_store = tool_call.__pydantic_extra__
    copied = dict(extra)
    if extra_store is None:
        object.__setattr__(tool_call, "__pydantic_extra__", {"extra_content": copied})
        return
    extra_store["extra_content"] = copied


def tool_call_extra_by_id(tool_calls: object) -> dict[str, dict[str, Any]]:
    """Return ``extra_content`` for each tool call that has it, keyed by call id."""
    if not isinstance(tool_calls, list):
        return {}
    extras: dict[str, dict[str, Any]] = {}
    for tool_call in tool_calls:
        tool_id = _tool_call_id(tool_call)
        extra = read_tool_call_extra_content(tool_call)
        if tool_id and extra:
            extras[tool_id] = extra
    return extras


def add_tool_call_extra_channel(
    channels: dict[str, list[ContentBlock]],
    extra_by_id: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Store provider tool-call extras so history and session save can replay them."""
    if not extra_by_id:
        return
    channels[OPENAI_TOOL_CALL_EXTRA_CONTENT] = [
        TextContent(
            type="text",
            text=json.dumps(
                {tool_id: dict(extra) for tool_id, extra in extra_by_id.items()},
            ),
        )
    ]


def extra_content_by_tool_id(
    channels: Mapping[str, Sequence[ContentBlock]] | None,
) -> dict[str, dict[str, Any]]:
    """Read the tool-call extra channel written by :func:`add_tool_call_extra_channel`."""
    if not channels:
        return {}
    blocks = channels.get(OPENAI_TOOL_CALL_EXTRA_CONTENT)
    if not blocks:
        return {}

    extras: dict[str, dict[str, Any]] = {}
    for block in blocks:
        text = get_text(block)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            _logger.debug("Skipping malformed OpenAI tool-call extra_content block")
            continue
        if not isinstance(payload, dict):
            _logger.debug("Skipping malformed OpenAI tool-call extra_content block")
            continue
        for tool_id, extra in payload.items():
            if not isinstance(tool_id, str) or not tool_id:
                continue
            parsed = _json_object(extra)
            if parsed is not None:
                extras[tool_id] = parsed
    return extras


def read_tool_call_extra_content(tool_call: object) -> dict[str, Any] | None:
    """Return a tool call's ``extra_content`` object, including SDK model extras."""
    return _json_object(_field(tool_call, "extra_content"))


def merge_tool_call_extra_content(
    current: Mapping[str, Any] | None,
    incoming: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge streamed ``extra_content`` objects. Later values replace earlier ones."""
    if not incoming:
        if not current:
            return None
        return dict(current)
    if not current:
        return dict(incoming)

    merged: dict[str, Any] = dict(current)
    for key, value in incoming.items():
        previous = merged.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            nested = merge_tool_call_extra_content(previous, value)
            if nested is not None:
                merged[key] = nested
            continue
        if value is not None:
            merged[key] = value
    return merged or None


def _tool_call_id(tool_call: object) -> str | None:
    tool_id = _field(tool_call, "id")
    if isinstance(tool_id, str) and tool_id:
        return tool_id
    return None


def _tool_call_index(tool_call: object) -> int | None:
    index = _field(tool_call, "index")
    if isinstance(index, int) and not isinstance(index, bool):
        return index
    return None


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    if isinstance(value, BaseModel):
        if name in value.__dict__:
            return value.__dict__[name]
        extra = value.model_extra
        if extra is not None and name in extra:
            return extra[name]
        return None
    try:
        attributes = vars(value)
    except TypeError:
        return None
    return attributes.get(name)


def _json_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError):
        return None
    decoded = json.loads(encoded)
    if isinstance(decoded, dict) and decoded:
        return decoded
    return None
