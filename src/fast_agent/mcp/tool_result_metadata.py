"""Typed metadata helpers for tool execution results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakKeyDictionary

from mcp.types import CallToolResult

if TYPE_CHECKING:
    from mcp.types import ContentBlock

    from fast_agent.mcp.url_elicitation_required import (
        URLElicitationRequiredDisplayPayload,
    )

_FATAL_TOOL_ERROR_META_KEY = "fast_agent/fatal_tool_error"
_URL_ELICITATION_META_KEY = "fast_agent/url_elicitation_required"
_MEDIA_PREVIEW_META_KEY = "fast_agent/media_preview_content"


_OBJECT_URL_ELICITATION_METADATA: WeakKeyDictionary[
    object, "URLElicitationRequiredDisplayPayload"
] = WeakKeyDictionary()
_STRONG_URL_ELICITATION_METADATA: dict[
    int, tuple[object, "URLElicitationRequiredDisplayPayload"]
] = {}


def _metadata(result: CallToolResult) -> dict[str, Any]:
    metadata = result.meta
    if metadata is None:
        metadata = {}
        result.meta = metadata
    return metadata


def set_fatal_tool_error(result: CallToolResult, message: str) -> CallToolResult:
    _metadata(result)[_FATAL_TOOL_ERROR_META_KEY] = message
    return result


def fatal_tool_error(result: CallToolResult) -> str | None:
    metadata = result.meta or {}
    value = metadata.get(_FATAL_TOOL_ERROR_META_KEY)
    return value if isinstance(value, str) else None


def set_url_elicitation_required_payload(
    target: object,
    payload: "URLElicitationRequiredDisplayPayload",
) -> None:
    if isinstance(target, CallToolResult):
        _metadata(target)[_URL_ELICITATION_META_KEY] = payload
        return
    try:
        _OBJECT_URL_ELICITATION_METADATA[target] = payload
    except TypeError:
        _STRONG_URL_ELICITATION_METADATA[id(target)] = (target, payload)


def url_elicitation_required_payload(
    target: object,
) -> "URLElicitationRequiredDisplayPayload | None":
    if isinstance(target, CallToolResult):
        metadata = target.meta or {}
        payload = metadata.get(_URL_ELICITATION_META_KEY)
        if _is_url_elicitation_required_payload(payload):
            return payload

    try:
        payload = _OBJECT_URL_ELICITATION_METADATA.get(target)
    except TypeError:
        item = _STRONG_URL_ELICITATION_METADATA.get(id(target))
        if item is None:
            return None
        stored_target, payload = item
        if stored_target is not target:
            return None
    return payload if _is_url_elicitation_required_payload(payload) else None


def _is_url_elicitation_required_payload(
    payload: object,
) -> bool:
    from fast_agent.mcp.url_elicitation_required import (
        URLElicitationRequiredDisplayPayload,
    )

    return isinstance(payload, URLElicitationRequiredDisplayPayload)


def set_tool_result_media_preview(
    result: CallToolResult,
    content: Sequence["ContentBlock"],
) -> None:
    """Attach media preview blocks without changing provider-facing result content."""
    _metadata(result)[_MEDIA_PREVIEW_META_KEY] = list(content)


def get_tool_result_media_preview(
    result: CallToolResult,
) -> Sequence["ContentBlock"] | None:
    """Return display-only media preview blocks attached to a tool result."""
    value = (result.meta or {}).get(_MEDIA_PREVIEW_META_KEY)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    return cast("Sequence[ContentBlock]", value)
