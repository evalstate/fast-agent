"""Convert values crossing the sandbox boundary.

Tool results become plain Python data inside the sandbox; the final script value becomes text
for the model. Binary media cannot enter the sandbox, so it is held host-side and represented by
a handle; handles in the final value bring their media back into the model's context.
"""

from __future__ import annotations

import json
import secrets
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from mcp_types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
)

from fast_agent.constants import FAST_AGENT_PTC_ERROR, FAST_AGENT_PTC_VALUE

MEDIA_ID_KEY = "media_id"
_MAX_RETAINED_MEDIA = 64


class ToolCallFailed(RuntimeError):
    """A nested tool returned an error result; surfaces in the sandbox as RuntimeError."""


class MediaStore:
    """Host-side media referenced by handles inside the sandbox, bounded to recent items."""

    def __init__(self) -> None:
        self._blocks: OrderedDict[str, ContentBlock] = OrderedDict()
        self._next = 0
        # Handles outlive this store (store() persists them with the session), so a restored
        # handle must not name media a later runtime fetched under the same number.
        self._prefix = f"media-{secrets.token_hex(4)}"

    def handle(self, block: ImageContent | AudioContent | EmbeddedResource) -> dict[str, Any]:
        self._next += 1
        media_id = f"{self._prefix}-{self._next}"
        self._blocks[media_id] = block
        while len(self._blocks) > _MAX_RETAINED_MEDIA:
            self._blocks.popitem(last=False)
        if isinstance(block, EmbeddedResource):
            return {
                "type": "resource",
                "uri": str(block.resource.uri),
                "mime_type": block.resource.mime_type,
                MEDIA_ID_KEY: media_id,
            }
        return {"type": block.type, "mime_type": block.mime_type, MEDIA_ID_KEY: media_id}

    def referenced(self, value: Any) -> list[ContentBlock]:
        """Return media whose handles appear in ``value``, in order and without repeats."""
        found: dict[str, ContentBlock] = {}
        self._collect(value, found, depth=0)
        return list(found.values())

    def clear(self) -> None:
        self._blocks.clear()

    def _collect(self, value: Any, found: dict[str, ContentBlock], *, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(value, Mapping):
            media_id = value.get(MEDIA_ID_KEY)
            if isinstance(media_id, str) and media_id in self._blocks:
                found.setdefault(media_id, self._blocks[media_id])
                return
            for item in value.values():
                self._collect(item, found, depth=depth + 1)
        elif isinstance(value, list | tuple):
            for item in value:
                self._collect(item, found, depth=depth + 1)


def tool_result_to_value(result: CallToolResult, media: MediaStore, *, name: str) -> Any:
    """Return the tool's program value, structured content, or parsed content blocks.

    A program value is the tool's own answer to sandbox code and is returned even for an error
    result: a shell command that exits non-zero, say, is data for the script. Other error
    results raise ToolCallFailed, named after the sandbox function ``name``, so sandbox code can
    use try/except.

    Structured results with non-text blocks use a ``structured_content``/``content`` envelope
    to preserve the payload without colliding with server-defined fields. Text blocks are
    superseded by structured content, as in direct tool results.
    """
    meta = result.meta or {}
    program_value = meta.get(FAST_AGENT_PTC_VALUE)
    if program_value is not None:
        return program_value
    if result.is_error:
        message = meta.get(FAST_AGENT_PTC_ERROR)
        if not isinstance(message, str):
            message = _error_message(_result_text(result))
        raise ToolCallFailed(f"{name}: {message or 'tool call failed'}")
    if result.structured_content is not None:
        content_values = [
            _content_to_value(block, media)
            for block in result.content
            if not isinstance(block, TextContent)
        ]
        if not content_values:
            return result.structured_content
        return {"structured_content": result.structured_content, "content": content_values}
    values = [_content_to_value(block, media) for block in result.content]
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=_json_fallback)
    except (TypeError, ValueError):
        return repr(value)


def _content_to_value(block: ContentBlock, media: MediaStore) -> Any:
    match block:
        case TextContent(text=text):
            return _maybe_json(text)
        case ImageContent() | AudioContent():
            return media.handle(block)
        case EmbeddedResource(resource=resource):
            if isinstance(resource, BlobResourceContents):
                return media.handle(block)
            return _maybe_json(resource.text)
        case ResourceLink(uri=uri, name=name):
            return {"type": "resource_link", "uri": str(uri), "name": name}
    return None


def _maybe_json(text: str) -> Any:
    if text.lstrip()[:1] not in ("{", "["):
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def _error_message(text: str) -> str:
    """Summarise a model-facing error text for an exception message.

    JSON error bodies (``{"error": ..., "message": ...}``), as returned by edit tools and many MCP
    servers, become their message; the redundant ``Error:`` prefix is dropped.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except ValueError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("message"), str):
            code = payload.get("error")
            return f"{payload['message']} [{code}]" if isinstance(code, str) else payload["message"]
    return stripped.removeprefix("Error:").strip()


def _result_text(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _json_fallback(value: Any) -> Any:
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    return repr(value)
