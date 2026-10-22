"""Convert values crossing the sandbox boundary.

Tool results become plain Python data inside the sandbox; the final script value becomes text
for the model.
"""

from __future__ import annotations

import json
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


class ToolCallFailed(RuntimeError):
    """A nested tool returned an error result; surfaces in the sandbox as RuntimeError."""


def tool_result_to_value(result: CallToolResult) -> Any:
    """Return structured content when present, else parsed content blocks.

    Raises ToolCallFailed for error results so sandbox code can use try/except.
    """
    if result.is_error:
        raise ToolCallFailed(_result_text(result) or "tool call failed")
    if result.structured_content is not None:
        return result.structured_content
    values = [_content_to_value(block) for block in result.content]
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


def _content_to_value(block: ContentBlock) -> Any:
    match block:
        case TextContent(text=text):
            return _maybe_json(text)
        case ImageContent(mime_type=mime_type) | AudioContent(mime_type=mime_type):
            # Binary media is not forwarded into the sandbox in this proof of concept.
            return {"type": block.type, "mime_type": mime_type, "omitted": True}
        case EmbeddedResource(resource=resource):
            if isinstance(resource, BlobResourceContents):
                return {
                    "type": "resource",
                    "uri": str(resource.uri),
                    "mime_type": resource.mime_type,
                }
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


def _result_text(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _json_fallback(value: Any) -> Any:
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    return repr(value)
