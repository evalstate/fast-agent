"""Conversions between MCP content blocks and A2A message parts."""

from __future__ import annotations

import base64
import json
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

from a2a.types import Part
from google.protobuf.json_format import ParseDict
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)


def part_from_content(content: ContentBlock) -> Part | None:
    """Convert one supported MCP content block to an A2A part."""
    if isinstance(content, TextContent):
        return Part(text=content.text)
    if isinstance(content, ImageContent | AudioContent):
        return Part(
            raw=base64.b64decode(content.data),
            media_type=content.mimeType,
        )
    if isinstance(content, ResourceLink):
        return Part(
            url=str(content.uri),
            media_type=content.mimeType or "",
            filename=content.name,
        )
    if not isinstance(content, EmbeddedResource):
        return None

    resource = content.resource
    if isinstance(resource, BlobResourceContents):
        return Part(
            raw=base64.b64decode(resource.blob),
            media_type=resource.mimeType or "",
            filename=filename_from_uri(str(resource.uri)),
        )
    if isinstance(resource, TextResourceContents):
        data_part = json_data_part(resource.text, media_type=resource.mimeType)
        if data_part is not None:
            return data_part
        return Part(
            text=resource.text,
            media_type=resource.mimeType or "text/plain",
            filename=filename_from_uri(str(resource.uri)),
        )
    return None


def filename_from_uri(uri: str) -> str:
    parsed = urlparse(uri)
    name = PurePosixPath(unquote(parsed.path)).name
    return name or parsed.netloc or "attachment"


def json_data_part(text: str, *, media_type: str | None) -> Part | None:
    if media_type != "application/json":
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    part = Part(media_type=media_type)
    ParseDict(data, part.data)
    return part
