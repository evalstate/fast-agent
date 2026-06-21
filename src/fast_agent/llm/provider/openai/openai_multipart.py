# openai_multipart.py
"""
Clean utilities for converting between PromptMessageExtended and OpenAI message formats.
Each function handles all content types consistently and is designed for simple testing.
"""

from collections.abc import Mapping
from typing import Any, Literal

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
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)

from fast_agent.mcp.resource_utils import parse_resource_marker
from fast_agent.types import PromptMessageExtended


def _coerce_extended_role(value: object) -> Literal["assistant", "user"]:
    return "user" if value == "user" else "assistant"


def _coerce_str(value: object, *, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _message_role_and_content(
    message: ChatCompletionMessage | ChatCompletionMessageParam | dict[str, Any],
) -> tuple[Literal["assistant", "user"], object]:
    if isinstance(message, ChatCompletionMessage):
        return _coerce_extended_role(message.role), message.content
    if isinstance(message, dict):
        return _coerce_extended_role(message.get("role", "assistant")), message.get("content", "")

    return _coerce_extended_role(getattr(message, "role", "assistant")), getattr(
        message, "content", ""
    )


def _part_mapping(part: object) -> dict[str, Any] | None:
    if not isinstance(part, Mapping):
        return None
    return {str(key): value for key, value in part.items()}


def _part_value(
    part: object, mapping: dict[str, Any] | None, key: str, default: object = None
) -> object:
    if mapping is not None:
        return mapping.get(key, default)
    return getattr(part, key, default)


def _mapping_value(value: object, key: str, default: object = None) -> object:
    if not isinstance(value, Mapping):
        return default
    for item_key, item_value in value.items():
        if item_key == key:
            return item_value
    return default


def _text_part_to_content(part: object, mapping: dict[str, Any] | None) -> TextContent | object:
    text = _coerce_str(_part_value(part, mapping, "text", ""))
    resource_marker = parse_resource_marker(text)
    if resource_marker:
        return resource_marker
    return TextContent(type="text", text=text)


def _image_url_part_to_content(part: object, mapping: dict[str, Any] | None) -> ImageContent | None:
    image_url = _part_value(part, mapping, "image_url")
    if not image_url:
        return None

    raw_url = (
        _mapping_value(image_url, "url", "")
        if isinstance(image_url, Mapping)
        else getattr(image_url, "url", "")
    )
    url = _coerce_str(raw_url)
    if not url or not url.startswith("data:image/") or "," not in url:
        return None

    mime_type = url.split(";", 1)[0].replace("data:", "")
    data = url.split(",", 1)[1]
    return ImageContent(type="image", data=data, mimeType=mime_type)


def _text_resource_to_content(resource: dict[str, Any]) -> TextContent | EmbeddedResource:
    mime_type = resource["mimeType"]
    uri = resource.get("uri", "resource://unknown")
    if mime_type == "text/plain":
        return TextContent(type="text", text=resource["text"])
    return EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            text=resource["text"],
            mimeType=mime_type,
            uri=uri,
        ),
    )


def _blob_resource_to_content(resource: dict[str, Any]) -> ImageContent | EmbeddedResource:
    mime_type = resource["mimeType"]
    uri = resource.get("uri", "resource://unknown")
    if mime_type.startswith("image/") and mime_type != "image/svg+xml":
        return ImageContent(
            type="image",
            data=resource["blob"],
            mimeType=mime_type,
        )
    return EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            blob=resource["blob"],
            mimeType=mime_type,
            uri=uri,
        ),
    )


def _resource_part_to_content(
    mapping: dict[str, Any] | None,
) -> TextContent | ImageContent | EmbeddedResource | None:
    if mapping is None:
        return None

    resource = mapping.get("resource")
    if not isinstance(resource, dict) or "mimeType" not in resource:
        return None

    if "text" in resource:
        return _text_resource_to_content(resource)
    if "blob" in resource:
        return _blob_resource_to_content(resource)
    return None


def _content_part_to_mcp_content(
    part: object,
) -> ContentBlock | None:
    mapping = _part_mapping(part)
    part_type = _part_value(part, mapping, "type")

    if part_type == "text":
        text_content = _text_part_to_content(part, mapping)
        if isinstance(
            text_content, (TextContent, ImageContent, AudioContent, ResourceLink, EmbeddedResource)
        ):
            return text_content
        return None
    if part_type == "image_url":
        return _image_url_part_to_content(part, mapping)
    if part_type == "resource":
        return _resource_part_to_content(mapping)
    return None


def openai_to_extended(
    message: ChatCompletionMessage
    | ChatCompletionMessageParam
    | dict[str, Any]
    | list[ChatCompletionMessage | ChatCompletionMessageParam | dict[str, Any]],
) -> PromptMessageExtended | list[PromptMessageExtended]:
    """
    Convert OpenAI messages to PromptMessageExtended format.

    Args:
        message: OpenAI Message, MessageParam, or list of them

    Returns:
        Equivalent message(s) in PromptMessageExtended format
    """
    if isinstance(message, list):
        return [_openai_message_to_extended(m) for m in message]
    return _openai_message_to_extended(message)


def _openai_message_to_extended(
    message: ChatCompletionMessage | ChatCompletionMessageParam | dict[str, Any],
) -> PromptMessageExtended:
    """Convert a single OpenAI message to PromptMessageExtended."""
    role, content = _message_role_and_content(message)

    mcp_contents: list[ContentBlock] = []

    if isinstance(content, str):
        mcp_contents.append(TextContent(type="text", text=content))
    elif isinstance(content, list):
        for part in content:
            mcp_content = _content_part_to_mcp_content(part)
            if mcp_content is not None:
                mcp_contents.append(mcp_content)

    return PromptMessageExtended(role=role, content=mcp_contents)
