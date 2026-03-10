# openai_multipart.py
"""
Clean utilities for converting between PromptMessageExtended and OpenAI message formats.
Each function handles all content types consistently and is designed for simple testing.
"""

from typing import Any, Literal, Union

from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
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


def openai_to_extended(
    message: Union[
        ChatCompletionMessage,
        ChatCompletionMessageParam,
        dict[str, Any],
        list[Union[ChatCompletionMessage, ChatCompletionMessageParam, dict[str, Any]]],
    ],
) -> Union[PromptMessageExtended, list[PromptMessageExtended]]:
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
    message: Union[ChatCompletionMessage, ChatCompletionMessageParam, dict[str, Any]],
) -> PromptMessageExtended:
    """Convert a single OpenAI message to PromptMessageExtended."""
    # Get role and content from message
    # ChatCompletionMessage is a class with attributes; MessageParam types are TypedDicts
    if isinstance(message, ChatCompletionMessage):
        role = _coerce_extended_role(message.role)
        content = message.content
    elif isinstance(message, dict):
        role = _coerce_extended_role(message.get("role", "assistant"))
        content = message.get("content", "")
    else:
        # Fallback for any other object with role/content attributes
        role = _coerce_extended_role(getattr(message, "role", "assistant"))
        content = getattr(message, "content", "")

    mcp_contents = []

    # Handle string content (simple case)
    if isinstance(content, str):
        mcp_contents.append(TextContent(type="text", text=content))

    # Handle list of content parts
    elif isinstance(content, list):
        for part in content:
            part_dict = dict(part) if isinstance(part, dict) else None
            part_type = part_dict.get("type") if part_dict is not None else getattr(part, "type", None)

            # Handle text content
            if part_type == "text":
                raw_text = part_dict.get("text") if part_dict is not None else getattr(part, "text", "")
                text = _coerce_str(raw_text)

                resource_marker = parse_resource_marker(text)
                if resource_marker:
                    mcp_contents.append(resource_marker)
                    continue

                # Regular text content
                mcp_contents.append(TextContent(type="text", text=text))

            # Handle image content
            elif part_type == "image_url":
                image_url = (
                    part_dict.get("image_url")
                    if part_dict is not None
                    else getattr(part, "image_url", None)
                )
                if image_url:
                    raw_url = (
                        image_url.get("url")
                        if isinstance(image_url, dict)
                        else getattr(image_url, "url", "")
                    )
                    url = _coerce_str(raw_url)
                    if url and url.startswith("data:image/"):
                        # Handle base64 data URLs
                        mime_type = url.split(";")[0].replace("data:", "")
                        data = url.split(",")[1]
                        mcp_contents.append(
                            ImageContent(type="image", data=data, mimeType=mime_type)
                        )

            # Handle explicit resource types
            elif part_type == "resource" and part_dict is not None:
                resource = part_dict.get("resource")
                if isinstance(resource, dict):
                    # Text resource
                    if "text" in resource and "mimeType" in resource:
                        mime_type = resource["mimeType"]
                        uri = resource.get("uri", "resource://unknown")

                        if mime_type == "text/plain":
                            mcp_contents.append(TextContent(type="text", text=resource["text"]))
                        else:
                            mcp_contents.append(
                                EmbeddedResource(
                                    type="resource",
                                    resource=TextResourceContents(
                                        text=resource["text"],
                                        mimeType=mime_type,
                                        uri=uri,
                                    ),
                                )
                            )
                    # Binary resource
                    elif "blob" in resource and "mimeType" in resource:
                        mime_type = resource["mimeType"]
                        uri = resource.get("uri", "resource://unknown")

                        if mime_type.startswith("image/") and mime_type != "image/svg+xml":
                            mcp_contents.append(
                                ImageContent(
                                    type="image",
                                    data=resource["blob"],
                                    mimeType=mime_type,
                                )
                            )
                        else:
                            mcp_contents.append(
                                EmbeddedResource(
                                    type="resource",
                                    resource=BlobResourceContents(
                                        blob=resource["blob"],
                                        mimeType=mime_type,
                                        uri=uri,
                                    ),
                                )
                            )

    return PromptMessageExtended(role=role, content=mcp_contents)
