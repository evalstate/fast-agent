# openai_multipart.py
"""
Clean utilities for converting between PromptMessageMultipart and OpenAI message formats.
Each function handles all content types consistently and is designed for simple testing.
"""

from typing import Any, Dict, List, Union

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

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


def openai_to_multipart(
    message: Union[
        ChatCompletionMessage,
        ChatCompletionMessageParam,
        List[Union[ChatCompletionMessage, ChatCompletionMessageParam]],
    ],
) -> Union[PromptMessageMultipart, List[PromptMessageMultipart]]:
    """
    Convert OpenAI messages to PromptMessageMultipart format.

    Args:
        message: OpenAI Message, MessageParam, or list of them

    Returns:
        Equivalent message(s) in PromptMessageMultipart format
    """
    if isinstance(message, list):
        return [_openai_message_to_multipart(m) for m in message]
    return _openai_message_to_multipart(message)


def _openai_message_to_multipart(
    message: Union[ChatCompletionMessage, Dict[str, Any]],
) -> PromptMessageMultipart:
    """Convert a single OpenAI message to PromptMessageMultipart."""
    # Get role and content from message
    if isinstance(message, dict):
        role = message.get("role", "assistant")
        content = message.get("content", "")
    else:
        role = message.role
        content = message.content

    mcp_contents = []

    # Handle string content (simple case)
    if isinstance(content, str):
        mcp_contents.append(TextContent(type="text", text=content))

    # Handle list of content parts
    elif isinstance(content, list):
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)

            # Handle text content
            if part_type == "text":
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", "")

                # Check if this is a resource marker
                if (
                    text
                    and (text.startswith("[Resource:") or text.startswith("[Binary Resource:"))
                    and "\n" in text
                ):
                    header, content_text = text.split("\n", 1)
                    if "MIME:" in header:
                        mime_match = header.split("MIME:", 1)[1].split("]")[0].strip()

                        # If not text/plain, create an embedded resource
                        if mime_match != "text/plain":
                            if "Resource:" in header and "Binary Resource:" not in header:
                                uri = header.split("Resource:", 1)[1].split(",")[0].strip()
                                mcp_contents.append(
                                    EmbeddedResource(
                                        type="resource",
                                        resource=TextResourceContents(
                                            uri=uri,
                                            mimeType=mime_match,
                                            text=content_text,
                                        ),
                                    )
                                )
                                continue

                # Regular text content
                mcp_contents.append(TextContent(type="text", text=text))

            # Handle image content
            elif part_type == "image_url":
                image_url = (
                    part.get("image_url", {})
                    if isinstance(part, dict)
                    else getattr(part, "image_url", None)
                )
                if image_url:
                    url = (
                        image_url.get("url")
                        if isinstance(image_url, dict)
                        else getattr(image_url, "url", "")
                    )
                    if url and url.startswith("data:image/"):
                        # Handle base64 data URLs
                        mime_type = url.split(";")[0].replace("data:", "")
                        data = url.split(",")[1]
                        mcp_contents.append(
                            ImageContent(type="image", data=data, mimeType=mime_type)
                        )

            # Handle explicit resource types
            elif part_type == "resource" and isinstance(part, dict) and "resource" in part:
                resource = part["resource"]
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

    return PromptMessageMultipart(role=role, content=mcp_contents)
