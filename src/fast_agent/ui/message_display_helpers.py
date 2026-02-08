from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.mcp.helpers.content_helpers import (
    is_image_content,
    is_resource_content,
    is_resource_link,
)
from fast_agent.types import LlmStopReason

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.types import PromptMessageExtended


def extract_user_attachments(message: PromptMessageExtended) -> list[str]:
    attachments: list[str] = []
    for content in message.content:
        if is_resource_link(content):
            # ResourceLink: show name or mime type
            from mcp.types import ResourceLink

            assert isinstance(content, ResourceLink)
            label = content.name or content.mimeType or "resource"
            attachments.append(label)
        elif is_image_content(content):
            attachments.append("image")
        elif is_resource_content(content):
            # EmbeddedResource: show name or uri
            from mcp.types import EmbeddedResource

            assert isinstance(content, EmbeddedResource)
            label = getattr(content.resource, "name", None) or str(content.resource.uri)
            attachments.append(label)
    return attachments


def build_user_message_display(
    messages: Sequence[PromptMessageExtended],
) -> tuple[str, list[str] | None]:
    if not messages:
        return "", None

    if len(messages) == 1:
        message = messages[0]
        message_text = message.last_text() or ""
        attachments = extract_user_attachments(message)
        return message_text, attachments or None

    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        attachments = extract_user_attachments(message)
        if attachments:
            lines.append(f"🔗 {', '.join(attachments)}")
        message_text = message.last_text() or ""
        if message_text:
            lines.append(message_text)
        if index < len(messages):
            lines.append("")

    return "\n".join(lines), None


def build_tool_use_additional_message(
    message: "PromptMessageExtended",
    last_text: str | None = None,
) -> Text | None:
    if message.stop_reason != LlmStopReason.TOOL_USE:
        return None
    if last_text is None:
        last_text = message.last_text()
    if last_text is not None:
        return None
    return Text("The assistant requested tool calls", style="dim green italic")


__all__ = [
    "build_tool_use_additional_message",
    "build_user_message_display",
    "extract_user_attachments",
]
