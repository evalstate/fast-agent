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
            source_uri = _content_source_uri(content)
            attachments.append(f"image ({source_uri})" if source_uri else "image")
        elif is_resource_content(content):
            # EmbeddedResource: show name or uri
            from mcp.types import EmbeddedResource

            assert isinstance(content, EmbeddedResource)
            label = getattr(content.resource, "name", None) or str(content.resource.uri)
            attachments.append(label)
    return attachments


def _content_source_uri(content: object) -> str | None:
    meta = getattr(content, "meta", None)
    if not isinstance(meta, dict):
        return None
    source_uri = meta.get("fast_agent_source_uri")
    return source_uri if isinstance(source_uri, str) and source_uri else None


def _message_display_text(message: PromptMessageExtended) -> str:
    from mcp.types import TextContent

    for content in message.content:
        if not isinstance(content, TextContent):
            continue
        meta = getattr(content, "meta", None)
        if isinstance(meta, dict):
            original_text = meta.get("fast_agent_original_text")
            if isinstance(original_text, str):
                return original_text
        return content.text
    return message.last_text() or ""


def build_user_message_display(
    messages: Sequence[PromptMessageExtended],
) -> tuple[str, list[str] | None]:
    if not messages:
        return "", None

    if len(messages) == 1:
        message = messages[0]
        message_text = _message_display_text(message)
        attachments = extract_user_attachments(message)
        return message_text, attachments or None

    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        attachments = extract_user_attachments(message)
        if attachments:
            lines.append(f"🔗 {', '.join(attachments)}")
        message_text = _message_display_text(message)
        if message_text:
            lines.append(message_text)
        if index < len(messages):
            lines.append("")

    return "\n".join(lines), None


def build_tool_use_additional_message(
    message: "PromptMessageExtended",
    last_text: str | None = None,
    *,
    shell_access: bool = False,
    file_read: bool = False,
) -> Text | None:
    if message.stop_reason != LlmStopReason.TOOL_USE:
        return None
    if last_text is None:
        last_text = message.last_text()
    if last_text is not None:
        return None
    if shell_access:
        message_text = "The assistant requested shell access"
    elif file_read:
        return None
    else:
        message_text = "The assistant requested tool calls"
    return Text(message_text, style="dim green italic")


def resolve_highlight_index(
    items: "Sequence[str] | None",
    highlight_items: str | "Sequence[str]" | None,
) -> int | None:
    """Resolve a highlighted item name (or names) to its index in a displayed list."""
    if items is None or len(items) == 0 or highlight_items is None:
        return None

    target: str
    if isinstance(highlight_items, str):
        if not highlight_items:
            return None
        target = highlight_items
    else:
        if len(highlight_items) == 0:
            return None
        target = highlight_items[0]

    for index, item in enumerate(items):
        if item == target:
            return index

    return None


def tool_use_requests_shell_access(
    message: "PromptMessageExtended",
    *,
    shell_tool_name: str | None = None,
    assume_execute_is_shell: bool = False,
) -> bool:
    """Return True when this TOOL_USE turn only requests local shell execution."""
    if message.stop_reason != LlmStopReason.TOOL_USE:
        return False

    tool_calls = message.tool_calls
    if not tool_calls:
        return False

    built_in_aliases = {"bash", "zsh", "sh", "pwsh", "powershell", "cmd", "shell"}
    if assume_execute_is_shell:
        built_in_aliases.add("execute")

    def _is_shell_tool(tool_name: str) -> bool:
        normalized = tool_name.lower()
        for sep in ("/", ".", ":"):
            if sep in normalized:
                normalized = normalized.rsplit(sep, 1)[-1]

        if shell_tool_name and tool_name == shell_tool_name:
            return True

        return normalized in built_in_aliases

    for call in tool_calls.values():
        params = getattr(call, "params", None)
        tool_name = getattr(params, "name", None) if params is not None else None
        if not tool_name or not _is_shell_tool(tool_name):
            return False

    return True


def tool_use_requests_file_read_access(
    message: "PromptMessageExtended",
    *,
    read_tool_name: str | None = None,
) -> bool:
    """Return True when this TOOL_USE turn only requests read_text_file calls."""
    if message.stop_reason != LlmStopReason.TOOL_USE:
        return False

    tool_calls = message.tool_calls
    if not tool_calls:
        return False

    def _is_read_tool(tool_name: str) -> bool:
        normalized = tool_name.lower()
        for sep in ("/", ".", ":"):
            if sep in normalized:
                normalized = normalized.rsplit(sep, 1)[-1]

        if read_tool_name and tool_name == read_tool_name:
            return True

        return normalized == "read_text_file" or normalized.endswith("__read_text_file")

    for call in tool_calls.values():
        params = getattr(call, "params", None)
        tool_name = getattr(params, "name", None) if params is not None else None
        if not tool_name or not _is_read_tool(tool_name):
            return False

    return True


__all__ = [
    "build_tool_use_additional_message",
    "build_user_message_display",
    "extract_user_attachments",
    "resolve_highlight_index",
    "tool_use_requests_file_read_access",
    "tool_use_requests_shell_access",
]
