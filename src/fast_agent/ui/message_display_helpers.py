from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rich.text import Text

from fast_agent.constants import FAST_AGENT_SAFETY_DETAILS
from fast_agent.mcp.helpers.content_helpers import (
    get_text,
    is_image_content,
    is_resource_content,
    is_resource_link,
)
from fast_agent.types import LlmStopReason
from fast_agent.utils.tool_names import (
    EXECUTE_TOOL_NAME,
    POLL_PROCESS_TOOL_NAME,
    SHELL_BUILTIN_TOOL_NAMES,
    TERMINATE_PROCESS_TOOL_NAME,
    is_read_text_file_tool_name,
    matches_tool_name,
    normalize_tool_name,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mcp.types import CallToolRequest, ContentBlock

    from fast_agent.types import PromptMessageExtended
    from fast_agent.ui.terminal_images import ImageRenderItem


@runtime_checkable
class _MetadataContent(Protocol):
    meta: object


def _content_metadata(content: object) -> "Mapping[str, object] | None":
    if not isinstance(content, _MetadataContent):
        return None
    meta = content.meta
    if not isinstance(meta, dict):
        return None
    return {key: value for key, value in meta.items() if isinstance(key, str)}


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


def extract_user_local_image_previews(message: PromptMessageExtended) -> list["ImageRenderItem"]:
    """Extract renderable previews for local image attachments in a user message."""
    image_blocks = [content for content in message.content if _is_local_image_content(content)]
    if not image_blocks:
        return []

    from fast_agent.ui.terminal_images import extract_image_render_items

    return extract_image_render_items(image_blocks)


def build_user_message_image_previews(
    messages: Sequence[PromptMessageExtended],
) -> list["ImageRenderItem"]:
    previews: list[ImageRenderItem] = []
    for message in messages:
        previews.extend(extract_user_local_image_previews(message))
    return previews


def _content_source_uri(content: object) -> str | None:
    meta = _content_metadata(content)
    if meta is None:
        return None
    source_uri = meta.get("fast_agent_source_uri")
    return source_uri if isinstance(source_uri, str) and source_uri else None


def _is_local_image_content(content: "ContentBlock") -> bool:
    if not is_image_content(content):
        return False
    source_uri = _content_source_uri(content)
    return source_uri is not None and source_uri.startswith("file://")


def _message_display_text(message: PromptMessageExtended) -> str:
    from mcp.types import TextContent

    for content in message.content:
        if not isinstance(content, TextContent):
            continue
        meta = _content_metadata(content)
        if meta is not None:
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
    if tool_use_requests_process_lifecycle(message):
        return None
    if shell_access:
        message_text = "The assistant requested shell access"
    elif file_read:
        return None
    else:
        message_text = "The assistant requested tool calls"
    return Text(message_text, style="dim green italic")


def _safety_details_category(message: "PromptMessageExtended") -> str | None:
    channels = message.channels
    if not channels:
        return None
    detail_blocks = channels.get(FAST_AGENT_SAFETY_DETAILS)
    if not detail_blocks:
        return None
    detail_text = get_text(detail_blocks[0])
    if not detail_text:
        return None
    try:
        payload = json.loads(detail_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    category = payload.get("category")
    return category if isinstance(category, str) and category else None


def build_safety_additional_message(message: "PromptMessageExtended") -> Text | None:
    """Build a user-visible refusal/safety stop message with provider details when present."""
    if message.stop_reason != LlmStopReason.SAFETY:
        return None
    category = _safety_details_category(message)
    suffix = f" ({category})" if category else ""
    return Text(f"\n\nRequest refused by safety classifier{suffix}.", style="dim red italic")


def resolve_highlight_indexes(
    items: "Sequence[str] | None",
    highlight_items: str | "Sequence[str]" | None,
) -> list[int]:
    """Resolve highlighted item names to indexes in a displayed list."""
    if not items or highlight_items is None:
        return []

    if isinstance(highlight_items, str):
        if not highlight_items:
            return []
        targets = {highlight_items}
    else:
        if not highlight_items:
            return []
        targets = set(highlight_items)

    return [index for index, item in enumerate(items) if item in targets]


def _tool_call_name(call: "CallToolRequest") -> str:
    return call.params.name or ""


def _tool_use_requests_only(
    message: "PromptMessageExtended",
    predicate: Callable[[str], bool],
) -> bool:
    if message.stop_reason != LlmStopReason.TOOL_USE:
        return False

    tool_calls = message.tool_calls
    if not tool_calls:
        return False

    return all(
        (tool_name := _tool_call_name(call)) and predicate(tool_name)
        for call in tool_calls.values()
    )


def tool_use_requests_shell_access(
    message: "PromptMessageExtended",
    *,
    shell_tool_name: str | None = None,
    assume_execute_is_shell: bool = False,
) -> bool:
    """Return True when this TOOL_USE turn only requests local shell execution."""
    built_in_aliases = set(SHELL_BUILTIN_TOOL_NAMES)
    if assume_execute_is_shell:
        built_in_aliases.add(EXECUTE_TOOL_NAME)

    def _is_shell_tool(tool_name: str) -> bool:
        normalized = normalize_tool_name(tool_name)

        if shell_tool_name and matches_tool_name(tool_name, shell_tool_name):
            return True

        return normalized in built_in_aliases

    return _tool_use_requests_only(message, _is_shell_tool)


def tool_use_requests_file_read_access(
    message: "PromptMessageExtended",
    *,
    read_tool_name: str | None = None,
) -> bool:
    """Return True when this TOOL_USE turn only requests read_text_file calls."""

    def _is_read_tool(tool_name: str) -> bool:
        if read_tool_name and matches_tool_name(tool_name, read_tool_name):
            return True

        return is_read_text_file_tool_name(tool_name)

    return _tool_use_requests_only(message, _is_read_tool)


def tool_use_requests_process_lifecycle(message: "PromptMessageExtended") -> bool:
    """Return True for turns containing only managed-process lifecycle calls."""

    def _is_process_lifecycle_tool(tool_name: str) -> bool:
        return normalize_tool_name(tool_name) in {
            POLL_PROCESS_TOOL_NAME,
            TERMINATE_PROCESS_TOOL_NAME,
        }

    return _tool_use_requests_only(message, _is_process_lifecycle_tool)


__all__ = [
    "build_safety_additional_message",
    "build_tool_use_additional_message",
    "build_user_message_display",
    "build_user_message_image_previews",
    "extract_user_attachments",
    "extract_user_local_image_previews",
    "resolve_highlight_indexes",
    "tool_use_requests_file_read_access",
    "tool_use_requests_process_lifecycle",
    "tool_use_requests_shell_access",
]
