"""Normalize tool-like activity across standard and provider-native encodings."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from mcp.types import CallToolResult, ContentBlock, TextContent

from fast_agent.constants import ANTHROPIC_ASSISTANT_RAW_CONTENT, ANTHROPIC_SERVER_TOOLS_CHANNEL
from fast_agent.mcp.helpers.content_helpers import get_text

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


ToolActivityKind = Literal["call", "result"]


@dataclass(slots=True)
class ToolActivity:
    kind: ToolActivityKind
    tool_use_id: str
    tool_name: str
    order: int
    arguments: dict[str, Any] | None = None
    result: CallToolResult | None = None
    is_remote: bool = False

    @property
    def type_label(self) -> str:
        return tool_activity_type_label(kind=self.kind, is_remote=self.is_remote)

    @property
    def is_error(self) -> bool:
        return bool(self.result.isError) if self.result is not None else False


def tool_activity_type_label(*, kind: ToolActivityKind, is_remote: bool) -> str:
    if kind == "call":
        return "remote tool call" if is_remote else "tool call"
    return "remote tool result" if is_remote else "tool result"


def tool_activity_display_title(*, kind: ToolActivityKind, tool_name: str, is_remote: bool) -> str:
    if is_remote:
        return f"remote tool: {tool_name}"
    return f"{tool_activity_type_label(kind=kind, is_remote=is_remote)}: {tool_name}"


def tool_activities_for_message(
    message: "PromptMessageExtended",
    *,
    tool_name_lookup: Mapping[str, str] | None = None,
) -> list[ToolActivity]:
    activities: list[ToolActivity] = []
    order = 0

    tool_calls = getattr(message, "tool_calls", None) or {}
    for tool_use_id, call in tool_calls.items():
        params = getattr(call, "params", None)
        tool_name = getattr(params, "name", None) or getattr(call, "name", None) or tool_use_id
        arguments = getattr(params, "arguments", None) if params is not None else None
        if not isinstance(arguments, Mapping):
            arguments = getattr(call, "arguments", None)
        if not isinstance(arguments, Mapping):
            arguments = {}
        activities.append(
            ToolActivity(
                kind="call",
                tool_use_id=tool_use_id,
                tool_name=str(tool_name),
                order=order,
                arguments=dict(arguments),
            )
        )
        order += 1

    tool_results = getattr(message, "tool_results", None) or {}
    for tool_use_id, result in tool_results.items():
        tool_name = tool_name_lookup.get(tool_use_id, tool_use_id) if tool_name_lookup else tool_use_id
        activities.append(
            ToolActivity(
                kind="result",
                tool_use_id=tool_use_id,
                tool_name=str(tool_name),
                order=order,
                result=result,
            )
        )
        order += 1

    for remote_activity in remote_tool_activities(message):
        activities.append(
            ToolActivity(
                kind=remote_activity.kind,
                tool_use_id=remote_activity.tool_use_id,
                tool_name=remote_activity.tool_name,
                order=order,
                arguments=remote_activity.arguments,
                result=remote_activity.result,
                is_remote=True,
            )
        )
        order += 1

    return activities


def remote_tool_activities(message: "PromptMessageExtended") -> list[ToolActivity]:
    payloads = _remote_tool_payloads(message)
    if not payloads:
        return []

    activities: list[ToolActivity] = []
    tool_names_by_id: dict[str, str] = {}

    for order, payload in enumerate(payloads):
        block_type = payload.get("type")
        if block_type == "mcp_tool_use":
            tool_use_id = payload.get("id")
            tool_name = _remote_tool_name(payload)
            if not isinstance(tool_use_id, str) or tool_name is None:
                continue

            arguments = payload.get("input")
            if not isinstance(arguments, Mapping):
                arguments = {}

            tool_names_by_id[tool_use_id] = tool_name
            activities.append(
                ToolActivity(
                    kind="call",
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    order=order,
                    arguments=dict(arguments),
                    is_remote=True,
                )
            )
            continue

        if block_type != "mcp_tool_result":
            continue

        tool_use_id = payload.get("tool_use_id")
        if not isinstance(tool_use_id, str):
            continue

        tool_name = tool_names_by_id.get(tool_use_id, tool_use_id)
        activities.append(
            ToolActivity(
                kind="result",
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                order=order,
                result=_result_from_payload(payload),
                is_remote=True,
            )
        )

    return activities


def message_tool_call_count(message: "PromptMessageExtended") -> int:
    return sum(1 for activity in tool_activities_for_message(message) if activity.kind == "call")


def message_tool_error_count(
    message: "PromptMessageExtended",
    *,
    tool_name_lookup: Mapping[str, str] | None = None,
) -> int:
    return sum(
        1
        for activity in tool_activities_for_message(message, tool_name_lookup=tool_name_lookup)
        if activity.kind == "result" and activity.is_error
    )


def message_tool_success_count(
    message: "PromptMessageExtended",
    *,
    tool_name_lookup: Mapping[str, str] | None = None,
) -> int:
    return sum(
        1
        for activity in tool_activities_for_message(message, tool_name_lookup=tool_name_lookup)
        if activity.kind == "result" and not activity.is_error
    )


def display_remote_tool_activities(
    display: Any,
    message: "PromptMessageExtended",
    *,
    name: str | None = None,
    truncate_content: bool = True,
) -> bool:
    activities = remote_tool_activities(message)
    if not activities:
        return False

    for activity in activities:
        if activity.kind == "call":
            display.show_tool_call(
                tool_name=activity.tool_name,
                tool_args=activity.arguments or {},
                name=name,
                tool_call_id=activity.tool_use_id,
                type_label=activity.type_label,
            )
            continue

        if activity.result is None:
            continue
        display.show_tool_result(
            result=activity.result,
            name=name,
            tool_name=activity.tool_name,
            tool_call_id=activity.tool_use_id,
            type_label=activity.type_label,
            truncate_content=truncate_content,
        )

    return True


def _remote_tool_name(payload: Mapping[str, Any]) -> str | None:
    raw_name = payload.get("name")
    if not isinstance(raw_name, str):
        return None
    raw_server_name = payload.get("server_name")
    if isinstance(raw_server_name, str) and raw_server_name:
        return f"{raw_server_name}/{raw_name}"
    return raw_name


def _remote_tool_payloads(message: "PromptMessageExtended") -> list[dict[str, Any]]:
    channels = getattr(message, "channels", None)
    if not isinstance(channels, Mapping):
        return []

    raw_payloads = _decode_channel_payloads(channels.get(ANTHROPIC_ASSISTANT_RAW_CONTENT))
    mcp_payloads = [payload for payload in raw_payloads if _is_mcp_payload(payload)]
    if mcp_payloads:
        return mcp_payloads

    fallback_payloads = _decode_channel_payloads(channels.get(ANTHROPIC_SERVER_TOOLS_CHANNEL))
    return [payload for payload in fallback_payloads if _is_mcp_payload(payload)]


def _decode_channel_payloads(blocks: Sequence[Any] | None) -> list[dict[str, Any]]:
    if not blocks:
        return []

    payloads: list[dict[str, Any]] = []
    for block in blocks:
        raw_text = get_text(block)
        if not raw_text:
            continue
        try:
            payload = json.loads(raw_text)
        except Exception:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _is_mcp_payload(payload: Mapping[str, Any]) -> bool:
    block_type = payload.get("type")
    return block_type == "mcp_tool_use" or block_type == "mcp_tool_result"


def _result_from_payload(payload: Mapping[str, Any]) -> CallToolResult:
    raw_content = payload.get("content")
    content: list[ContentBlock] = []
    if isinstance(raw_content, Sequence) and not isinstance(raw_content, (str, bytes)):
        for item in raw_content:
            content.append(_content_from_payload(item))

    raw_is_error = payload.get("is_error")
    if not isinstance(raw_is_error, bool):
        raw_is_error = payload.get("isError")

    return CallToolResult(content=content, isError=bool(raw_is_error))


def _content_from_payload(payload: object) -> ContentBlock:
    if isinstance(payload, Mapping):
        payload_map = {key: value for key, value in payload.items() if isinstance(key, str)}
        block_type = payload_map.get("type")
        if block_type == "text":
            text = payload_map.get("text")
            if isinstance(text, str):
                return TextContent(type="text", text=text)
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(payload)
    return TextContent(type="text", text=text)
