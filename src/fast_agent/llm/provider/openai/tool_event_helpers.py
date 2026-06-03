from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from fast_agent.event_progress import ProgressAction
from fast_agent.tool_activity_presentation import (
    ToolActivityFamily,
    build_tool_activity_presentation,
    classify_tool_activity_family,
)

_RESPONSES_TOOL_TYPES = frozenset(
    {
        "function_call",
        "custom_tool_call",
        "tool_search_call",
        "web_search_call",
        "mcp_list_tools",
        "mcp_call",
    }
)

_FIXED_TOOL_NAME_BY_ITEM_TYPE = {
    "tool_search_call": "tool_search",
    "web_search_call": "web_search",
}
_RESPONSES_FUNCTION_TOOL_CALL_TYPES = frozenset(("function_call", "custom_tool_call"))
_RESPONSES_FUNCTION_TOOL_STATUS_TYPES = frozenset(
    (
        "function_call",
        "custom_tool_call",
        "function_call_arguments",
        "custom_tool_call_input",
    )
)
_RESPONSES_LIFECYCLE_TOOL_TYPES = frozenset(
    ("tool_search_call", "web_search_call", "mcp_list_tools", "mcp_call")
)
type ToolStreamLifecycleEvent = Literal["start", "stop"]
type ToolLifecycleKind = Literal["start", "stop", "status"]
_TOOL_START_STATUSES = frozenset(("in_progress", "queued", "started", "searching"))
_TOOL_STOP_STATUSES = frozenset(("completed", "failed", "cancelled", "incomplete"))
_TOOL_STATUS_PATTERN = re.compile(r"^response\.(?P<item_type>[^.]+)\.(?P<status>[^.]+)$")

_TOOL_STREAM_LOG_MESSAGE: dict[ToolStreamLifecycleEvent, str] = {
    "start": "Model started streaming tool call",
    "stop": "Model finished streaming tool call",
}
_TOOL_STREAM_FALLBACK_LOG_MESSAGE = "Model emitted fallback tool notification"


@dataclass(frozen=True, slots=True)
class _ResponsesItemInfo:
    item_type: str | None
    name: str | None
    tool_name: str | None
    server_label: str | None
    call_id: str | None
    item_id: str | None


@dataclass(frozen=True, slots=True)
class ResponsesLifecycleEventInfo:
    item_type: str
    tool_name: str
    status: str
    lifecycle: ToolLifecycleKind


def _raw_item_value(item: Any, field: str) -> Any:
    return getattr(item, field, None)


def _string_item_value(item: Any, field: str) -> str | None:
    return first_nonempty_string(_raw_item_value(item, field))


def _responses_item_info(item: Any) -> _ResponsesItemInfo:
    item_type = _raw_item_value(item, "type")
    return _ResponsesItemInfo(
        item_type=item_type if isinstance(item_type, str) else None,
        name=_string_item_value(item, "name"),
        tool_name=_string_item_value(item, "tool_name"),
        server_label=_string_item_value(item, "server_label"),
        call_id=_string_item_value(item, "call_id"),
        item_id=_string_item_value(item, "id"),
    )


def responses_item_type(item: Any) -> str | None:
    return _responses_item_info(item).item_type


def item_is_responses_tool(item: Any) -> bool:
    item_type = responses_item_type(item)
    return item_type in _RESPONSES_TOOL_TYPES


def item_type_is_responses_function_tool_call(item_type: object) -> bool:
    return isinstance(item_type, str) and item_type in _RESPONSES_FUNCTION_TOOL_CALL_TYPES


def responses_tool_name_for_item_type(item_type: str) -> str:
    if item_type in _FIXED_TOOL_NAME_BY_ITEM_TYPE:
        return _FIXED_TOOL_NAME_BY_ITEM_TYPE[item_type]
    if item_type in {"mcp_list_tools", "mcp_call"}:
        return item_type
    return "tool"


def responses_lifecycle_event_info(
    event_type: object,
    *,
    include_function_calls: bool = False,
) -> ResponsesLifecycleEventInfo | None:
    if not isinstance(event_type, str):
        return None
    match = _TOOL_STATUS_PATTERN.match(event_type)
    if match is None:
        return None

    item_type = match.group("item_type")
    lifecycle_tool_types = _RESPONSES_LIFECYCLE_TOOL_TYPES
    if include_function_calls:
        lifecycle_tool_types = lifecycle_tool_types | _RESPONSES_FUNCTION_TOOL_STATUS_TYPES
    if item_type not in lifecycle_tool_types:
        return None

    status = match.group("status")
    lifecycle: ToolLifecycleKind = "status"
    if status in _TOOL_START_STATUSES:
        lifecycle = "start"
    elif status in _TOOL_STOP_STATUSES:
        lifecycle = "stop"
    return ResponsesLifecycleEventInfo(
        item_type=item_type,
        tool_name=responses_tool_name_for_item_type(item_type),
        status=status,
        lifecycle=lifecycle,
    )


def first_nonempty_string(*values: Any) -> str | None:
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def _qualified_mcp_tool_name(info: _ResponsesItemInfo, fallback_name: str) -> str:
    tool_name = info.name or info.tool_name or fallback_name
    if info.server_label:
        return f"{info.server_label}/{tool_name}"
    return tool_name


def responses_tool_name(item: Any) -> str:
    info = _responses_item_info(item)
    if info.item_type in _FIXED_TOOL_NAME_BY_ITEM_TYPE:
        return _FIXED_TOOL_NAME_BY_ITEM_TYPE[info.item_type]
    if info.item_type == "mcp_list_tools":
        return _qualified_mcp_tool_name(info, "mcp_list_tools")
    if info.item_type == "mcp_call":
        return _qualified_mcp_tool_name(info, "mcp_call")
    return info.name or "tool"


def responses_item_tool_use_id(item: Any, item_id: str | None = None) -> str | None:
    info = _responses_item_info(item)
    return first_nonempty_string(info.call_id, info.item_id, item_id)


def responses_event_item_id(event: Any, item: Any | None = None) -> str | None:
    return first_nonempty_string(
        _string_item_value(event, "item_id"),
        _responses_item_info(item).item_id,
    )


def responses_tool_use_id(item: Any, index: int | None, item_id: str | None = None) -> str:
    tool_use = responses_item_tool_use_id(item, item_id)
    if tool_use is not None:
        return tool_use
    suffix = str(index) if index is not None else "unknown"
    item_type = responses_item_type(item) or "tool"
    return f"{item_type}-{suffix}"


def tool_family_for_item_type(item_type: str | None) -> ToolActivityFamily:
    return classify_tool_activity_family(tool_name="", provider_tool_type=item_type)


def tool_presentation_payload(
    *,
    tool_name: str,
    family: ToolActivityFamily,
    phase: Literal["call", "result"],
) -> dict[str, Any]:
    presentation = build_tool_activity_presentation(
        tool_name=tool_name,
        family=family,
        phase=phase,
    )
    return {
        "presentation_family": presentation.family,
        "preserve_details": presentation.preserve_sections,
        "tool_display_name": presentation.display_name,
    }


def tool_event_payload(
    *,
    tool_name: str,
    tool_use_id: str | None,
    index: int,
    family: ToolActivityFamily,
    phase: Literal["call", "result"],
    status: str | None = None,
    chunk: str | None = None,
) -> dict[str, Any]:
    payload = {
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "index": index,
    }
    payload.update(
        tool_presentation_payload(
            tool_name=tool_name,
            family=family,
            phase=phase,
        )
    )
    if status is not None:
        payload["status"] = status
    if chunk is not None:
        payload["chunk"] = chunk
    return payload


def tool_stream_log_record(
    *,
    agent_name: str | None,
    model: str,
    tool_name: str | None,
    tool_use_id: str | None,
    event_type: ToolStreamLifecycleEvent,
    fallback: bool = False,
) -> tuple[str, dict[str, Any]]:
    message = (
        _TOOL_STREAM_FALLBACK_LOG_MESSAGE
        if fallback
        else _TOOL_STREAM_LOG_MESSAGE[event_type]
    )
    data: dict[str, Any] = {
        "progress_action": ProgressAction.CALLING_TOOL,
        "agent_name": agent_name,
        "model": model,
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "tool_event": event_type,
    }
    if event_type == "stop":
        data["tool_terminal"] = True
    if fallback:
        data["fallback"] = True
    return message, data


def fallback_tool_spec(item: Any, index: int) -> tuple[str, str, ToolActivityFamily]:
    item_type = responses_item_type(item)
    return (
        responses_tool_name(item),
        responses_tool_use_id(item, index),
        tool_family_for_item_type(item_type),
    )
