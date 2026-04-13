from __future__ import annotations

from typing import Any, Literal

from fast_agent.tool_activity_presentation import (
    ToolActivityFamily,
    build_tool_activity_presentation,
)


def item_is_responses_tool(item: Any) -> bool:
    return getattr(item, "type", None) in {
        "function_call",
        "custom_tool_call",
        "tool_search_call",
        "web_search_call",
        "mcp_list_tools",
        "mcp_call",
    }


def responses_tool_name(item: Any) -> str:
    item_type = getattr(item, "type", None)
    if item_type == "tool_search_call":
        return "tool_search"
    if item_type == "web_search_call":
        return "web_search"
    if item_type == "mcp_list_tools":
        server_label = getattr(item, "server_label", None)
        if isinstance(server_label, str) and server_label:
            return f"{server_label}/mcp_list_tools"
        return "mcp_list_tools"
    if item_type == "mcp_call":
        tool_name = getattr(item, "name", None) or getattr(item, "tool_name", None)
        server_label = getattr(item, "server_label", None)
        if isinstance(server_label, str) and server_label and isinstance(tool_name, str) and tool_name:
            return f"{server_label}/{tool_name}"
        return tool_name or "mcp_call"
    return getattr(item, "name", None) or "tool"


def responses_tool_use_id(item: Any, index: int | None, item_id: str | None = None) -> str:
    tool_use = getattr(item, "call_id", None) or getattr(item, "id", None) or item_id
    if isinstance(tool_use, str) and tool_use:
        return tool_use
    suffix = str(index) if index is not None else "unknown"
    item_type = getattr(item, "type", None) or "tool"
    return f"{item_type}-{suffix}"


def tool_family_for_item_type(item_type: str | None) -> ToolActivityFamily:
    if item_type == "tool_search_call":
        return "remote_tool_search"
    if item_type == "web_search_call":
        return "web_search"
    if item_type == "mcp_list_tools":
        return "remote_tool_listing"
    if item_type == "mcp_call":
        return "remote_tool"
    return "tool"


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


def fallback_tool_spec(item: Any, index: int) -> tuple[str, str, ToolActivityFamily]:
    item_type = getattr(item, "type", None)
    if item_type == "tool_search_call":
        return (
            "tool_search",
            getattr(item, "call_id", None) or getattr(item, "id", None) or f"tool-{index}",
            "remote_tool_search",
        )
    if item_type == "web_search_call":
        return ("web_search", getattr(item, "id", None) or f"tool-{index}", "web_search")
    if item_type == "mcp_list_tools":
        return (
            "mcp_list_tools",
            getattr(item, "id", None) or f"tool-{index}",
            "remote_tool_listing",
        )
    if item_type == "mcp_call":
        return (
            responses_tool_name(item),
            getattr(item, "call_id", None) or getattr(item, "id", None) or f"tool-{index}",
            "remote_tool",
        )
    return (
        getattr(item, "name", None) or "tool",
        getattr(item, "call_id", None) or getattr(item, "id", None) or f"tool-{index}",
        "tool",
    )
