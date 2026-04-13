from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ToolActivityFamily = Literal[
    "tool",
    "remote_tool",
    "web_search",
    "remote_tool_search",
    "remote_tool_listing",
]
ToolActivityPhase = Literal["call", "result"]

_REMOTE_TOOL_SEARCH_LABEL = "Deferred tool search"
_REMOTE_TOOL_LISTING_LABEL = "Loading remote tools"
_WEB_SEARCH_LABEL = "Searching the web"


@dataclass(slots=True, frozen=True)
class ToolActivityPresentation:
    family: ToolActivityFamily
    display_name: str
    type_label: str | None
    preserve_sections: bool


def classify_tool_activity_family(
    *,
    tool_name: str,
    remote: bool = False,
    provider_tool_type: str | None = None,
    server_name: str | None = None,
) -> ToolActivityFamily:
    normalized_type = provider_tool_type or ""
    normalized_name = tool_name.strip()

    if normalized_type in {"tool_search_call", "tool_search_output"} or normalized_name == "tool_search":
        return "remote_tool_search"
    if normalized_type == "web_search_call" or normalized_name in {"web_search", "web_search_call"}:
        return "web_search"
    if (
        normalized_type == "mcp_list_tools"
        or normalized_name == "mcp_list_tools"
        or normalized_name.endswith("/mcp_list_tools")
    ):
        return "remote_tool_listing"
    if remote or bool(server_name):
        return "remote_tool"
    return "tool"


def build_tool_activity_presentation(
    *,
    tool_name: str,
    phase: ToolActivityPhase | None = None,
    family: ToolActivityFamily | None = None,
    remote: bool = False,
    provider_tool_type: str | None = None,
    server_name: str | None = None,
) -> ToolActivityPresentation:
    resolved_family = family or classify_tool_activity_family(
        tool_name=tool_name,
        remote=remote,
        provider_tool_type=provider_tool_type,
        server_name=server_name,
    )
    return ToolActivityPresentation(
        family=resolved_family,
        display_name=_display_name(tool_name=tool_name, family=resolved_family),
        type_label=_type_label(family=resolved_family, phase=phase),
        preserve_sections=tool_activity_family_preserves_sections(resolved_family),
    )


def tool_activity_family_preserves_sections(family: ToolActivityFamily) -> bool:
    return family in {"remote_tool", "remote_tool_search"}


def tool_activity_status_text(*, family: ToolActivityFamily, status: str) -> str:
    if family == "web_search":
        return _web_search_status_text(status)
    if family == "remote_tool_search":
        return _remote_tool_search_status_text(status)
    if family == "remote_tool_listing":
        return _remote_tool_listing_status_text(status)
    if family == "remote_tool":
        return _remote_tool_status_text(status)
    return _generic_status_text(status)


def _display_name(*, tool_name: str, family: ToolActivityFamily) -> str:
    if family == "web_search":
        return _WEB_SEARCH_LABEL
    if family == "remote_tool_search":
        return _REMOTE_TOOL_SEARCH_LABEL
    if family == "remote_tool_listing":
        return _REMOTE_TOOL_LISTING_LABEL
    if family == "remote_tool":
        return f"remote tool: {tool_name.split('/', 1)[-1]}"
    return tool_name


def _type_label(*, family: ToolActivityFamily, phase: ToolActivityPhase | None) -> str | None:
    if phase is None:
        return None
    if family == "remote_tool":
        return f"remote tool {phase}"
    if family == "tool":
        return f"tool {phase}"
    return None


def _remote_tool_search_status_text(status: str) -> str:
    if status == "in_progress":
        return "searching deferred tools..."
    if status == "completed":
        return "deferred tool search complete"
    if status == "failed":
        return "deferred tool search failed"
    return _generic_status_text(status)


def _web_search_status_text(status: str) -> str:
    if status == "in_progress":
        return "starting search..."
    if status == "searching":
        return "searching..."
    if status == "completed":
        return "search complete"
    if status == "failed":
        return "search failed"
    return _generic_status_text(status)


def _remote_tool_listing_status_text(status: str) -> str:
    if status == "in_progress":
        return "loading remote tools..."
    if status == "completed":
        return "remote tools loaded"
    if status == "failed":
        return "failed to load remote tools"
    return _generic_status_text(status)


def _remote_tool_status_text(status: str) -> str:
    if status == "in_progress":
        return "calling remote tool..."
    if status == "completed":
        return "remote tool call complete"
    if status == "failed":
        return "remote tool call failed"
    return _generic_status_text(status)


def _generic_status_text(status: str) -> str:
    normalized = status.strip().lower()
    if not normalized:
        return ""
    known = {
        "in_progress": "starting...",
        "queued": "queued...",
        "started": "started...",
        "searching": "searching...",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
        "incomplete": "incomplete",
    }
    return known.get(normalized, normalized.replace("_", " "))
