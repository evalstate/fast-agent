from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeGuard

from fast_agent.utils.action_normalization import normalize_action_token
from fast_agent.utils.text import strip_casefold

ToolActivityFamily = Literal[
    "tool",
    "remote_tool",
    "web_search",
    "remote_tool_search",
    "remote_tool_listing",
]
ToolActivityPhase = Literal["call", "result"]
TOOL_ACTIVITY_FAMILIES: tuple[ToolActivityFamily, ...] = (
    "tool",
    "remote_tool",
    "web_search",
    "remote_tool_search",
    "remote_tool_listing",
)
REMOTE_STATUS_TOOL_FAMILIES: tuple[ToolActivityFamily, ...] = (
    "remote_tool_search",
    "remote_tool_listing",
)
PRESERVE_SECTION_TOOL_FAMILIES: tuple[ToolActivityFamily, ...] = (
    "remote_tool",
    "remote_tool_search",
)

_REMOTE_TOOL_SEARCH_LABEL = "Deferred tool search"
_REMOTE_TOOL_LISTING_LABEL = "Loading remote tools"
_WEB_SEARCH_LABEL = "Searching the web"

_DISPLAY_NAME_BY_FAMILY: dict[ToolActivityFamily, str] = {
    "web_search": _WEB_SEARCH_LABEL,
    "remote_tool_search": _REMOTE_TOOL_SEARCH_LABEL,
    "remote_tool_listing": _REMOTE_TOOL_LISTING_LABEL,
}

_TYPE_LABEL_PREFIX_BY_FAMILY: dict[ToolActivityFamily, str] = {
    "remote_tool": "remote tool",
    "tool": "tool",
}

_FAMILY_BY_PROVIDER_TOOL_TYPE: dict[str, ToolActivityFamily] = {
    "tool_search_call": "remote_tool_search",
    "tool_search_output": "remote_tool_search",
    "x_search_call": "remote_tool",
    "web_search_call": "web_search",
    "mcp_list_tools": "remote_tool_listing",
    "mcp_call": "remote_tool",
}

_FAMILY_BY_TOOL_NAME: dict[str, ToolActivityFamily] = {
    "tool_search": "remote_tool_search",
    "web_search": "web_search",
    "web_search_call": "web_search",
    "mcp_list_tools": "remote_tool_listing",
}

_STATUS_TEXT_BY_FAMILY: dict[ToolActivityFamily, dict[str, str]] = {
    "remote_tool_search": {
        "in_progress": "searching deferred tools...",
        "completed": "deferred tool search complete",
        "failed": "deferred tool search failed",
    },
    "web_search": {
        "in_progress": "starting search...",
        "searching": "searching...",
        "completed": "search complete",
        "failed": "search failed",
    },
    "remote_tool_listing": {
        "in_progress": "loading remote tools...",
        "completed": "remote tools loaded",
        "failed": "failed to load remote tools",
    },
    "remote_tool": {
        "in_progress": "calling remote tool...",
        "completed": "remote tool call complete",
        "failed": "remote tool call failed",
    },
}

_GENERIC_STATUS_TEXT = {
    "in_progress": "starting...",
    "queued": "queued...",
    "started": "started...",
    "searching": "searching...",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "incomplete": "incomplete",
}


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
    normalized_type = normalize_action_token(provider_tool_type)
    normalized_name = normalize_action_token(tool_name)

    provider_family = _FAMILY_BY_PROVIDER_TOOL_TYPE.get(normalized_type)
    if provider_family is not None:
        return provider_family

    tool_name_family = _FAMILY_BY_TOOL_NAME.get(normalized_name)
    if tool_name_family is not None:
        return tool_name_family

    if normalized_name.endswith("/mcp_list_tools"):
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
    return family in PRESERVE_SECTION_TOOL_FAMILIES


def tool_activity_family_uses_status_body(family: ToolActivityFamily | None) -> bool:
    return family in REMOTE_STATUS_TOOL_FAMILIES


def is_tool_activity_family(value: object) -> TypeGuard[ToolActivityFamily]:
    return value in TOOL_ACTIVITY_FAMILIES


def tool_activity_status_text(*, family: ToolActivityFamily, status: str) -> str:
    return _STATUS_TEXT_BY_FAMILY.get(family, {}).get(status, _generic_status_text(status))


def _display_name(*, tool_name: str, family: ToolActivityFamily) -> str:
    if family in _DISPLAY_NAME_BY_FAMILY:
        return _DISPLAY_NAME_BY_FAMILY[family]
    if family == "remote_tool":
        return f"remote tool: {tool_name.split('/', 1)[-1]}"
    return tool_name


def _type_label(*, family: ToolActivityFamily, phase: ToolActivityPhase | None) -> str | None:
    if phase is None:
        return None
    prefix = _TYPE_LABEL_PREFIX_BY_FAMILY.get(family)
    return f"{prefix} {phase}" if prefix else None


def _generic_status_text(status: str) -> str:
    normalized = strip_casefold(status)
    if not normalized:
        return ""
    return _GENERIC_STATUS_TEXT.get(normalized, normalized.replace("_", " "))
