from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fast_agent.constants import HUMAN_INPUT_TOOL_NAME
from fast_agent.mcp.tool_result_metadata import (
    update_tool_result_display_metadata,
)
from fast_agent.ui.message_display_helpers import resolve_highlight_indexes
from fast_agent.utils.numeric import positive_int_or_none
from fast_agent.utils.tool_names import is_read_text_file_tool_name

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mcp_types import CallToolResult

    from fast_agent.agents.mcp_tool_planning import McpToolRoute
    from fast_agent.mcp.mcp_aggregator import MCPToolCatalog
    from fast_agent.mcp.tool_result_metadata import ToolResultDisplayMetadata


def unique_preserving_order(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


@dataclass(frozen=True, slots=True)
class McpToolPresentation:
    display_name: str
    bottom_items: list[str] | None
    highlight_indexes: list[int]


def build_mcp_tool_presentation(
    route: McpToolRoute,
    catalog: MCPToolCatalog,
    *,
    local_tool_names: Iterable[str] | None,
    fallback_order: Iterable[str],
    display_name_overrides: Mapping[str, str],
) -> McpToolPresentation:
    active_tool = route.active_namespaced_tool
    bottom_items: list[str] | None = None
    highlight_target: str | None = None

    if active_tool is not None:
        bottom_items = unique_preserving_order(catalog.server_tool_names(active_tool.server_name))
        bottom_items = bottom_items or None
        highlight_target = active_tool.tool.name
    elif local_tool_names is not None:
        bottom_items = unique_preserving_order(local_tool_names)
        highlight_target = route.requested_name
    elif route.requested_name == HUMAN_INPUT_TOOL_NAME:
        bottom_items = [HUMAN_INPUT_TOOL_NAME]
        highlight_target = HUMAN_INPUT_TOOL_NAME

    highlight_indexes = resolve_highlight_indexes(bottom_items, highlight_target)
    if bottom_items is None:
        bottom_items = unique_preserving_order(fallback_order) or None
        if bottom_items is not None:
            fallback_target = (
                route.display_name if route.display_name in bottom_items else route.requested_name
            )
            highlight_indexes = resolve_highlight_indexes(bottom_items, fallback_target)

    if bottom_items is not None:
        bottom_items = [display_name_overrides.get(name, name) for name in bottom_items]

    return McpToolPresentation(
        display_name=route.display_name,
        bottom_items=bottom_items,
        highlight_indexes=highlight_indexes,
    )


def attach_read_text_file_display_metadata(
    result: CallToolResult,
    *,
    display_tool_name: str,
    tool_args: Mapping[str, Any],
) -> None:
    if not is_read_text_file_tool_name(display_tool_name):
        return

    path = tool_args.get("path")
    if not isinstance(path, str) or not (path := path.strip()):
        return

    metadata: ToolResultDisplayMetadata = {"read_text_file_path": path}
    if line := positive_int_or_none(tool_args.get("line")):
        metadata["read_text_file_line"] = line
    if limit := positive_int_or_none(tool_args.get("limit")):
        metadata["read_text_file_limit"] = limit
    update_tool_result_display_metadata(result, metadata)


def tool_result_type_label(display_tool_name: str) -> str | None:
    return "file read" if is_read_text_file_tool_name(display_tool_name) else None
