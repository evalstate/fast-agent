from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp_types import ListToolsResult

    from fast_agent.mcp.mcp_aggregator import MCPToolCatalog, NamespacedTool


@dataclass(frozen=True, slots=True)
class McpToolRoute:
    requested_name: str
    namespaced_tool: NamespacedTool | None
    candidate_namespaced_tool: NamespacedTool | None
    route_to_namespaced_candidate: bool

    @property
    def active_namespaced_tool(self) -> NamespacedTool | None:
        return self.namespaced_tool or self.candidate_namespaced_tool

    @property
    def execution_name(self) -> str:
        # Filesystem collisions explicitly select the MCP implementation. Other
        # unprefixed candidates remain unprefixed for aggregator-side resolution.
        if self.route_to_namespaced_candidate and self.candidate_namespaced_tool is not None:
            return self.candidate_namespaced_tool.namespaced_tool_name
        return self.requested_name

    @property
    def display_name(self) -> str:
        active_tool = self.active_namespaced_tool
        return active_tool.namespaced_tool_name if active_tool is not None else self.requested_name


def build_mcp_tool_route(
    *,
    requested_name: str,
    catalog: MCPToolCatalog,
    local_tool_exists: bool,
    is_filesystem_runtime_tool: bool,
) -> McpToolRoute:
    namespaced_tool = catalog.namespaced_tool(requested_name)
    candidate = (
        None
        if namespaced_tool is not None or local_tool_exists
        else catalog.first_tool_named(requested_name)
    )
    return McpToolRoute(
        requested_name=requested_name,
        namespaced_tool=namespaced_tool,
        candidate_namespaced_tool=candidate,
        # Model-visible filesystem tools may collide with discovered MCP tools;
        # planning intentionally gives the remote tool the explicit route.
        route_to_namespaced_candidate=(
            namespaced_tool is None and candidate is not None and is_filesystem_runtime_tool
        ),
    )


@dataclass(frozen=True, slots=True)
class PlannedMcpToolCall:
    correlation_id: str
    route: McpToolRoute
    tool_args: dict[str, Any]
    bottom_items: list[str] | None
    highlight_indexes: list[int]
    source_label: str | None
    server_name: str | None
    is_local_shell: bool = False
    metadata: dict[str, Any] | None = None

    @property
    def tool_name(self) -> str:
        return self.route.requested_name

    @property
    def execution_tool_name(self) -> str:
        return self.route.execution_name

    @property
    def display_tool_name(self) -> str:
        return self.route.display_name

    @property
    def namespaced_tool(self) -> NamespacedTool | None:
        return self.route.namespaced_tool

    @property
    def candidate_namespaced_tool(self) -> NamespacedTool | None:
        return self.route.candidate_namespaced_tool


def listed_tool_names(listed_tools: ListToolsResult) -> list[str]:
    return list(dict.fromkeys(tool.name for tool in listed_tools.tools))
