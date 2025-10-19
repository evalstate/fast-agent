"""
Tool provider wrapper for MCP aggregators.
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from fast_agent.tools.providers.base import BaseToolProvider

if TYPE_CHECKING:
    from mcp.types import CallToolResult, Tool

    from fast_agent.mcp.mcp_aggregator import MCPAggregator


class McpToolProvider(BaseToolProvider):
    """Expose MCP aggregator tools to agents via the provider interface."""

    name = "mcp"

    def __init__(
        self,
        aggregator: MCPAggregator,
        *,
        tool_filters: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._aggregator = aggregator
        self._tool_filters = tool_filters
        self._tool_names: set[str] = set()

    async def list_tools(self) -> list[Tool]:
        result = await self._aggregator.list_tools()
        tools = list(result.tools or [])

        if self._tool_filters is not None:
            tools = self._apply_filters(tools, self._tool_filters)

        self._tool_names = {tool.name for tool in tools}
        return tools

    def can_handle_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_names

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        return await self._aggregator.call_tool(tool_name, arguments)

    async def list_prompts(self) -> dict[str, list[Any]]:
        prompts = await self._aggregator.list_prompts()
        return {key: list(value) for key, value in prompts.items()}

    async def list_resources(self) -> dict[str, list[Any]]:
        return await self._aggregator.list_resources()

    def summary_counts(self) -> dict[str, int]:
        return {
            "servers": len(self._aggregator.server_names),
            "tools": len(self._tool_names),
        }

    @staticmethod
    def _apply_filters(tools: Iterable[Tool], filters: Dict[str, List[str]]) -> List[Tool]:
        filtered: List[Tool] = []
        for tool in tools:
            server_name = McpToolProvider._infer_server_name(tool.name, filters.keys())
            if server_name is None:
                continue

            patterns = filters.get(server_name, [])
            for pattern in patterns:
                if McpToolProvider._matches_pattern(tool.name, pattern, server_name):
                    filtered.append(tool)
                    break
        return filtered

    @staticmethod
    def _infer_server_name(tool_name: str, candidates: Iterable[str]) -> Optional[str]:
        for candidate in candidates:
            if tool_name.startswith(f"{candidate}-"):
                return candidate
        return None

    @staticmethod
    def _matches_pattern(name: str, pattern: str, server_name: str) -> bool:
        if name.startswith(f"{server_name}-"):
            full_pattern = f"{server_name}-{pattern}"
            return fnmatch.fnmatch(name, full_pattern)
        return fnmatch.fnmatch(name, pattern)
