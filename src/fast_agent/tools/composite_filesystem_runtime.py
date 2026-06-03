from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, TextContent

from fast_agent.tools.filesystem_tool_definitions import (
    ATTACH_MEDIA_TOOL_NAME,
    ATTACH_RESOURCE_TOOL_ALIAS,
)
from fast_agent.utils.collections import unique_preserve_order

if TYPE_CHECKING:
    from mcp.types import Tool

    from fast_agent.tools.filesystem_runtime_protocol import FilesystemRuntime
    from fast_agent.types import RequestParams


_TOOL_ALIASES = {ATTACH_RESOURCE_TOOL_ALIAS: ATTACH_MEDIA_TOOL_NAME}


def _tool_names(runtime: FilesystemRuntime) -> set[str]:
    return {tool.name for tool in runtime.tools}


def _runtime_supports_tool(runtime: FilesystemRuntime, tool_name: str) -> bool:
    names = _tool_names(runtime)
    return tool_name in names or _TOOL_ALIASES.get(tool_name) in names


def _unsupported_tool_result(name: str) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Error: unsupported filesystem tool '{name}'",
            )
        ],
        isError=True,
    )


class CompositeFilesystemRuntime:
    """Merge ACP-provided filesystem tools with local shell edit tools."""

    def __init__(
        self,
        primary: FilesystemRuntime,
        fallback: FilesystemRuntime,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    @property
    def tools(self) -> list[Tool]:
        return unique_preserve_order(
            [*self.primary.tools, *self.fallback.tools],
            key=lambda tool: tool.name,
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params: RequestParams | None = None,
    ) -> CallToolResult:
        target_runtime = _runtime_for_tool(self.primary, self.fallback, name)
        if target_runtime is None:
            return _unsupported_tool_result(name)

        return await target_runtime.call_tool(
            name,
            arguments,
            tool_use_id,
            request_params=request_params,
        )

    def metadata(self) -> dict[str, Any]:
        primary = self.primary.metadata()
        fallback = self.fallback.metadata()
        tools = [tool.name for tool in self.tools]
        return {
            "type": "composite_filesystem",
            "primary": primary,
            "fallback": fallback,
            "tools": tools,
        }


def _runtime_for_tool(
    primary: FilesystemRuntime,
    fallback: FilesystemRuntime,
    tool_name: str,
) -> FilesystemRuntime | None:
    if _runtime_supports_tool(primary, tool_name):
        return primary
    if _runtime_supports_tool(fallback, tool_name):
        return fallback
    return None
