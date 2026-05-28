from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.types import CallToolResult, Tool

    from fast_agent.tools.filesystem_runtime_protocol import FilesystemRuntime
    from fast_agent.types import RequestParams


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
        merged: list[Tool] = []
        seen: set[str] = set()
        for runtime in (self.primary, self.fallback):
            for tool in runtime.tools:
                if tool.name in seen:
                    continue
                merged.append(tool)
                seen.add(tool.name)
        return merged

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
            from mcp.types import CallToolResult, TextContent

            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Error: unsupported filesystem tool '{name}'",
                    )
                ],
                isError=True,
            )

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
    if any(tool.name == tool_name for tool in primary.tools):
        return primary
    if any(tool.name == tool_name for tool in fallback.tools):
        return fallback
    return None
