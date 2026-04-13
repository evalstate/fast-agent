from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp.types import CallToolResult, Tool

    from fast_agent.types import RequestParams


@runtime_checkable
class FilesystemRuntime(Protocol):
    """Protocol for runtimes that expose filesystem tools via ``McpAgent``."""

    @property
    def tools(self) -> Sequence[Tool]: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params: RequestParams | None = None,
    ) -> CallToolResult: ...

    def metadata(self) -> dict[str, Any]: ...
