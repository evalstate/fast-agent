"""Protocol for terminal-like runtimes injected into agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from mcp.types import CallToolResult, Tool


class ExternalRuntime(Protocol):
    """Runtime that exposes a single tool and executes it outside MCP."""

    @property
    def tool(self) -> Tool: ...

    async def execute(
        self,
        arguments: dict[str, Any],
        tool_use_id: str | None = None,
    ) -> CallToolResult: ...

    def metadata(self) -> dict[str, Any]: ...
