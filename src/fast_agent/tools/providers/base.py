"""
Common interfaces and utilities for tool providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mcp.types import CallToolResult, Tool


@runtime_checkable
class ToolProvider(Protocol):
    """Protocol for components that expose tools to agents."""

    name: str

    async def list_tools(self) -> List[Tool]:
        """Return the tools exposed by this provider."""
        ...

    def can_handle_tool(self, tool_name: str) -> bool:
        """Return True if the provider can execute the specified tool."""
        ...

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any] | None = None
    ) -> CallToolResult:
        """Execute the named tool."""
        ...

    async def list_prompts(self) -> Dict[str, List[Any]]:
        """Return prompts grouped by provider-specific namespace."""
        ...

    async def list_resources(self) -> Dict[str, List[Any]]:
        """Return resources grouped by provider-specific namespace."""
        ...

    def summary_counts(self) -> Dict[str, int]:
        """Return summary counts for UI display."""
        ...


class BaseToolProvider:
    """Base class providing default behaviour for ToolProvider implementers."""

    name: str = "provider"

    async def list_tools(self) -> list[Tool]:  # pragma: no cover - default behaviour
        return []

    def can_handle_tool(self, tool_name: str) -> bool:  # pragma: no cover - default behaviour
        return False

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:  # pragma: no cover - default behaviour
        raise NotImplementedError

    async def list_prompts(self) -> dict[str, list[Any]]:  # pragma: no cover - default behaviour
        return {}

    async def list_resources(self) -> dict[str, list[Any]]:  # pragma: no cover - default behaviour
        return {}

    def summary_counts(self) -> dict[str, int]:  # pragma: no cover - default behaviour
        return {}
