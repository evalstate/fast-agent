"""
Tool permission handler protocol for MCP aggregator.

Provides a clean interface for hooking into tool permission checks,
allowing permission systems (like ACP) to be integrated without tight coupling.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolPermissionResult:
    """Result of a tool permission check."""

    allowed: bool
    """Whether the tool execution is permitted."""

    remember: bool = False
    """Whether this decision was remembered (from cache/persistence)."""

    is_cancelled: bool = False
    """Whether the permission request was cancelled by the user."""

    error_message: str | None = None
    """Optional error message to return to the LLM when permission is denied."""

    @classmethod
    def allow(cls) -> "ToolPermissionResult":
        """Create an allowed result."""
        return cls(allowed=True)

    @classmethod
    def deny(cls, message: str | None = None) -> "ToolPermissionResult":
        """Create a denied result with optional error message."""
        return cls(allowed=False, error_message=message)

    @classmethod
    def cancelled(cls) -> "ToolPermissionResult":
        """Create a cancelled result."""
        return cls(allowed=False, is_cancelled=True, error_message="Permission request cancelled")


@runtime_checkable
class ToolPermissionHandler(Protocol):
    """
    Protocol for handling tool permission checks.

    Implementations can check permissions via various mechanisms:
    - ACP session/request_permission
    - Local permission store
    - Custom permission logic
    """

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> ToolPermissionResult:
        """
        Check if tool execution is permitted.

        Args:
            tool_name: Name of the tool to execute
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments
            tool_use_id: LLM's tool use ID (for tracking)

        Returns:
            ToolPermissionResult indicating whether execution is allowed
        """
        ...


class NoOpToolPermissionHandler(ToolPermissionHandler):
    """Default no-op handler that allows all tool executions."""

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> ToolPermissionResult:
        """Always allows tool execution."""
        return ToolPermissionResult.allow()
