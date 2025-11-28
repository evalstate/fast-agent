"""
Tool execution handler protocol for MCP aggregator.

Provides a clean interface for hooking into tool execution lifecycle,
similar to how elicitation handlers work.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from mcp.types import ContentBlock


@dataclass
class ToolPermissionCheckResult:
    """
    Result of a permission check for tool execution.

    Used to communicate the outcome of a permission check from the
    ToolPermissionHandler to the MCP aggregator.
    """

    allowed: bool
    """Whether the tool execution is permitted."""

    error_message: str | None = None
    """Optional error message to return if not allowed."""

    def __bool__(self) -> bool:
        """Allow boolean evaluation for simple checks."""
        return self.allowed

    @classmethod
    def allow(cls) -> "ToolPermissionCheckResult":
        """Create a result that permits execution."""
        return cls(allowed=True)

    @classmethod
    def deny(cls, message: str = "Tool execution denied") -> "ToolPermissionCheckResult":
        """Create a result that denies execution."""
        return cls(allowed=False, error_message=message)


@runtime_checkable
class ToolPermissionHandler(Protocol):
    """
    Protocol for checking tool execution permissions.

    Implementations can hook into tool execution to request user permission
    before tools are executed (e.g., for ACP permission requests).
    """

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None,
        tool_call_id: str | None = None,
    ) -> ToolPermissionCheckResult:
        """
        Check if a tool execution is permitted.

        This is called BEFORE tool execution begins. If the result
        indicates denial, the tool will not be executed and an error
        result will be returned to the LLM.

        Args:
            tool_name: Name of the tool to execute
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments
            tool_call_id: Optional tool call ID for tracking

        Returns:
            ToolPermissionCheckResult indicating whether to proceed
        """
        ...


class NoOpToolPermissionHandler(ToolPermissionHandler):
    """Default handler that allows all tool executions."""

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None,
        tool_call_id: str | None = None,
    ) -> ToolPermissionCheckResult:
        """Always allow tool execution."""
        return ToolPermissionCheckResult.allow()


@runtime_checkable
class ToolExecutionHandler(Protocol):
    """
    Protocol for handling tool execution lifecycle events.

    Implementations can hook into tool execution to track progress,
    request permissions, or send notifications (e.g., for ACP).
    """

    async def on_tool_start(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict | None,
        tool_use_id: str | None = None,
    ) -> str:
        """
        Called when a tool execution starts.

        Args:
            tool_name: Name of the tool being called
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments
            tool_use_id: LLM's tool use ID (for matching with stream events)

        Returns:
            A unique tool_call_id for tracking this execution
        """
        ...

    async def on_tool_progress(
        self,
        tool_call_id: str,
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        """
        Called when tool execution reports progress.

        Args:
            tool_call_id: The tracking ID from on_tool_start
            progress: Current progress value
            total: Total value for progress calculation (optional)
            message: Optional progress message
        """
        ...

    async def on_tool_complete(
        self,
        tool_call_id: str,
        success: bool,
        content: list[ContentBlock] | None,
        error: str | None,
    ) -> None:
        """
        Called when tool execution completes.

        Args:
            tool_call_id: The tracking ID from on_tool_start
            success: Whether the tool executed successfully
            content: Optional content blocks (text, images, etc.) if successful
            error: Optional error message if failed
        """
        ...


class NoOpToolExecutionHandler(ToolExecutionHandler):
    """Default no-op handler that maintains existing behavior."""

    async def on_tool_start(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict | None,
        tool_use_id: str | None = None,
    ) -> str:
        """Generate a simple UUID for tracking."""
        import uuid
        return str(uuid.uuid4())

    async def on_tool_progress(
        self,
        tool_call_id: str,
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        """No-op - does nothing."""
        pass

    async def on_tool_complete(
        self,
        tool_call_id: str,
        success: bool,
        content: list[ContentBlock] | None,
        error: str | None,
    ) -> None:
        """No-op - does nothing."""
        pass
