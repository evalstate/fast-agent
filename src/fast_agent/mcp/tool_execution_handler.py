"""
Tool execution handler protocol for MCP aggregator.

Provides a clean interface for hooking into tool execution lifecycle,
similar to how elicitation handlers work.
"""

from typing import Protocol, runtime_checkable

from mcp.types import ContentBlock


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

    async def on_tool_permission_denied(
        self,
        tool_name: str,
        server_name: str,
        tool_use_id: str | None,
        error: str | None = None,
    ) -> None:
        """
        Optional hook invoked when tool execution is denied before start.

        Implementations can use this to notify external systems (e.g., ACP)
        that a tool call was cancelled or declined.
        """
        ...

    async def get_tool_call_id_for_tool_use(self, tool_use_id: str) -> str | None:
        """
        Get the ACP toolCallId for a given LLM tool_use_id.

        This allows callers to look up an existing tool_call_id (e.g., from
        streaming notifications) before on_tool_start is called.

        Args:
            tool_use_id: The LLM's tool use ID

        Returns:
            The toolCallId if one exists for this tool_use_id, None otherwise
        """
        ...

    async def ensure_tool_call_exists(
        self,
        tool_use_id: str,
        tool_name: str,
        server_name: str,
        arguments: dict | None = None,
    ) -> str:
        """
        Ensure a tool call notification exists for the given tool_use_id.

        If a notification was already created (e.g., via streaming), returns that toolCallId.
        Otherwise creates a new pending notification with the provided info.

        This handles the non-streaming case where tool calls arrive in one chunk
        and we need to ensure the notification exists before sending diffs.

        Args:
            tool_use_id: The LLM's tool use ID
            tool_name: Name of the tool being called
            server_name: Name of the server providing the tool
            arguments: Tool arguments (for display)

        Returns:
            The ACP toolCallId (existing or newly created)
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

    async def on_tool_permission_denied(
        self,
        tool_name: str,
        server_name: str,
        tool_use_id: str | None,
        error: str | None = None,
    ) -> None:
        """No-op - does nothing."""
        pass

    async def get_tool_call_id_for_tool_use(self, tool_use_id: str) -> str | None:
        """No-op - always returns None."""
        return None

    async def ensure_tool_call_exists(
        self,
        tool_use_id: str,
        tool_name: str,
        server_name: str,
        arguments: dict | None = None,
    ) -> str:
        """No-op - generates a simple UUID."""
        import uuid
        return str(uuid.uuid4())
