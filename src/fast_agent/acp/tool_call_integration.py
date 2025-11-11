"""
Integration module for tool calls with MCP and ACP.

This module provides middleware to intercept MCP tool calls and send
notifications to ACP clients according to the protocol specification.
"""

from typing import Any, Callable, Optional

from acp.schema import ToolCallProgress
from mcp.types import CallToolResult

from fast_agent.acp.tool_call_permission_handler import (
    AlwaysAllowPermissionHandler,
    ToolCallPermissionHandler,
    ToolCallPermissionRequest,
)
from fast_agent.acp.tool_call_tracker import ToolCallTracker
from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


class ACPToolCallMiddleware:
    """
    Middleware that wraps tool calls to send ACP notifications.

    This class intercepts tool calls from the agent, tracks their lifecycle,
    sends progress notifications, and optionally requests permissions.
    """

    def __init__(
        self,
        tracker: ToolCallTracker,
        permission_handler: Optional[ToolCallPermissionHandler] = None,
        enable_permissions: bool = False,
    ):
        """
        Initialize the middleware.

        Args:
            tracker: Tool call tracker for sending notifications
            permission_handler: Optional handler for permission requests
            enable_permissions: Whether to require permissions for tool calls
        """
        self.tracker = tracker
        self.permission_handler = permission_handler or AlwaysAllowPermissionHandler()
        self.enable_permissions = enable_permissions
        self._tool_permissions_cache: dict[str, bool] = {}

    def create_progress_callback(self, tool_call_id: str) -> Callable:
        """
        Create a progress callback that forwards MCP progress to ACP.

        Args:
            tool_call_id: The tool call ID to update

        Returns:
            Async progress callback function
        """

        async def progress_callback(
            progress: float, total: float | None, message: str | None
        ) -> None:
            """Handle progress notifications from MCP tool execution."""
            # Create ACP progress object
            acp_progress = ToolCallProgress(
                current=int(progress) if progress else 0,
                total=int(total) if total else None,
            )

            # Update the tool call with progress
            await self.tracker.update_tool_call(
                tool_call_id=tool_call_id,
                progress=acp_progress,
            )

            logger.debug(
                f"Progress update for tool call {tool_call_id}: {progress}/{total}",
                name="acp_tool_progress",
                tool_call_id=tool_call_id,
                progress=progress,
                total=total,
            )

        return progress_callback

    async def wrap_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        execute_fn: Callable,
        tool_kind: str = "other",
        tool_title: Optional[str] = None,
    ) -> CallToolResult:
        """
        Wrap a tool call with ACP notifications and permission checking.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            execute_fn: Async function to execute the tool
            tool_kind: Kind of tool (read, edit, delete, etc.)
            tool_title: Human-readable description

        Returns:
            Result of the tool call
        """
        # Generate title if not provided
        if tool_title is None:
            tool_title = f"Calling tool: {tool_name}"

        # Check permissions if enabled
        if self.enable_permissions:
            # Check cache first
            cache_key = f"{tool_name}"
            if cache_key not in self._tool_permissions_cache:
                # Request permission
                permission_request = ToolCallPermissionRequest(
                    tool_name=tool_name,
                    tool_title=tool_title,
                    tool_kind=tool_kind,
                    arguments=arguments,
                    session_id=self.tracker.session_id,
                )

                permission_response = await self.permission_handler.request_permission(
                    permission_request, self.tracker.connection
                )

                if permission_response.cancelled:
                    # Create error result
                    from mcp.types import TextContent

                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text", text="Tool call cancelled by user"
                            )
                        ],
                        isError=True,
                    )

                if not permission_response.allowed:
                    from mcp.types import TextContent

                    return CallToolResult(
                        content=[
                            TextContent(type="text", text="Tool call permission denied")
                        ],
                        isError=True,
                    )

                # Cache the decision if requested
                if permission_response.remember:
                    self._tool_permissions_cache[cache_key] = permission_response.allowed

        # Create tool call and track it
        tool_call_id = await self.tracker.create_tool_call(
            title=tool_title,
            kind=tool_kind,
            tool_name=tool_name,
            arguments=arguments,
        )

        try:
            # Update to in_progress
            await self.tracker.update_tool_call(tool_call_id, status="in_progress")

            # Execute the tool
            result = await execute_fn()

            # Complete the tool call
            await self.tracker.complete_tool_call(
                tool_call_id=tool_call_id,
                content=result.content if hasattr(result, "content") else None,
                raw_output=result.model_dump() if hasattr(result, "model_dump") else None,
                is_error=getattr(result, "isError", False),
            )

            return result

        except Exception as e:
            # Mark as failed
            from mcp.types import TextContent

            error_result = CallToolResult(
                content=[TextContent(type="text", text=f"Tool execution error: {str(e)}")],
                isError=True,
            )

            await self.tracker.complete_tool_call(
                tool_call_id=tool_call_id,
                content=error_result.content,
                is_error=True,
            )

            return error_result


def infer_tool_kind(tool_name: str) -> str:
    """
    Infer the tool kind from the tool name.

    Args:
        tool_name: Name of the tool

    Returns:
        One of: read, edit, delete, move, search, execute, think, fetch, other
    """
    name_lower = tool_name.lower()

    # Common patterns
    if any(word in name_lower for word in ["read", "get", "fetch", "list", "view"]):
        return "read"
    elif any(word in name_lower for word in ["write", "edit", "update", "modify", "patch"]):
        return "edit"
    elif any(word in name_lower for word in ["delete", "remove", "clear"]):
        return "delete"
    elif any(word in name_lower for word in ["move", "rename", "copy"]):
        return "move"
    elif any(word in name_lower for word in ["search", "find", "query", "grep"]):
        return "search"
    elif any(
        word in name_lower
        for word in ["exec", "run", "execute", "command", "shell", "bash"]
    ):
        return "execute"
    elif any(word in name_lower for word in ["think", "reason", "plan"]):
        return "think"
    elif any(word in name_lower for word in ["http", "api", "download", "upload"]):
        return "fetch"
    else:
        return "other"


def create_tool_title(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
    """
    Create a human-readable title for a tool call.

    Args:
        tool_name: Name of the tool
        arguments: Tool arguments

    Returns:
        Human-readable title
    """
    # Remove namespace prefix if present (e.g., "server__tool" -> "tool")
    display_name = tool_name.split("__")[-1] if "__" in tool_name else tool_name

    # Convert snake_case to Title Case
    display_name = display_name.replace("_", " ").title()

    # Add context from arguments if available
    if arguments:
        # Common argument keys to include in title
        context_keys = ["path", "file", "query", "command", "url", "name"]
        for key in context_keys:
            if key in arguments:
                value = arguments[key]
                if isinstance(value, str) and len(value) < 50:
                    return f"{display_name}: {value}"

    return display_name
