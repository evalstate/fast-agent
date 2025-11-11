"""
Tool Call Manager - Track and manage tool call lifecycle for ACP.

This module provides the ToolCallManager class which:
- Generates unique tool call IDs
- Tracks tool call state (pending, in_progress, completed, failed)
- Maps tool names to ACP ToolKind categories
- Sends tool call notifications (ToolCallStart, ToolCallProgress) via ACP connection
"""

import uuid
from typing import TYPE_CHECKING, Any, Optional

from acp.schema import (
    ContentToolCallContent,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
    ToolCallStatus,
    ToolKind,
)

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)


class ToolCallManager:
    """
    Manages tool call lifecycle for ACP sessions.

    Responsibilities:
    - Generate unique tool call IDs
    - Track tool call state per session
    - Map tool names to appropriate ToolKind
    - Send ToolCallStart and ToolCallProgress notifications
    """

    # Map common tool patterns to ACP ToolKind
    TOOL_KIND_MAPPING: dict[str, ToolKind] = {
        # Read operations
        "read": "read",
        "read_file": "read",
        "read_resource": "read",
        "get": "read",
        "fetch": "fetch",
        "fetch_url": "fetch",
        # Edit operations
        "write": "edit",
        "edit": "edit",
        "edit_file": "edit",
        "update": "edit",
        "patch": "edit",
        # Delete operations
        "delete": "delete",
        "remove": "delete",
        "rm": "delete",
        # Move operations
        "move": "move",
        "mv": "move",
        "rename": "move",
        # Search operations
        "search": "search",
        "find": "search",
        "grep": "search",
        "query": "search",
        # Execute operations
        "execute": "execute",
        "exec": "execute",
        "run": "execute",
        "bash": "execute",
        "shell": "execute",
        "command": "execute",
        # Think operations
        "think": "think",
        "reason": "think",
        "plan": "think",
    }

    def __init__(self, connection: "AgentSideConnection", session_id: str):
        """
        Initialize the tool call manager.

        Args:
            connection: ACP connection for sending notifications
            session_id: Session ID for this manager
        """
        self.connection = connection
        self.session_id = session_id

        # Track active tool calls: {tool_call_id: {status, tool_name, server_name, ...}}
        self.active_calls: dict[str, dict[str, Any]] = {}

    def infer_tool_kind(self, tool_name: str) -> ToolKind:
        """
        Infer the ACP ToolKind from a tool name.

        Args:
            tool_name: Name of the tool (e.g., "read_file", "execute_bash")

        Returns:
            Appropriate ToolKind, defaulting to "other" if no match
        """
        # Normalize to lowercase for comparison
        normalized = tool_name.lower()

        # Check exact matches first
        if normalized in self.TOOL_KIND_MAPPING:
            return self.TOOL_KIND_MAPPING[normalized]

        # Check for partial matches (e.g., "read_file" contains "read")
        for pattern, kind in self.TOOL_KIND_MAPPING.items():
            if pattern in normalized:
                return kind

        # Default to "other" if no match
        return "other"

    async def create_tool_call(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any],
        title: Optional[str] = None,
        kind: Optional[ToolKind] = None,
    ) -> str:
        """
        Create a new tool call and send ToolCallStart notification.

        Args:
            tool_name: Name of the tool being called
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments
            title: Optional human-readable title (defaults to tool name)
            kind: Optional explicit ToolKind (auto-inferred if not provided)

        Returns:
            The generated tool call ID
        """
        tool_call_id = str(uuid.uuid4())

        # Auto-generate title if not provided
        if not title:
            title = f"{tool_name} ({server_name})"

        # Auto-infer kind if not provided
        if not kind:
            kind = self.infer_tool_kind(tool_name)

        # Track this call
        self.active_calls[tool_call_id] = {
            "status": "pending",
            "tool_name": tool_name,
            "server_name": server_name,
            "arguments": arguments,
            "kind": kind,
            "title": title,
        }

        logger.info(
            "Creating tool call",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            server_name=server_name,
            kind=kind,
        )

        # Send ToolCallStart notification
        notification = ToolCallStart(
            toolCallId=tool_call_id,
            title=title,
            status="pending",
            kind=kind,
            rawInput=arguments,
        )

        try:
            await self.connection.sessionUpdate({
                "sessionId": self.session_id,
                **notification.model_dump(exclude_none=True),
            })
        except Exception as e:
            logger.error(
                f"Failed to send tool call start notification: {e}",
                tool_call_id=tool_call_id,
                exc_info=True,
            )

        return tool_call_id

    async def update_tool_call(
        self,
        tool_call_id: str,
        status: Optional[ToolCallStatus] = None,
        content: Optional[str] = None,
        locations: Optional[list[str]] = None,
        raw_output: Optional[Any] = None,
    ) -> None:
        """
        Update a tool call and send ToolCallProgress notification.

        Args:
            tool_call_id: ID of the tool call to update
            status: New status (pending, in_progress, completed, failed)
            content: Content/output to append
            locations: File paths affected by this tool
            raw_output: Raw output from the tool
        """
        if tool_call_id not in self.active_calls:
            logger.warning(
                f"Attempted to update unknown tool call: {tool_call_id}",
                tool_call_id=tool_call_id,
            )
            return

        call_info = self.active_calls[tool_call_id]

        # Update tracked state
        if status:
            call_info["status"] = status

        logger.debug(
            "Updating tool call",
            tool_call_id=tool_call_id,
            status=status,
            has_content=bool(content),
            has_locations=bool(locations),
        )

        # Build the update notification
        update = ToolCallProgress(
            toolCallId=tool_call_id,
        )

        # Add optional fields
        if status:
            update.status = status

        if content:
            update.content = [ContentToolCallContent(type="text", text=content)]

        if locations:
            update.locations = [ToolCallLocation(path=loc) for loc in locations]

        if raw_output is not None:
            update.rawOutput = raw_output

        # Send notification
        try:
            await self.connection.sessionUpdate({
                "sessionId": self.session_id,
                **update.model_dump(exclude_none=True),
            })
        except Exception as e:
            logger.error(
                f"Failed to send tool call update notification: {e}",
                tool_call_id=tool_call_id,
                exc_info=True,
            )

    async def progress_update(
        self,
        tool_call_id: str,
        progress: float,
        total: Optional[float],
        message: Optional[str],
    ) -> None:
        """
        Send a progress update for a tool call.

        This is called from MCP progress callbacks.

        Args:
            tool_call_id: ID of the tool call
            progress: Current progress value
            total: Total expected value (for percentage calculation)
            message: Progress message
        """
        if tool_call_id not in self.active_calls:
            logger.warning(
                f"Progress update for unknown tool call: {tool_call_id}",
                tool_call_id=tool_call_id,
            )
            return

        # Format progress message
        progress_text = f"Progress: {progress}"
        if total is not None:
            percentage = (progress / total) * 100 if total > 0 else 0
            progress_text = f"Progress: {percentage:.1f}% ({progress}/{total})"

        if message:
            progress_text = f"{progress_text} - {message}"

        # Send as content update
        await self.update_tool_call(
            tool_call_id=tool_call_id,
            content=progress_text,
        )

    async def complete_tool_call(
        self,
        tool_call_id: str,
        output: Any,
        is_error: bool = False,
    ) -> None:
        """
        Mark a tool call as completed or failed.

        Args:
            tool_call_id: ID of the tool call
            output: Tool output/result
            is_error: Whether the tool execution failed
        """
        status: ToolCallStatus = "failed" if is_error else "completed"

        # Extract text content if it's a CallToolResult
        content = None
        if hasattr(output, "content") and output.content:
            # MCP CallToolResult - extract text from content blocks
            text_parts = []
            for block in output.content:
                if hasattr(block, "type") and block.type == "text":
                    text_parts.append(block.text)
            content = "\n".join(text_parts) if text_parts else None
        elif isinstance(output, str):
            content = output

        await self.update_tool_call(
            tool_call_id=tool_call_id,
            status=status,
            content=content,
            raw_output=output if not isinstance(output, str) else None,
        )

        logger.info(
            "Tool call completed",
            tool_call_id=tool_call_id,
            status=status,
        )

        # Clean up from active calls
        self.active_calls.pop(tool_call_id, None)

    def get_active_call_count(self) -> int:
        """Get the number of active tool calls."""
        return len(self.active_calls)
