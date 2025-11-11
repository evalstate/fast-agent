"""
Tool Call Tracker for Agent Client Protocol (ACP).

This module implements tracking and notification of tool calls according to the
ACP specification at https://agentclientprotocol.com/protocol/tool-calls.md
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from acp.helpers import session_notification
from acp.schema import (
    AgentMessageChunk,
    Location,
    SessionUpdate,
    ToolCall,
    ToolCallProgress,
    ToolCallStatus,
)

from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ToolCallTracker:
    """
    Tracks tool calls and sends notifications to ACP clients.

    Manages the lifecycle of tool calls:
    - pending → in_progress → completed/failed

    Sends session/update notifications with tool call information.
    """

    session_id: str
    connection: Any  # AgentSideConnection
    _active_tools: dict[str, ToolCall] = field(default_factory=dict)

    async def create_tool_call(
        self,
        title: str,
        kind: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        locations: list[Location] | None = None,
    ) -> str:
        """
        Create a new tool call and notify the client.

        Args:
            title: Human-readable description (e.g., "Reading configuration file")
            kind: Category (read, edit, delete, move, search, execute, think, fetch, other)
            tool_name: Name of the tool being called
            arguments: Optional tool arguments
            locations: Optional file paths/line numbers affected

        Returns:
            The unique tool call ID
        """
        tool_call_id = str(uuid.uuid4())

        tool_call = ToolCall(
            toolCallId=tool_call_id,
            title=title,
            kind=kind,
            status="pending",
            rawInput={"tool": tool_name, "arguments": arguments} if arguments else None,
            locations=locations,
        )

        self._active_tools[tool_call_id] = tool_call

        # Send initial notification
        await self._send_tool_call_notification(tool_call)

        logger.debug(
            f"Created tool call: {tool_call_id} - {title}",
            name="acp_tool_call_created",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )

        return tool_call_id

    async def update_tool_call(
        self,
        tool_call_id: str,
        status: Optional[ToolCallStatus] = None,
        progress: Optional[ToolCallProgress] = None,
        content: Optional[list] = None,
        raw_output: Optional[Any] = None,
    ) -> None:
        """
        Update an existing tool call and notify the client.

        Args:
            tool_call_id: The tool call ID to update
            status: New status (in_progress, completed, failed)
            progress: Optional progress information
            content: Optional result content
            raw_output: Optional raw output from the tool
        """
        if tool_call_id not in self._active_tools:
            logger.warning(
                f"Attempted to update unknown tool call: {tool_call_id}",
                name="acp_tool_call_unknown",
            )
            return

        tool_call = self._active_tools[tool_call_id]

        # Update fields (only what changed)
        updates = {}
        if status is not None:
            updates["status"] = status
        if progress is not None:
            updates["progress"] = progress
        if content is not None:
            updates["content"] = content
        if raw_output is not None:
            updates["rawOutput"] = raw_output

        # Create updated tool call
        updated_tool_call = tool_call.model_copy(update=updates)
        self._active_tools[tool_call_id] = updated_tool_call

        # Send update notification
        await self._send_tool_call_notification(updated_tool_call)

        logger.debug(
            f"Updated tool call: {tool_call_id} - status={status}",
            name="acp_tool_call_updated",
            tool_call_id=tool_call_id,
            status=status,
        )

    async def complete_tool_call(
        self,
        tool_call_id: str,
        content: list | None = None,
        raw_output: Any | None = None,
        is_error: bool = False,
    ) -> None:
        """
        Mark a tool call as completed or failed.

        Args:
            tool_call_id: The tool call ID
            content: Optional result content
            raw_output: Optional raw output from the tool
            is_error: Whether the tool call failed
        """
        status: ToolCallStatus = "failed" if is_error else "completed"
        await self.update_tool_call(
            tool_call_id=tool_call_id,
            status=status,
            content=content,
            raw_output=raw_output,
        )

        # Remove from active tools
        if tool_call_id in self._active_tools:
            del self._active_tools[tool_call_id]

    async def _send_tool_call_notification(self, tool_call: ToolCall) -> None:
        """Send a tool_call_update notification to the client."""
        try:
            # Create session update with tool call
            update = SessionUpdate(
                sessionUpdate="tool_call_update",
                toolCall=tool_call,
            )

            notification = session_notification(self.session_id, update)
            await self.connection.sessionUpdate(notification)

        except Exception as e:
            logger.error(
                f"Error sending tool call notification: {e}",
                name="acp_tool_call_notification_error",
                exc_info=True,
            )
