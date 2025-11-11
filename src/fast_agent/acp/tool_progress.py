"""
ACP Tool Progress Tracking

Provides integration between MCP tool execution and ACP tool call notifications.
When MCP tools execute and report progress, this module:
1. Sends initial tool_call notifications to the ACP client
2. Updates with progress via tool_call_update notifications
3. Handles status transitions (pending -> in_progress -> completed/failed)
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from acp.schema import (
    SessionNotification,
    ToolCallContent,
    ToolCallDiff,
    ToolCallStatus,
    ToolCallUpdate,
)

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)


class ToolKind(str, Enum):
    """ACP tool kinds for categorizing tool calls."""

    READ = "read"
    EDIT = "edit"
    DELETE = "delete"
    MOVE = "move"
    SEARCH = "search"
    EXECUTE = "execute"
    THINK = "think"
    FETCH = "fetch"
    OTHER = "other"


@dataclass
class ToolCallTracker:
    """Tracks the state of a tool call for ACP notifications."""

    tool_call_id: str
    session_id: str
    tool_name: str
    server_name: str
    status: ToolCallStatus = "pending"
    title: str = ""
    kind: ToolKind = ToolKind.OTHER
    arguments: dict[str, Any] | None = None
    content: list[Any] = field(default_factory=list)
    locations: list[dict[str, Any]] = field(default_factory=list)


class ACPToolProgressManager:
    """
    Manages tool call progress notifications for ACP clients.

    This class tracks active tool calls and sends appropriate notifications
    to the ACP client as tools execute and report progress.
    """

    def __init__(self, connection: "AgentSideConnection") -> None:
        """
        Initialize the progress manager.

        Args:
            connection: The ACP connection to send notifications on
        """
        self._connection = connection
        self._active_tools: dict[str, ToolCallTracker] = {}
        self._lock = asyncio.Lock()

    def _infer_tool_kind(self, tool_name: str, arguments: dict[str, Any] | None) -> ToolKind:
        """
        Infer the tool kind from the tool name and arguments.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments

        Returns:
            The inferred ToolKind
        """
        name_lower = tool_name.lower()

        # Common patterns for tool categorization
        if any(word in name_lower for word in ["read", "get", "fetch", "list", "show"]):
            return ToolKind.READ
        elif any(word in name_lower for word in ["write", "edit", "update", "modify", "patch"]):
            return ToolKind.EDIT
        elif any(
            word in name_lower for word in ["delete", "remove", "clear", "clean", "rm"]
        ):
            return ToolKind.DELETE
        elif any(word in name_lower for word in ["move", "rename", "mv"]):
            return ToolKind.MOVE
        elif any(word in name_lower for word in ["search", "find", "query", "grep"]):
            return ToolKind.SEARCH
        elif any(
            word in name_lower
            for word in ["execute", "run", "exec", "command", "bash", "shell"]
        ):
            return ToolKind.EXECUTE
        elif any(word in name_lower for word in ["think", "plan", "reason"]):
            return ToolKind.THINK
        elif any(word in name_lower for word in ["fetch", "download", "http", "request"]):
            return ToolKind.FETCH

        return ToolKind.OTHER

    async def start_tool_call(
        self,
        session_id: str,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        """
        Notify the client that a tool call has started.

        Args:
            session_id: The ACP session ID
            tool_name: Name of the tool being called
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments

        Returns:
            The tool call ID for tracking
        """
        tool_call_id = str(uuid.uuid4())

        # Infer tool kind
        kind = self._infer_tool_kind(tool_name, arguments)

        # Create title
        title = f"{server_name}/{tool_name}"
        if arguments:
            # Include key argument info in title
            arg_str = ", ".join(f"{k}={v}" for k, v in list(arguments.items())[:2])
            if len(arg_str) > 50:
                arg_str = arg_str[:47] + "..."
            title = f"{title}({arg_str})"

        # Create tracker
        tracker = ToolCallTracker(
            tool_call_id=tool_call_id,
            session_id=session_id,
            tool_name=tool_name,
            server_name=server_name,
            status="pending",
            title=title,
            kind=kind,
            arguments=arguments,
        )

        async with self._lock:
            self._active_tools[tool_call_id] = tracker

        # Send initial notification
        try:
            notification = SessionNotification(
                sessionId=session_id,
                update=ToolCallUpdate(
                    sessionUpdate="tool_call",
                    toolCallId=tool_call_id,
                    title=title,
                    kind=kind.value,
                    status="pending",
                ),
            )
            await self._connection.sessionUpdate(notification)

            logger.debug(
                f"Started tool call tracking: {tool_call_id}",
                name="acp_tool_call_start",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                server_name=server_name,
            )
        except Exception as e:
            logger.error(
                f"Error sending tool_call notification: {e}",
                name="acp_tool_call_error",
                exc_info=True,
            )

        return tool_call_id

    async def update_tool_progress(
        self,
        tool_call_id: str,
        progress: float | None = None,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        """
        Update the progress of a tool call.

        Args:
            tool_call_id: The tool call ID
            progress: Current progress value
            total: Total value for progress calculation
            message: Optional progress message
        """
        async with self._lock:
            tracker = self._active_tools.get(tool_call_id)
            if not tracker:
                logger.warning(
                    f"Tool call {tool_call_id} not found for progress update",
                    name="acp_tool_progress_not_found",
                )
                return

            # Update status to in_progress if still pending
            if tracker.status == "pending":
                tracker.status = "in_progress"

        # Build content for progress update
        content = None
        if message:
            content = ToolCallContent(
                type="text",
                text=message,
            )

        # Send progress update
        try:
            update_data: dict[str, Any] = {
                "sessionUpdate": "tool_call_update",
                "toolCallId": tool_call_id,
                "status": tracker.status,
            }

            if content:
                update_data["content"] = content

            notification = SessionNotification(
                sessionId=tracker.session_id,
                update=ToolCallUpdate(**update_data),
            )
            await self._connection.sessionUpdate(notification)

            logger.debug(
                f"Updated tool call progress: {tool_call_id}",
                name="acp_tool_progress_update",
                progress=progress,
                total=total,
                message=message,
            )
        except Exception as e:
            logger.error(
                f"Error sending tool_call_update notification: {e}",
                name="acp_tool_progress_error",
                exc_info=True,
            )

    async def complete_tool_call(
        self,
        tool_call_id: str,
        success: bool = True,
        result_text: str | None = None,
        error: str | None = None,
    ) -> None:
        """
        Mark a tool call as completed or failed.

        Args:
            tool_call_id: The tool call ID
            success: Whether the tool call succeeded
            result_text: Optional result text to include
            error: Optional error message if failed
        """
        async with self._lock:
            tracker = self._active_tools.get(tool_call_id)
            if not tracker:
                logger.warning(
                    f"Tool call {tool_call_id} not found for completion",
                    name="acp_tool_complete_not_found",
                )
                return

            # Update status
            tracker.status = "completed" if success else "failed"

        # Build content
        content = None
        if result_text or error:
            content = ToolCallContent(
                type="text",
                text=error if error else result_text or "",
            )

        # Send completion notification
        try:
            update_data: dict[str, Any] = {
                "sessionUpdate": "tool_call_update",
                "toolCallId": tool_call_id,
                "status": tracker.status,
            }

            if content:
                update_data["content"] = content

            notification = SessionNotification(
                sessionId=tracker.session_id,
                update=ToolCallUpdate(**update_data),
            )
            await self._connection.sessionUpdate(notification)

            logger.info(
                f"Completed tool call: {tool_call_id}",
                name="acp_tool_call_complete",
                status=tracker.status,
            )
        except Exception as e:
            logger.error(
                f"Error sending tool_call completion notification: {e}",
                name="acp_tool_complete_error",
                exc_info=True,
            )
        finally:
            # Clean up tracker
            async with self._lock:
                self._active_tools.pop(tool_call_id, None)

    async def cleanup_session_tools(self, session_id: str) -> None:
        """
        Clean up all tool trackers for a session.

        Args:
            session_id: The session ID to clean up
        """
        async with self._lock:
            to_remove = [
                tool_id
                for tool_id, tracker in self._active_tools.items()
                if tracker.session_id == session_id
            ]
            for tool_id in to_remove:
                self._active_tools.pop(tool_id, None)

        logger.debug(
            f"Cleaned up {len(to_remove)} tool trackers for session {session_id}",
            name="acp_tool_cleanup",
        )
