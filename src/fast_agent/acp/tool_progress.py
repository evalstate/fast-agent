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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from acp.helpers import session_notification
from acp.schema import (
    ContentToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallStatus,
    ToolKind,
)

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)


@dataclass
class ToolCallTracker:
    """Tracks the state of a tool call for ACP notifications."""

    tool_call_id: str
    session_id: str
    tool_name: str
    server_name: str
    status: ToolCallStatus = "pending"
    title: str = ""
    kind: ToolKind = "other"
    arguments: dict[str, Any] | None = None


class ACPToolProgressManager:
    """
    Manages tool call progress notifications for ACP clients.

    Implements the ToolExecutionHandler protocol to provide lifecycle hooks
    for tool execution. Sends sessionUpdate notifications to ACP clients as
    tools execute and report progress.
    """

    def __init__(self, connection: "AgentSideConnection", session_id: str) -> None:
        """
        Initialize the progress manager.

        Args:
            connection: The ACP connection to send notifications on
            session_id: The ACP session ID for this manager
        """
        self._connection = connection
        self._session_id = session_id
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
            return "read"
        elif any(word in name_lower for word in ["write", "edit", "update", "modify", "patch"]):
            return "edit"
        elif any(word in name_lower for word in ["delete", "remove", "clear", "clean", "rm"]):
            return "delete"
        elif any(word in name_lower for word in ["move", "rename", "mv"]):
            return "move"
        elif any(word in name_lower for word in ["search", "find", "query", "grep"]):
            return "search"
        elif any(
            word in name_lower for word in ["execute", "run", "exec", "command", "bash", "shell"]
        ):
            return "execute"
        elif any(word in name_lower for word in ["think", "plan", "reason"]):
            return "think"
        elif any(word in name_lower for word in ["fetch", "download", "http", "request"]):
            return "fetch"

        return "other"

    def _build_text_content(self, message: str | None) -> list[ContentToolCallContent] | None:
        """
        Convert a text string into ACP-compatible tool call content blocks.
        """
        if not message:
            return None
        return [
            ContentToolCallContent(
                type="content",
                content=TextContentBlock(type="text", text=message),
            )
        ]

    async def on_tool_start(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        """
        Called when a tool execution starts.

        Implements ToolExecutionHandler.on_tool_start protocol method.

        Args:
            tool_name: Name of the tool being called
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments

        Returns:
            The tool call ID for tracking
        """
        session_id = self._session_id
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
            tool_call_start = ToolCallStart(
                sessionUpdate="tool_call",
                toolCallId=tool_call_id,
                title=title,
                kind=kind,
                status="pending",
                rawInput=arguments,
            )

            notification = session_notification(session_id, tool_call_start)
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

    async def on_tool_progress(
        self,
        tool_call_id: str,
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        """
        Called when tool execution reports progress.

        Implements ToolExecutionHandler.on_tool_progress protocol method.

        Args:
            tool_call_id: The tool call ID
            progress: Current progress value
            total: Total value for progress calculation (optional)
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
        content_blocks = self._build_text_content(message)

        # Send progress update
        try:
            update_data = ToolCallProgress(
                sessionUpdate="tool_call_update",
                toolCallId=tool_call_id,
                status=tracker.status,
                content=content_blocks,
            )

            notification = session_notification(tracker.session_id, update_data)
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

    async def on_tool_complete(
        self,
        tool_call_id: str,
        success: bool,
        result_text: str | None = None,
        error: str | None = None,
    ) -> None:
        """
        Called when tool execution completes.

        Implements ToolExecutionHandler.on_tool_complete protocol method.

        Args:
            tool_call_id: The tool call ID
            success: Whether the tool execution succeeded
            result_text: Optional result text if successful
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
        content_blocks = self._build_text_content(error if error else result_text)

        # Send completion notification
        try:
            update_data = ToolCallProgress(
                sessionUpdate="tool_call_update",
                toolCallId=tool_call_id,
                status=tracker.status,
                content=content_blocks,
                rawOutput=result_text if success else error,
            )

            notification = session_notification(tracker.session_id, update_data)
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
