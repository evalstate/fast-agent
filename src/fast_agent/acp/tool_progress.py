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
from typing import TYPE_CHECKING, Any

from acp.contrib import ToolCallTracker
from acp.helpers import session_notification, text_block, tool_content
from acp.schema import ToolKind

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)


class ACPToolProgressManager:
    """
    Manages tool call progress notifications for ACP clients.

    Implements the ToolExecutionHandler protocol to provide lifecycle hooks
    for tool execution. Sends sessionUpdate notifications to ACP clients as
    tools execute and report progress.

    Uses the SDK's ToolCallTracker for state management and notification generation.
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
        # Use SDK's ToolCallTracker for state management
        self._tracker = ToolCallTracker()
        # Map ACP tool_call_id → external_id for reverse lookups
        self._tool_call_id_to_external_id: dict[str, str] = {}
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
        # Generate external ID for SDK tracker
        external_id = str(uuid.uuid4())

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

        # Use SDK tracker to create the tool call start notification
        async with self._lock:
            tool_call_start = self._tracker.start(
                external_id=external_id,
                title=title,
                kind=kind,
                status="pending",
                raw_input=arguments,
            )
            # Store mapping from ACP tool_call_id to external_id for later lookups
            self._tool_call_id_to_external_id[tool_call_start.toolCallId] = external_id

        # Send initial notification
        try:
            notification = session_notification(self._session_id, tool_call_start)
            await self._connection.sessionUpdate(notification)

            logger.debug(
                f"Started tool call tracking: {tool_call_start.toolCallId}",
                name="acp_tool_call_start",
                tool_call_id=tool_call_start.toolCallId,
                external_id=external_id,
                tool_name=tool_name,
                server_name=server_name,
            )
        except Exception as e:
            logger.error(
                f"Error sending tool_call notification: {e}",
                name="acp_tool_call_error",
                exc_info=True,
            )

        # Return the ACP tool_call_id for caller to track
        return tool_call_start.toolCallId

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
        # Look up external_id from tool_call_id
        async with self._lock:
            external_id = self._tool_call_id_to_external_id.get(tool_call_id)
            if not external_id:
                logger.warning(
                    f"Tool call {tool_call_id} not found for progress update",
                    name="acp_tool_progress_not_found",
                )
                return

            # Build content for progress update using SDK helpers
            content = None
            if message:
                content = [tool_content(text_block(message))]

            # Use SDK tracker to create progress update
            try:
                update_data = self._tracker.progress(
                    external_id=external_id,
                    status="in_progress",
                    content=content,
                )
            except Exception as e:
                logger.error(
                    f"Error creating progress update: {e}",
                    name="acp_tool_progress_create_error",
                    exc_info=True,
                )
                return

        # Send progress update
        try:
            notification = session_notification(self._session_id, update_data)
            await self._connection.sessionUpdate(notification)

            logger.debug(
                f"Updated tool call progress: {tool_call_id}",
                name="acp_tool_progress_update",
                tool_call_id=tool_call_id,
                external_id=external_id,
                progress=progress,
                total=total,
                progress_message=message,
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
        # Look up external_id from tool_call_id
        async with self._lock:
            external_id = self._tool_call_id_to_external_id.get(tool_call_id)
            if not external_id:
                logger.warning(
                    f"Tool call {tool_call_id} not found for completion",
                    name="acp_tool_complete_not_found",
                )
                return

            # Build content using SDK helpers
            content = None
            message = error if error else result_text
            if message:
                content = [tool_content(text_block(message))]

            # Use SDK tracker to create completion update
            status = "completed" if success else "failed"
            try:
                update_data = self._tracker.progress(
                    external_id=external_id,
                    status=status,
                    content=content,
                    raw_output=result_text if success else error,
                )
            except Exception as e:
                logger.error(
                    f"Error creating completion update: {e}",
                    name="acp_tool_complete_create_error",
                    exc_info=True,
                )
                # Clean up even on error
                self._tracker.forget(external_id)
                self._tool_call_id_to_external_id.pop(tool_call_id, None)
                return

        # Send completion notification
        try:
            notification = session_notification(self._session_id, update_data)
            await self._connection.sessionUpdate(notification)

            logger.info(
                f"Completed tool call: {tool_call_id}",
                name="acp_tool_call_complete",
                tool_call_id=tool_call_id,
                external_id=external_id,
                status=status,
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
                self._tracker.forget(external_id)
                self._tool_call_id_to_external_id.pop(tool_call_id, None)

    async def cleanup_session_tools(self, session_id: str) -> None:
        """
        Clean up all tool trackers for a session.

        Args:
            session_id: The session ID to clean up
        """
        # Since this manager is already scoped to a single session (self._session_id),
        # we just need to clear all tracked tool calls
        async with self._lock:
            count = len(self._tool_call_id_to_external_id)
            # Clear all mappings - SDK tracker cleanup is handled by forget()
            self._tool_call_id_to_external_id.clear()

        logger.debug(
            f"Cleaned up {count} tool trackers for session {session_id}",
            name="acp_tool_cleanup",
        )
