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
from acp.helpers import (
    audio_block,
    embedded_blob_resource,
    embedded_text_resource,
    image_block,
    resource_link_block,
    session_notification,
    text_block,
    tool_content,
)
from acp.schema import EmbeddedResourceContentBlock, ToolKind
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

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
        # Map LLM tool_use_id → ACP tool_call_id for early notifications
        self._tool_use_id_to_call_id: dict[str, str] = {}
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

    def _convert_mcp_content_to_acp(self, content: list[ContentBlock] | None):
        """
        Convert MCP content blocks to ACP tool call content using SDK helpers.

        Args:
            content: List of MCP content blocks (TextContent, ImageContent, etc.)

        Returns:
            List of ContentToolCallContent blocks, or None if no content
        """
        if not content:
            return None

        acp_content = []

        for block in content:
            try:
                if isinstance(block, TextContent):
                    # MCP TextContent -> ACP TextContentBlock using SDK helper
                    acp_content.append(tool_content(text_block(block.text)))

                elif isinstance(block, ImageContent):
                    # MCP ImageContent -> ACP ImageContentBlock using SDK helper
                    acp_content.append(tool_content(image_block(block.data, block.mimeType)))

                elif isinstance(block, AudioContent):
                    # MCP AudioContent -> ACP AudioContentBlock using SDK helper
                    acp_content.append(tool_content(audio_block(block.data, block.mimeType)))

                elif isinstance(block, ResourceLink):
                    # MCP ResourceLink -> ACP ResourceContentBlock using SDK helper
                    # Note: ResourceLink has uri, mimeType but resource_link_block wants name
                    # Use the URI as the name for now
                    acp_content.append(
                        tool_content(
                            resource_link_block(
                                name=str(block.uri),
                                uri=str(block.uri),
                                mime_type=block.mimeType if hasattr(block, "mimeType") else None,
                            )
                        )
                    )

                elif isinstance(block, EmbeddedResource):
                    # MCP EmbeddedResource -> ACP EmbeddedResourceContentBlock
                    resource = block.resource
                    if isinstance(resource, TextResourceContents):
                        embedded_res = embedded_text_resource(
                            uri=str(resource.uri),
                            text=resource.text,
                            mime_type=resource.mimeType,
                        )
                        acp_content.append(
                            tool_content(
                                EmbeddedResourceContentBlock(
                                    type="embedded_resource", resource=embedded_res
                                )
                            )
                        )
                    elif isinstance(resource, BlobResourceContents):
                        embedded_res = embedded_blob_resource(
                            uri=str(resource.uri),
                            blob=resource.blob,
                            mime_type=resource.mimeType,
                        )
                        acp_content.append(
                            tool_content(
                                EmbeddedResourceContentBlock(
                                    type="embedded_resource", resource=embedded_res
                                )
                            )
                        )
                else:
                    # Unknown content type - log warning and skip
                    logger.warning(
                        f"Unknown content type: {type(block).__name__}",
                        name="acp_unknown_content_type",
                    )
            except Exception as e:
                logger.error(
                    f"Error converting content block {type(block).__name__}: {e}",
                    name="acp_content_conversion_error",
                    exc_info=True,
                )

        return acp_content if acp_content else None

    async def on_tool_start(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> str:
        """
        Called when a tool execution starts.

        Implements ToolExecutionHandler.on_tool_start protocol method.

        Args:
            tool_name: Name of the tool being called
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments
            tool_use_id: Optional LLM tool use ID for correlation with early notifications

        Returns:
            The tool call ID for tracking
        """
        # Check if we already sent an early notification for this tool
        async with self._lock:
            existing_tool_call_id = self._tool_use_id_to_call_id.get(tool_use_id) if tool_use_id else None

        if existing_tool_call_id:
            # We already sent a "pending" notification via on_tool_declared
            # Update it to "in_progress" now that execution is actually starting
            async with self._lock:
                external_id = self._tool_call_id_to_external_id.get(existing_tool_call_id)
                if external_id:
                    try:
                        update_data = self._tracker.progress(
                            external_id=external_id,
                            status="in_progress",
                        )
                        notification = session_notification(self._session_id, update_data)
                        await self._connection.sessionUpdate(notification)

                        logger.debug(
                            f"Updated tool call to in_progress: {existing_tool_call_id}",
                            name="acp_tool_call_in_progress",
                            tool_call_id=existing_tool_call_id,
                            external_id=external_id,
                            tool_use_id=tool_use_id,
                        )
                    except Exception as e:
                        logger.error(
                            f"Error updating tool_call to in_progress: {e}",
                            name="acp_tool_progress_error",
                            exc_info=True,
                        )

            return existing_tool_call_id

        # No early notification - create a new one (for tools called outside LLM flow)
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
                status="in_progress",  # Start as in_progress since execution is beginning
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

    async def on_tool_declared(
        self,
        tool_use_id: str,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        """
        Called when the LLM declares it will use a tool (before actual execution).

        This sends an early notification to ACP clients with "pending" status,
        improving responsiveness by notifying clients as soon as the LLM declares
        the tool use (after streaming), rather than waiting for actual execution.

        Args:
            tool_use_id: The LLM's tool use ID for correlation
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
            # Store mapping from LLM tool_use_id to ACP tool_call_id
            self._tool_use_id_to_call_id[tool_use_id] = tool_call_start.toolCallId

        # Send initial notification
        try:
            notification = session_notification(self._session_id, tool_call_start)
            await self._connection.sessionUpdate(notification)

            logger.debug(
                f"Declared tool call (early notification): {tool_call_start.toolCallId}",
                name="acp_tool_call_declared",
                tool_call_id=tool_call_start.toolCallId,
                external_id=external_id,
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                server_name=server_name,
            )
        except Exception as e:
            logger.error(
                f"Error sending tool_call declared notification: {e}",
                name="acp_tool_declared_error",
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
                    name="acp_progress_creation_error",
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
        content: list[ContentBlock] | None = None,
        error: str | None = None,
    ) -> None:
        """
        Called when tool execution completes.

        Implements ToolExecutionHandler.on_tool_complete protocol method.

        Args:
            tool_call_id: The tool call ID
            success: Whether the tool execution succeeded
            content: Optional content blocks (text, images, etc.) if successful
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

        # Build content blocks
        if error:
            # Error case: convert error string to text content using SDK helper
            content_blocks = [tool_content(text_block(error))]
            raw_output = error
        elif content:
            # Success case with structured content: convert MCP content to ACP using SDK helpers
            content_blocks = self._convert_mcp_content_to_acp(content)
            # For rawOutput, extract just text content for backward compatibility
            text_parts = [c.text for c in content if isinstance(c, TextContent)]
            raw_output = "\n".join(text_parts) if text_parts else None
        else:
            # No content or error
            content_blocks = None
            raw_output = None

        # Determine status
        status = "completed" if success else "failed"

        # Use SDK tracker to create completion update
        try:
            async with self._lock:
                update_data = self._tracker.progress(
                    external_id=external_id,
                    status=status,
                    content=content_blocks,
                    raw_output=raw_output,
                )
        except Exception as e:
            logger.error(
                f"Error creating completion update: {e}",
                name="acp_completion_creation_error",
                exc_info=True,
            )
            return

        # Send completion notification
        try:
            notification = session_notification(self._session_id, update_data)
            await self._connection.sessionUpdate(notification)

            logger.info(
                f"Completed tool call: {tool_call_id}",
                name="acp_tool_call_complete",
                status=status,
                content_blocks=len(content_blocks) if content_blocks else 0,
            )
        except Exception as e:
            logger.error(
                f"Error sending tool_call completion notification: {e}",
                name="acp_tool_complete_error",
                exc_info=True,
            )
        finally:
            # Clean up tracker using SDK's forget method
            async with self._lock:
                self._tracker.forget(external_id)
                self._tool_call_id_to_external_id.pop(tool_call_id, None)
                # Also clean up tool_use_id mapping if present
                tool_use_id_to_remove = None
                for tuid, tcid in self._tool_use_id_to_call_id.items():
                    if tcid == tool_call_id:
                        tool_use_id_to_remove = tuid
                        break
                if tool_use_id_to_remove:
                    self._tool_use_id_to_call_id.pop(tool_use_id_to_remove, None)

    async def cleanup_session_tools(self, session_id: str) -> None:
        """
        Clean up all tool trackers for a session.

        Args:
            session_id: The session ID to clean up
        """
        # The SDK tracker doesn't maintain session associations,
        # so we just clear our mapping
        async with self._lock:
            count = len(self._tool_call_id_to_external_id)
            # Forget all tracked tools
            for external_id in list(self._tracker._tool_calls.keys()):
                self._tracker.forget(external_id)
            self._tool_call_id_to_external_id.clear()
            self._tool_use_id_to_call_id.clear()

        logger.debug(
            f"Cleaned up {count} tool trackers for session {session_id}",
            name="acp_tool_cleanup",
        )
