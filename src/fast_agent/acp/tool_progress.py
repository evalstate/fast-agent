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
    resource_block,
    resource_link_block,
    text_block,
    tool_content,
)
from acp.schema import ToolKind
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from mcp.types import (
    ContentBlock as MCPContentBlock,
)

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.common import get_resource_name, get_server_name, is_namespaced_name

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
        # Map tool_call_id → simple title (server/tool) for progress updates
        self._simple_titles: dict[str, str] = {}
        # Map tool_call_id → full title (with args) for completion
        self._full_titles: dict[str, str] = {}
        # Track tool_use_id from stream events to avoid duplicate notifications
        self._stream_tool_use_ids: dict[str, str] = {}  # tool_use_id → external_id
        # Track pending stream notification tasks
        self._stream_tasks: dict[str, asyncio.Task] = {}  # tool_use_id → task
        # Track stream chunk counts for title updates
        self._stream_chunk_counts: dict[str, int] = {}  # tool_use_id → chunk count
        # Track base titles for streaming tools (before chunk count suffix)
        self._stream_base_titles: dict[str, str] = {}  # tool_use_id → base title
        self._lock = asyncio.Lock()

    async def get_tool_call_id_for_tool_use(self, tool_use_id: str) -> str | None:
        """
        Get the ACP toolCallId for a given LLM tool_use_id.

        This is used by the permission handler to ensure the permission request
        references the same toolCallId as any existing streaming notification.

        Args:
            tool_use_id: The LLM's tool use ID

        Returns:
            The ACP toolCallId if a streaming notification was already sent, None otherwise
        """
        # Check if there's a pending stream notification task for this tool_use_id
        # If so, wait for it to complete so the toolCallId is available
        task = self._stream_tasks.get(tool_use_id)
        if task and not task.done():
            try:
                await task
            except Exception:
                pass  # Ignore errors, just ensure task completed

        # Now look up the toolCallId
        external_id = self._stream_tool_use_ids.get(tool_use_id)
        if external_id:
            # Look up the toolCallId from the tracker
            async with self._lock:
                try:
                    model = self._tracker.tool_call_model(external_id)
                    if model and hasattr(model, "toolCallId"):
                        return model.toolCallId
                except Exception:
                    # Swallow and fall back to local mapping
                    pass
                # Fallback: check our own mapping
                for tool_call_id, ext_id in self._tool_call_id_to_external_id.items():
                    if ext_id == external_id:
                        return tool_call_id
        return None

    def handle_tool_stream_event(self, event_type: str, info: dict[str, Any] | None = None) -> None:
        """
        Handle tool stream events from the LLM during streaming.

        This gets called when the LLM streams tool use blocks, BEFORE tool execution.
        Sends early ACP notifications so clients see tool calls immediately.

        Args:
            event_type: Type of stream event ("start", "delta", "text", "stop")
            info: Event payload containing tool_name, tool_use_id, etc.
        """
        if event_type == "start" and info:
            tool_name = info.get("tool_name")
            tool_use_id = info.get("tool_use_id")

            if tool_name and tool_use_id:
                # Generate external_id SYNCHRONOUSLY to avoid race with delta events
                external_id = str(uuid.uuid4())
                self._stream_tool_use_ids[tool_use_id] = external_id

                # Schedule async notification sending and store the task
                task = asyncio.create_task(
                    self._send_stream_start_notification(tool_name, tool_use_id, external_id)
                )
                # Store task reference so we can await it in on_tool_start if needed
                self._stream_tasks[tool_use_id] = task

        elif event_type == "delta" and info:
            tool_use_id = info.get("tool_use_id")
            chunk = info.get("chunk")

            if tool_use_id and chunk:
                # Schedule async notification with accumulated arguments
                asyncio.create_task(self._send_stream_delta_notification(tool_use_id, chunk))

    async def _send_stream_start_notification(
        self, tool_name: str, tool_use_id: str, external_id: str
    ) -> None:
        """
        Send early ACP notification when tool stream starts.

        Args:
            tool_name: Name of the tool being called (may be namespaced like "server__tool")
            tool_use_id: LLM's tool use ID
            external_id: Pre-generated external ID for SDK tracker
        """
        try:
            # Parse the tool name if it's namespaced (e.g., "acp_filesystem__write_text_file")
            if is_namespaced_name(tool_name):
                server_name = get_server_name(tool_name)
                base_tool_name = get_resource_name(tool_name)
            else:
                server_name = None
                base_tool_name = tool_name

            # Infer tool kind (without arguments yet)
            kind = self._infer_tool_kind(base_tool_name, None)

            # Create title with server name if available
            if server_name:
                title = f"{server_name}/{base_tool_name}"
            else:
                title = base_tool_name

            # Use SDK tracker to create the tool call start notification
            async with self._lock:
                tool_call_start = self._tracker.start(
                    external_id=external_id,
                    title=title,
                    kind=kind,
                    status="pending",
                    raw_input=None,  # Don't have args yet
                )
                # Store mapping from ACP tool_call_id to external_id
                self._tool_call_id_to_external_id[tool_call_start.toolCallId] = external_id
                # Initialize streaming state for this tool
                self._stream_base_titles[tool_use_id] = title
                self._stream_chunk_counts[tool_use_id] = 0

            # Send initial notification
            await self._connection.session_update(
                session_id=self._session_id, update=tool_call_start
            )

            logger.debug(
                f"Sent early stream tool call notification: {tool_call_start.toolCallId}",
                name="acp_tool_stream_start",
                tool_call_id=tool_call_start.toolCallId,
                external_id=external_id,
                base_tool_name=base_tool_name,
                server_name=server_name,
                tool_use_id=tool_use_id,
            )
        except Exception as e:
            logger.error(
                f"Error sending stream tool_call notification: {e}",
                name="acp_tool_stream_error",
                exc_info=True,
            )
        finally:
            # Clean up task reference
            if tool_use_id in self._stream_tasks:
                del self._stream_tasks[tool_use_id]

    async def _send_stream_delta_notification(self, tool_use_id: str, chunk: str) -> None:
        """
        Send ACP notification with tool argument chunk as it streams.

        Accumulates chunks into content and updates title with chunk count.

        Args:
            tool_use_id: LLM's tool use ID
            chunk: JSON fragment chunk
        """
        try:
            async with self._lock:
                external_id = self._stream_tool_use_ids.get(tool_use_id)
                if not external_id:
                    # No start notification sent yet, skip this chunk
                    return

                # Increment chunk count and build title with count
                self._stream_chunk_counts[tool_use_id] = (
                    self._stream_chunk_counts.get(tool_use_id, 0) + 1
                )
                chunk_count = self._stream_chunk_counts[tool_use_id]
                base_title = self._stream_base_titles.get(tool_use_id, "Tool")
                title_with_count = f"{base_title} (streaming: {chunk_count} chunks)"

                # Use SDK's append_stream_text to accumulate chunks into content
                update = self._tracker.append_stream_text(
                    external_id=external_id,
                    text=chunk,
                    title=title_with_count,
                )

            # Only send notifications after 25 chunks to avoid UI noise for small calls
            if chunk_count < 25:
                return

            # Send notification outside the lock
            await self._connection.session_update(session_id=self._session_id, update=update)

        except Exception as e:
            logger.debug(
                f"Error sending stream delta notification: {e}",
                name="acp_tool_stream_delta_error",
                tool_use_id=tool_use_id,
            )

    # Tool kind patterns: mapping from ToolKind to keyword patterns
    _TOOL_KIND_PATTERNS: dict[ToolKind, tuple[str, ...]] = {
        "read": ("read", "get", "fetch", "list", "show"),
        "edit": ("write", "edit", "update", "modify", "patch"),
        "delete": ("delete", "remove", "clear", "clean", "rm"),
        "move": ("move", "rename", "mv"),
        "search": ("search", "find", "query", "grep"),
        "execute": ("execute", "run", "exec", "command", "bash", "shell"),
        "think": ("think", "plan", "reason"),
        "fetch": ("fetch", "download", "http", "request"),
    }

    def _infer_tool_kind(self, tool_name: str, arguments: dict[str, Any] | None) -> ToolKind:
        """
        Infer the tool kind from the tool name and arguments.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments (reserved for future use)

        Returns:
            The inferred ToolKind
        """
        name_lower = tool_name.lower()

        for kind, patterns in self._TOOL_KIND_PATTERNS.items():
            if any(pattern in name_lower for pattern in patterns):
                return kind

        return "other"

    def _convert_mcp_content_to_acp(self, content: list[MCPContentBlock] | None) -> list | None:
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
                match block:
                    case TextContent():
                        acp_content.append(tool_content(text_block(block.text)))

                    case ImageContent():
                        acp_content.append(tool_content(image_block(block.data, block.mimeType)))

                    case AudioContent():
                        acp_content.append(tool_content(audio_block(block.data, block.mimeType)))

                    case ResourceLink():
                        # Use URI as the name for resource links
                        acp_content.append(
                            tool_content(
                                resource_link_block(
                                    name=str(block.uri),
                                    uri=str(block.uri),
                                    mime_type=getattr(block, "mimeType", None),
                                )
                            )
                        )

                    case EmbeddedResource():
                        # Use SDK's resource_block helper with embedded resource contents
                        match block.resource:
                            case TextResourceContents():
                                embedded_res = embedded_text_resource(
                                    uri=str(block.resource.uri),
                                    text=block.resource.text,
                                    mime_type=block.resource.mimeType,
                                )
                            case BlobResourceContents():
                                embedded_res = embedded_blob_resource(
                                    uri=str(block.resource.uri),
                                    blob=block.resource.blob,
                                    mime_type=block.resource.mimeType,
                                )
                            case _:
                                continue  # Skip unsupported resource types
                        acp_content.append(tool_content(resource_block(embedded_res)))

                    case _:
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
            tool_use_id: LLM's tool use ID (for matching with stream events)

        Returns:
            The tool call ID for tracking
        """
        # Check if we already sent a stream notification for this tool_use_id
        existing_external_id = None
        if tool_use_id:
            # If there's a pending stream task, await it first
            pending_task = self._stream_tasks.get(tool_use_id)
            if pending_task and not pending_task.done():
                logger.debug(
                    f"Waiting for pending stream notification task to complete: {tool_use_id}",
                    name="acp_tool_await_stream_task",
                    tool_use_id=tool_use_id,
                )
                try:
                    await pending_task
                except Exception as e:
                    logger.warning(
                        f"Stream notification task failed: {e}",
                        name="acp_stream_task_failed",
                        tool_use_id=tool_use_id,
                        exc_info=True,
                    )

            async with self._lock:
                existing_external_id = self._stream_tool_use_ids.get(tool_use_id)
                if existing_external_id:
                    logger.debug(
                        f"Found existing stream notification for tool_use_id: {tool_use_id}",
                        name="acp_tool_execution_match",
                        tool_use_id=tool_use_id,
                        external_id=existing_external_id,
                    )
                else:
                    logger.debug(
                        f"No stream notification found for tool_use_id: {tool_use_id}",
                        name="acp_tool_execution_no_match",
                        tool_use_id=tool_use_id,
                        available_ids=list(self._stream_tool_use_ids.keys()),
                    )

        # Infer tool kind
        kind = self._infer_tool_kind(tool_name, arguments)

        # Create title
        title = f"{server_name}/{tool_name}"
        if arguments:
            # Include trimmed arg list info in title
            arg_str = ", ".join(f"{k}={v}" for k, v in list(arguments.items()))
            if len(arg_str) > 50:
                arg_str = arg_str[:47] + "..."
            title = f"{title}({arg_str})"

        # Use SDK tracker to create or update the tool call notification
        async with self._lock:
            if existing_external_id:
                # Get final chunk count before clearing
                final_chunk_count = self._stream_chunk_counts.get(tool_use_id or "", 0)

                # Update title with streamed count only if we showed streaming progress
                if final_chunk_count >= 25:
                    title = f"{title} (streamed {final_chunk_count} chunks)"

                # Update the existing stream notification with full details
                # Clear streaming content by setting content=[] since we now have full rawInput
                tool_call_update = self._tracker.progress(
                    external_id=existing_external_id,
                    title=title,  # Update with server_name and args
                    kind=kind,  # Re-infer with arguments
                    status="in_progress",  # Move from pending to in_progress
                    raw_input=arguments,  # Add complete arguments
                    content=[],  # Clear streaming content
                )
                tool_call_id = tool_call_update.toolCallId

                # Ensure mapping exists - progress() may return different ID than start()
                # or the stream notification task may not have stored it yet
                self._tool_call_id_to_external_id[tool_call_id] = existing_external_id
                # Store simple title (server/tool) for progress updates - no args
                self._simple_titles[tool_call_id] = f"{server_name}/{tool_name}"
                # Store full title (with args) for completion
                self._full_titles[tool_call_id] = title

                # Clean up streaming state since we're now in execution
                if tool_use_id:
                    self._stream_chunk_counts.pop(tool_use_id, None)
                    self._stream_base_titles.pop(tool_use_id, None)
                    self._stream_tool_use_ids.pop(tool_use_id, None)

                logger.debug(
                    f"Updated stream tool call with execution details: {tool_call_id}",
                    name="acp_tool_execution_update",
                    tool_call_id=tool_call_id,
                    external_id=existing_external_id,
                    tool_name=tool_name,
                    server_name=server_name,
                    tool_use_id=tool_use_id,
                )
            else:
                # No stream notification - create new one (normal path)
                external_id = str(uuid.uuid4())
                tool_call_start = self._tracker.start(
                    external_id=external_id,
                    title=title,
                    kind=kind,
                    status="pending",
                    raw_input=arguments,
                )
                # Store mapping from ACP tool_call_id to external_id for later lookups
                self._tool_call_id_to_external_id[tool_call_start.toolCallId] = external_id
                tool_call_id = tool_call_start.toolCallId
                tool_call_update = tool_call_start
                # Store simple title (server/tool) for progress updates - no args
                self._simple_titles[tool_call_id] = f"{server_name}/{tool_name}"
                # Store full title (with args) for completion
                self._full_titles[tool_call_id] = title

                logger.debug(
                    f"Started tool call tracking: {tool_call_id}",
                    name="acp_tool_call_start",
                    tool_call_id=tool_call_id,
                    external_id=external_id,
                    tool_name=tool_name,
                    server_name=server_name,
                )

        # Send notification (either new start or update)
        try:
            await self._connection.session_update(
                session_id=self._session_id, update=tool_call_update
            )
        except Exception as e:
            logger.error(
                f"Error sending tool_call notification: {e}",
                name="acp_tool_call_error",
                exc_info=True,
            )

        # Return the ACP tool_call_id for caller to track
        return tool_call_id

    async def on_tool_permission_denied(
        self,
        tool_name: str,
        server_name: str,
        tool_use_id: str | None,
        error: str | None = None,
    ) -> None:
        """
        Called when tool execution is denied before it starts.

        Uses any pending stream-start notification to mark the call as failed
        so ACP clients see the cancellation/denial.
        """
        if not tool_use_id:
            return

        # Wait for any pending stream notification to finish
        pending_task = self._stream_tasks.get(tool_use_id)
        if pending_task and not pending_task.done():
            try:
                await pending_task
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Stream notification task failed for denied tool: {e}",
                    name="acp_permission_denied_stream_task_failed",
                    tool_use_id=tool_use_id,
                    exc_info=True,
                )

        async with self._lock:
            external_id = self._stream_tool_use_ids.get(tool_use_id)

            if not external_id:
                # No stream notification; nothing to update
                return

            try:
                update_data = self._tracker.progress(
                    external_id=external_id,
                    status="failed",
                    content=[tool_content(text_block(error))] if error else None,
                )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"Error creating permission-denied update: {e}",
                    name="acp_permission_denied_update_error",
                    exc_info=True,
                )
                return

        # Send the failure notification
        try:
            await self._connection.session_update(session_id=self._session_id, update=update_data)
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"Error sending permission-denied notification: {e}",
                name="acp_permission_denied_notification_error",
                exc_info=True,
            )
        finally:
            # Clean up tracker and mappings
            async with self._lock:
                self._tracker.forget(external_id)
                self._stream_tool_use_ids.pop(tool_use_id, None)
                self._stream_chunk_counts.pop(tool_use_id, None)
                self._stream_base_titles.pop(tool_use_id, None)

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
        Updates the title with progress percentage and/or message.

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

            # Build updated title with progress info (using simple title without args)
            simple_title = self._simple_titles.get(tool_call_id, "Tool")
            title_parts = [simple_title]

            # Add progress indicator
            if total is not None and total > 0:
                # Show progress/total format (e.g., [50/100])
                title_parts.append(f"[{progress:.0f}/{total:.0f}]")
            else:
                # Show just progress value (e.g., [50])
                title_parts.append(f"[{progress:.0f}]")

            # Add message if present
            if message:
                title_parts.append(f"- {message}")

            updated_title = " ".join(title_parts)

            # Use SDK tracker to create progress update with updated title
            # Note: We don't include content since the title now shows the progress message
            try:
                update_data = self._tracker.progress(
                    external_id=external_id,
                    status="in_progress",
                    title=updated_title,
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
            await self._connection.session_update(session_id=self._session_id, update=update_data)

            logger.debug(
                f"Updated tool call progress: {tool_call_id}",
                name="acp_tool_progress_update",
                progress=progress,
                total=total,
                progress_message=message,
                title=updated_title,
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
        content: list[MCPContentBlock] | None = None,
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
        logger.debug(
            f"on_tool_complete called: {tool_call_id}",
            name="acp_tool_complete_entry",
            success=success,
            has_content=content is not None,
            content_types=[type(c).__name__ for c in (content or [])],
            has_error=error is not None,
        )

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
                # Restore full title with parameters for completion
                full_title = self._full_titles.get(tool_call_id)
                update_data = self._tracker.progress(
                    external_id=external_id,
                    status=status,
                    title=full_title,  # Restore original title with args
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
            await self._connection.session_update(session_id=self._session_id, update=update_data)

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
                self._simple_titles.pop(tool_call_id, None)
                self._full_titles.pop(tool_call_id, None)

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
            tracker_calls = getattr(self._tracker, "_calls", {})
            for external_id in list(tracker_calls.keys()):
                self._tracker.forget(external_id)
            self._tool_call_id_to_external_id.clear()
            self._simple_titles.clear()
            self._full_titles.clear()
            self._stream_tool_use_ids.clear()
            self._stream_chunk_counts.clear()
            self._stream_base_titles.clear()

        logger.debug(
            f"Cleaned up {count} tool trackers for session {session_id}",
            name="acp_tool_cleanup",
        )
