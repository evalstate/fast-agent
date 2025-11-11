"""
ACP Tool Wrapper - Wraps MCP tool execution with ACP-specific functionality.

This module provides a wrapper that intercepts MCP tool calls and adds:
- Tool call lifecycle notifications (ToolCallStart, ToolCallProgress, completion)
- Permission checking before tool execution
- Progress forwarding from MCP to ACP
"""

from typing import TYPE_CHECKING, Any, Optional

from mcp.types import CallToolResult

from fast_agent.acp.tool_call_manager import ToolCallManager
from fast_agent.acp.tool_permission import ToolPermissionContext, ToolPermissionManager
from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_aggregator import MCPAggregator

logger = get_logger(__name__)


class ACPToolWrapper:
    """
    Wraps MCPAggregator to provide ACP tool call notifications and permissions.

    This wrapper:
    1. Checks permissions before executing tools
    2. Sends ToolCallStart notifications
    3. Forwards MCP progress to ACP ToolCallProgress
    4. Sends completion/failure notifications
    """

    def __init__(
        self,
        aggregator: "MCPAggregator",
        tool_call_manager: ToolCallManager,
        permission_manager: Optional[ToolPermissionManager] = None,
    ):
        """
        Initialize the ACP tool wrapper.

        Args:
            aggregator: The underlying MCPAggregator
            tool_call_manager: Manager for tool call lifecycle
            permission_manager: Optional permission manager (if None, auto-allow all)
        """
        self.aggregator = aggregator
        self.tool_call_manager = tool_call_manager
        self.permission_manager = permission_manager

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """
        Call a tool with ACP wrapping.

        This method:
        1. Parses the server and tool name
        2. Checks permissions
        3. Creates tool call notification
        4. Executes the tool
        5. Sends progress updates (via modified callback)
        6. Sends completion notification

        Args:
            name: Tool name (possibly namespaced as "server__tool")
            arguments: Tool arguments

        Returns:
            CallToolResult from the tool execution
        """
        # Parse server and tool name
        server_name, local_tool_name = await self.aggregator._parse_resource_name(name, "tool")

        if server_name is None:
            logger.error(f"Tool '{name}' not found")
            from mcp.types import TextContent
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Tool '{name}' not found")],
            )

        # Create tool call notification
        tool_call_id = await self.tool_call_manager.create_tool_call(
            tool_name=local_tool_name,
            server_name=server_name,
            arguments=arguments or {},
        )

        try:
            # Check permissions (if permission manager is configured)
            if self.permission_manager:
                permission_context = ToolPermissionContext(
                    session_id=self.tool_call_manager.session_id,
                    tool_name=local_tool_name,
                    server_name=server_name,
                    arguments=arguments or {},
                    tool_call_id=tool_call_id,
                )

                permission_result = await self.permission_manager.check_permission(
                    permission_context
                )

                if permission_result.cancelled:
                    # User cancelled the prompt turn
                    logger.info(
                        "Tool execution cancelled by user",
                        tool_call_id=tool_call_id,
                        tool_name=local_tool_name,
                    )
                    # Mark as failed
                    await self.tool_call_manager.complete_tool_call(
                        tool_call_id=tool_call_id,
                        output="Cancelled by user",
                        is_error=True,
                    )
                    from mcp.types import TextContent
                    return CallToolResult(
                        isError=True,
                        content=[TextContent(type="text", text="Tool execution cancelled by user")],
                    )

                if not permission_result.allowed:
                    # Permission denied
                    logger.info(
                        "Tool execution denied by permission handler",
                        tool_call_id=tool_call_id,
                        tool_name=local_tool_name,
                    )
                    # Mark as failed
                    await self.tool_call_manager.complete_tool_call(
                        tool_call_id=tool_call_id,
                        output="Permission denied",
                        is_error=True,
                    )
                    from mcp.types import TextContent
                    return CallToolResult(
                        isError=True,
                        content=[TextContent(type="text", text="Tool execution denied by user")],
                    )

            # Update status to in_progress
            await self.tool_call_manager.update_tool_call(
                tool_call_id=tool_call_id,
                status="in_progress",
            )

            # Wrap the progress callback to forward to tool call manager
            original_callback = self.aggregator._create_progress_callback(
                server_name, local_tool_name
            )

            async def acp_progress_callback(
                progress: float,
                total: float | None,
                message: str | None,
            ) -> None:
                """Combined progress callback that calls both MCP and ACP handlers."""
                # Call original MCP callback (for logging)
                await original_callback(progress, total, message)
                # Forward to ACP tool call manager
                await self.tool_call_manager.progress_update(
                    tool_call_id=tool_call_id,
                    progress=progress,
                    total=total,
                    message=message,
                )

            # Execute the tool with progress forwarding
            # We need to temporarily replace the aggregator's progress callback
            # Since _execute_on_server uses _create_progress_callback, we'll call it directly
            from opentelemetry import trace

            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"MCP Tool: {server_name}/{local_tool_name}"):
                trace.get_current_span().set_attribute("tool_name", local_tool_name)
                trace.get_current_span().set_attribute("server_name", server_name)

                result = await self.aggregator._execute_on_server(
                    server_name=server_name,
                    operation_type="tools/call",
                    operation_name=local_tool_name,
                    method_name="call_tool",
                    method_args={
                        "name": local_tool_name,
                        "arguments": arguments,
                    },
                    error_factory=lambda msg: CallToolResult(
                        isError=True,
                        content=[
                            __import__("mcp.types", fromlist=["TextContent"]).TextContent(
                                type="text", text=msg
                            )
                        ],
                    ),
                    progress_callback=acp_progress_callback,
                )

            # Mark as completed
            await self.tool_call_manager.complete_tool_call(
                tool_call_id=tool_call_id,
                output=result,
                is_error=result.isError,
            )

            return result

        except Exception as e:
            # Mark as failed
            logger.error(
                f"Tool execution failed: {e}",
                tool_call_id=tool_call_id,
                tool_name=local_tool_name,
                exc_info=True,
            )
            await self.tool_call_manager.complete_tool_call(
                tool_call_id=tool_call_id,
                output=str(e),
                is_error=True,
            )
            raise

    # Delegate all other methods to the underlying aggregator
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped aggregator."""
        return getattr(self.aggregator, name)
