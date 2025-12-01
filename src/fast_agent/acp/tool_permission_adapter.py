"""
ACP Tool Permission Adapter

Bridges ACPToolPermissionManager to the MCP ToolPermissionHandler protocol,
allowing ACP permission checking to be injected into the MCP aggregator.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fast_agent.acp.permission_store import PermissionStore
from fast_agent.acp.tool_permissions import ACPToolPermissionManager
from fast_agent.mcp.common import create_namespaced_name
from fast_agent.mcp.tool_permission_handler import ToolPermissionHandler, ToolPermissionResult

if TYPE_CHECKING:
    from acp import AgentSideConnection

    from fast_agent.acp.tool_progress import ACPToolProgressManager


class ACPToolPermissionAdapter(ToolPermissionHandler):
    """
    Adapts ACPToolPermissionManager to implement the ToolPermissionHandler protocol.

    This adapter translates between the ACP-specific permission types and the
    generic MCP permission handler interface.
    """

    def __init__(
        self,
        connection: "AgentSideConnection",
        session_id: str,
        store: PermissionStore | None = None,
        cwd: str | Path | None = None,
        tool_handler: "ACPToolProgressManager | None" = None,
    ) -> None:
        """
        Initialize the adapter.

        Args:
            connection: The ACP connection to send permission requests on
            session_id: The ACP session ID
            store: Optional PermissionStore for persistence
            cwd: Working directory for the store (only used if store not provided)
            tool_handler: Optional tool progress manager for toolCallId lookup
        """
        self._tool_handler = tool_handler
        self._manager = ACPToolPermissionManager(
            connection=connection,
            session_id=session_id,
            store=store,
            cwd=cwd,
        )

    @property
    def manager(self) -> ACPToolPermissionManager:
        """Access the underlying permission manager."""
        return self._manager

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> ToolPermissionResult:
        """
        Check if tool execution is permitted.

        Delegates to ACPToolPermissionManager and converts the result
        to ToolPermissionResult.

        Args:
            tool_name: Name of the tool to execute
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments
            tool_use_id: LLM's tool use ID

        Returns:
            ToolPermissionResult indicating whether execution is allowed
        """
        # Look up the ACP toolCallId if a streaming notification was already sent
        # This ensures the permission request references the same tool call
        tool_call_id = tool_use_id
        if tool_use_id and self._tool_handler:
            acp_tool_call_id = await self._tool_handler.get_tool_call_id_for_tool_use(tool_use_id)
            if acp_tool_call_id:
                tool_call_id = acp_tool_call_id

        result = await self._manager.check_permission(
            tool_name=tool_name,
            server_name=server_name,
            arguments=arguments,
            tool_call_id=tool_call_id,
        )

        namespaced_tool_name = create_namespaced_name(server_name, tool_name)

        # Convert PermissionResult to ToolPermissionResult
        if result.is_cancelled:
            return ToolPermissionResult.cancelled()
        elif result.allowed:
            return ToolPermissionResult(allowed=True, remember=result.remember)
        else:
            # Distinguish between one-time and persistent rejection for clearer UX
            if result.remember:
                error_message = (
                    f"The user has permanently declined permission to use this tool: "
                    f"{namespaced_tool_name}"
                )
            else:
                error_message = (
                    f"The user has declined permission to use this tool: {namespaced_tool_name}"
                )

            return ToolPermissionResult(
                allowed=False,
                remember=result.remember,
                error_message=error_message,
            )

    async def clear_session_cache(self) -> None:
        """Clear the session-level permission cache."""
        await self._manager.clear_session_cache()
