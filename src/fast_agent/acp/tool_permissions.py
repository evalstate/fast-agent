"""
ACP Tool Call Permissions

Provides a permission handler that requests tool execution permission from the ACP client.
This follows the same pattern as elicitation handlers but for tool execution authorization.
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from acp.schema import RequestPermissionRequest

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)


@dataclass
class ToolPermissionRequest:
    """Request for tool execution permission."""

    tool_name: str
    server_name: str
    arguments: dict[str, Any] | None
    tool_call_id: str | None = None


@dataclass
class ToolPermissionResponse:
    """Response from tool permission request."""

    allowed: bool
    remember: bool  # Whether to remember this decision
    cancelled: bool = False


# Type for permission handler callbacks
ToolPermissionHandlerT = Callable[[ToolPermissionRequest], asyncio.Future[ToolPermissionResponse]]


class ACPToolPermissionManager:
    """
    Manages tool execution permission requests via ACP.

    This class provides a handler that can be used to request permission
    from the ACP client before executing tools.
    """

    def __init__(self, connection: "AgentSideConnection") -> None:
        """
        Initialize the permission manager.

        Args:
            connection: The ACP connection to send permission requests on
        """
        self._connection = connection
        self._remembered_permissions: dict[str, bool] = {}
        self._lock = asyncio.Lock()

    def _get_permission_key(self, tool_name: str, server_name: str) -> str:
        """Get a unique key for remembering permissions."""
        return f"{server_name}/{tool_name}"

    async def request_permission(
        self,
        session_id: str,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> ToolPermissionResponse:
        """
        Request permission to execute a tool.

        Args:
            session_id: The ACP session ID
            tool_name: Name of the tool to execute
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments
            tool_call_id: Optional tool call ID for tracking

        Returns:
            ToolPermissionResponse indicating whether execution is allowed
        """
        permission_key = self._get_permission_key(tool_name, server_name)

        # Check remembered permissions
        async with self._lock:
            if permission_key in self._remembered_permissions:
                allowed = self._remembered_permissions[permission_key]
                logger.debug(
                    f"Using remembered permission for {permission_key}: {allowed}",
                    name="acp_tool_permission_remembered",
                )
                return ToolPermissionResponse(allowed=allowed, remember=True)

        # Build prompt message
        prompt_parts = [f"Allow execution of tool: {server_name}/{tool_name}"]
        if arguments:
            # Show key arguments (limit to avoid overwhelming the user)
            arg_items = list(arguments.items())[:3]
            arg_str = ", ".join(f"{k}={v}" for k, v in arg_items)
            if len(arguments) > 3:
                arg_str += ", ..."
            prompt_parts.append(f"Arguments: {arg_str}")

        prompt = "\n".join(prompt_parts)

        # Create permission request with options
        # Use plain dictionaries for options per ACP specification
        options = [
            {
                "id": "allow_once",
                "kind": "allow_once",
                "title": "Allow Once",
                "description": "Allow this tool execution",
            },
            {
                "id": "allow_always",
                "kind": "allow_always",
                "title": "Always Allow",
                "description": f"Always allow {server_name}/{tool_name}",
            },
            {
                "id": "reject_once",
                "kind": "reject_once",
                "title": "Reject Once",
                "description": "Reject this tool execution",
            },
            {
                "id": "reject_always",
                "kind": "reject_always",
                "title": "Never Allow",
                "description": f"Never allow {server_name}/{tool_name}",
            },
        ]

        request = RequestPermissionRequest(
            sessionId=session_id,
            prompt=prompt,
            options=options,
            toolCall=tool_call_id,
        )

        try:
            logger.info(
                f"Requesting permission for {permission_key}",
                name="acp_tool_permission_request",
                tool_name=tool_name,
                server_name=server_name,
            )

            # Send permission request to client
            response = await self._connection.requestPermission(request)

            # Handle response
            outcome = response.outcome
            if hasattr(outcome, "outcome"):
                outcome_type = outcome.outcome

                if outcome_type == "cancelled":
                    logger.info(
                        f"Permission request cancelled for {permission_key}",
                        name="acp_tool_permission_cancelled",
                    )
                    return ToolPermissionResponse(allowed=False, remember=False, cancelled=True)

                elif outcome_type == "selected":
                    option_id = getattr(outcome, "optionId", None)

                    if option_id == "allow_once":
                        return ToolPermissionResponse(allowed=True, remember=False)

                    elif option_id == "allow_always":
                        async with self._lock:
                            self._remembered_permissions[permission_key] = True
                        logger.info(
                            f"Remembering allow for {permission_key}",
                            name="acp_tool_permission_remember_allow",
                        )
                        return ToolPermissionResponse(allowed=True, remember=True)

                    elif option_id == "reject_once":
                        return ToolPermissionResponse(allowed=False, remember=False)

                    elif option_id == "reject_always":
                        async with self._lock:
                            self._remembered_permissions[permission_key] = False
                        logger.info(
                            f"Remembering reject for {permission_key}",
                            name="acp_tool_permission_remember_reject",
                        )
                        return ToolPermissionResponse(allowed=False, remember=True)

            # Default to rejection if we can't parse the response
            logger.warning(
                f"Unknown permission response for {permission_key}, defaulting to reject",
                name="acp_tool_permission_unknown",
            )
            return ToolPermissionResponse(allowed=False, remember=False)

        except Exception as e:
            logger.error(
                f"Error requesting tool permission: {e}",
                name="acp_tool_permission_error",
                exc_info=True,
            )
            # Default to allowing on error to avoid breaking execution
            # Real implementations might want to configure this behavior
            return ToolPermissionResponse(allowed=True, remember=False)

    async def clear_remembered_permissions(self, tool_name: str | None = None, server_name: str | None = None) -> None:
        """
        Clear remembered permissions.

        Args:
            tool_name: Optional tool name to clear (clears all if None)
            server_name: Optional server name to clear (clears all if None)
        """
        async with self._lock:
            if tool_name and server_name:
                permission_key = self._get_permission_key(tool_name, server_name)
                self._remembered_permissions.pop(permission_key, None)
                logger.info(
                    f"Cleared permission for {permission_key}",
                    name="acp_tool_permission_cleared",
                )
            else:
                self._remembered_permissions.clear()
                logger.info(
                    "Cleared all remembered permissions",
                    name="acp_tool_permissions_cleared_all",
                )


def create_acp_permission_handler(
    permission_manager: ACPToolPermissionManager,
    session_id: str,
) -> ToolPermissionHandlerT:
    """
    Create a tool permission handler for ACP integration.

    This creates a handler that can be injected into the tool execution
    pipeline to request permission before executing tools.

    Args:
        permission_manager: The ACPToolPermissionManager instance
        session_id: The ACP session ID

    Returns:
        A permission handler function
    """

    async def handler(request: ToolPermissionRequest) -> ToolPermissionResponse:
        """Handle tool permission request."""
        return await permission_manager.request_permission(
            session_id=session_id,
            tool_name=request.tool_name,
            server_name=request.server_name,
            arguments=request.arguments,
            tool_call_id=request.tool_call_id,
        )

    return handler
