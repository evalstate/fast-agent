"""
Tool Call Permission Handler for Agent Client Protocol (ACP).

This module implements permission handling for tool calls according to the
ACP specification at https://agentclientprotocol.com/protocol/tool-calls.md

Permission handlers allow customization of how tool call permissions are requested
from the ACP client, similar to how elicitation handlers work for MCP servers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from acp.schema import (
    PermissionOption,
    RequestPermissionRequest,
    RequestPermissionResponse,
)

from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ToolCallPermissionRequest:
    """
    Information about a tool call that needs permission.

    Attributes:
        tool_name: Name of the tool being called
        tool_title: Human-readable description of what the tool does
        tool_kind: Category of the tool (read, edit, delete, etc.)
        arguments: Arguments being passed to the tool
        session_id: The ACP session ID
    """

    tool_name: str
    tool_title: str
    tool_kind: str
    arguments: dict[str, Any] | None
    session_id: str


@dataclass
class ToolCallPermissionResponse:
    """
    Response to a tool call permission request.

    Attributes:
        allowed: Whether the tool call is allowed
        remember: Whether to remember this decision for future calls
        option_id: The ID of the selected permission option (if applicable)
        cancelled: Whether the request was cancelled
    """

    allowed: bool
    remember: bool = False
    option_id: Optional[str] = None
    cancelled: bool = False


class ToolCallPermissionHandler(ABC):
    """
    Abstract base class for tool call permission handlers.

    Subclasses can implement custom logic for requesting and managing
    tool call permissions from ACP clients.
    """

    @abstractmethod
    async def request_permission(
        self,
        request: ToolCallPermissionRequest,
        connection: Any,  # AgentSideConnection
    ) -> ToolCallPermissionResponse:
        """
        Request permission to execute a tool call.

        Args:
            request: Information about the tool call
            connection: The ACP connection for sending requests

        Returns:
            Permission response indicating whether to proceed
        """
        pass


class ACPClientPermissionHandler(ToolCallPermissionHandler):
    """
    Default permission handler that requests permission from the ACP client.

    This handler uses the standard ACP session/request_permission method
    to ask the client whether to allow the tool call.
    """

    async def request_permission(
        self,
        request: ToolCallPermissionRequest,
        connection: Any,
    ) -> ToolCallPermissionResponse:
        """
        Request permission from the ACP client using session/request_permission.

        Args:
            request: Information about the tool call
            connection: The ACP connection

        Returns:
            Permission response based on client's answer
        """
        try:
            # Build permission options
            options = [
                PermissionOption(
                    id="allow_once",
                    title="Allow Once",
                    description="Allow this tool call one time",
                ),
                PermissionOption(
                    id="allow_always",
                    title="Always Allow",
                    description=f"Always allow {request.tool_name} tool calls",
                ),
                PermissionOption(
                    id="reject_once",
                    title="Reject Once",
                    description="Reject this tool call one time",
                ),
                PermissionOption(
                    id="reject_always",
                    title="Always Reject",
                    description=f"Always reject {request.tool_name} tool calls",
                ),
            ]

            # Create the permission request
            args = ""
            if request.arguments:
                # Format arguments nicely
                arg_strs = [f"{k}={v}" for k, v in request.arguments.items()]
                args = f"({', '.join(arg_strs)})"

            message = (
                f"The agent wants to execute a tool:\n\n"
                f"Tool: {request.tool_name}{args}\n"
                f"Action: {request.tool_title}\n"
                f"Type: {request.tool_kind}\n\n"
                f"Do you want to allow this?"
            )

            permission_request = RequestPermissionRequest(
                sessionId=request.session_id,
                title=f"Tool Call Permission: {request.tool_name}",
                message=message,
                options=options,
            )

            # Send the request
            logger.info(
                f"Requesting permission for tool call: {request.tool_name}",
                name="acp_tool_permission_request",
                tool_name=request.tool_name,
            )

            response: RequestPermissionResponse = await connection.requestPermission(
                permission_request
            )

            # Parse the response
            outcome = response.outcome

            # Handle cancellation
            if outcome.outcome == "cancelled":
                logger.info(
                    f"Tool call permission cancelled: {request.tool_name}",
                    name="acp_tool_permission_cancelled",
                )
                return ToolCallPermissionResponse(
                    allowed=False,
                    cancelled=True,
                )

            # Handle selected option
            if outcome.outcome == "selected":
                option_id = outcome.optionId
                allowed = option_id in ["allow_once", "allow_always"]
                remember = option_id in ["allow_always", "reject_always"]

                logger.info(
                    f"Tool call permission {option_id}: {request.tool_name}",
                    name="acp_tool_permission_response",
                    option_id=option_id,
                    allowed=allowed,
                )

                return ToolCallPermissionResponse(
                    allowed=allowed,
                    remember=remember,
                    option_id=option_id,
                )

            # Default to denied
            logger.warning(
                f"Unexpected permission outcome: {outcome.outcome}",
                name="acp_tool_permission_unexpected",
            )
            return ToolCallPermissionResponse(allowed=False)

        except Exception as e:
            logger.error(
                f"Error requesting tool call permission: {e}",
                name="acp_tool_permission_error",
                exc_info=True,
            )
            # Default to denied on error
            return ToolCallPermissionResponse(allowed=False)


class AlwaysAllowPermissionHandler(ToolCallPermissionHandler):
    """
    Permission handler that always allows all tool calls.

    Useful for trusted environments where permission prompts are not needed.
    """

    async def request_permission(
        self,
        request: ToolCallPermissionRequest,
        connection: Any,
    ) -> ToolCallPermissionResponse:
        """Always allow tool calls without prompting."""
        logger.debug(
            f"Auto-allowing tool call: {request.tool_name}",
            name="acp_tool_permission_auto_allow",
        )
        return ToolCallPermissionResponse(allowed=True)


# Type alias for permission handler factory functions
ToolCallPermissionHandlerFactory = Callable[[], ToolCallPermissionHandler]


# Global default handler
_default_permission_handler: Optional[ToolCallPermissionHandler] = None


def set_default_permission_handler(handler: ToolCallPermissionHandler) -> None:
    """Set the default tool call permission handler."""
    global _default_permission_handler
    _default_permission_handler = handler


def get_default_permission_handler() -> ToolCallPermissionHandler:
    """Get the default tool call permission handler."""
    global _default_permission_handler
    if _default_permission_handler is None:
        # Default to always allow (for backward compatibility)
        _default_permission_handler = AlwaysAllowPermissionHandler()
    return _default_permission_handler
