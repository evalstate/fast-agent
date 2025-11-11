"""
Tool Permission Handlers - Handle tool execution permission requests for ACP.

This module provides permission handler infrastructure similar to elicitation handlers.
Handlers can request permission from ACP clients before executing tools.
"""

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from acp.schema import (
    PermissionOption,
    RequestPermissionOutcome1,
    RequestPermissionOutcome2,
    RequestPermissionRequest,
    RequestPermissionResponse,
    ToolCall,
)
from pydantic import BaseModel

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)


class ToolPermissionContext(BaseModel):
    """Context for tool permission requests."""

    session_id: str
    tool_name: str
    server_name: str
    arguments: dict[str, Any]
    tool_call_id: str


class ToolPermissionResult(BaseModel):
    """Result of a tool permission request."""

    allowed: bool
    remember: bool = False  # Whether to remember this decision (allow_always/reject_always)
    cancelled: bool = False  # Whether the user cancelled the prompt turn


# Type alias for tool permission handler functions
ToolPermissionHandlerFnT = Callable[
    [ToolPermissionContext, "AgentSideConnection"],
    Awaitable[ToolPermissionResult],
]


async def acp_tool_permission_handler(
    context: ToolPermissionContext,
    connection: "AgentSideConnection",
) -> ToolPermissionResult:
    """
    ACP tool permission handler.

    Sends session/request_permission to the client and waits for a response.

    Args:
        context: Permission request context
        connection: ACP connection

    Returns:
        ToolPermissionResult with the client's decision
    """
    logger.info(
        "Requesting tool permission from ACP client",
        tool_name=context.tool_name,
        server_name=context.server_name,
        session_id=context.session_id,
    )

    # Build the ToolCall object for the permission request
    tool_call = ToolCall(
        toolCallId=context.tool_call_id,
        title=f"{context.tool_name} ({context.server_name})",
        rawInput=context.arguments,
    )

    # Define permission options
    options = [
        PermissionOption(
            optionId="allow_once",
            kind="allow_once",
            label="Allow Once",
        ),
        PermissionOption(
            optionId="allow_always",
            kind="allow_always",
            label=f"Always Allow {context.tool_name}",
        ),
        PermissionOption(
            optionId="reject_once",
            kind="reject_once",
            label="Reject Once",
        ),
        PermissionOption(
            optionId="reject_always",
            kind="reject_always",
            label=f"Always Reject {context.tool_name}",
        ),
    ]

    # Build the request
    request = RequestPermissionRequest(
        sessionId=context.session_id,
        toolCall=tool_call,
        options=options,
    )

    try:
        # Send the request and wait for response
        response_dict = await connection._conn.send_request(
            "session/request_permission",
            request.model_dump(exclude_none=True),
        )

        # Parse the response
        response = RequestPermissionResponse(**response_dict)

        # Check for cancellation
        if isinstance(response.outcome, RequestPermissionOutcome2):
            # Cancelled outcome
            logger.info(
                "Tool permission request cancelled",
                tool_name=context.tool_name,
                session_id=context.session_id,
            )
            return ToolPermissionResult(
                allowed=False,
                cancelled=True,
            )

        # Extract the selected option
        outcome: RequestPermissionOutcome1 = response.outcome
        selected_id = outcome.optionId

        # Determine result based on selection
        if selected_id in ["allow_once", "allow_always"]:
            allowed = True
            remember = selected_id == "allow_always"
        elif selected_id in ["reject_once", "reject_always"]:
            allowed = False
            remember = selected_id == "reject_always"
        else:
            # Unknown option - default to reject
            logger.warning(
                f"Unknown permission option: {selected_id}",
                tool_name=context.tool_name,
            )
            allowed = False
            remember = False

        logger.info(
            "Tool permission decision received",
            tool_name=context.tool_name,
            allowed=allowed,
            remember=remember,
            session_id=context.session_id,
        )

        return ToolPermissionResult(
            allowed=allowed,
            remember=remember,
        )

    except Exception as e:
        logger.error(
            f"Error requesting tool permission: {e}",
            tool_name=context.tool_name,
            session_id=context.session_id,
            exc_info=True,
        )
        # Default to rejecting on error
        return ToolPermissionResult(allowed=False)


async def allow_all_tool_permission_handler(
    context: ToolPermissionContext,
    connection: "AgentSideConnection",
) -> ToolPermissionResult:
    """
    Permission handler that allows all tool executions without prompting.

    Use this for non-ACP mode or when tool permissions are not required.

    Args:
        context: Permission request context
        connection: ACP connection (unused)

    Returns:
        ToolPermissionResult allowing execution
    """
    logger.debug(
        "Auto-allowing tool execution (allow-all handler)",
        tool_name=context.tool_name,
        server_name=context.server_name,
    )
    return ToolPermissionResult(allowed=True)


async def deny_all_tool_permission_handler(
    context: ToolPermissionContext,
    connection: "AgentSideConnection",
) -> ToolPermissionResult:
    """
    Permission handler that denies all tool executions.

    Useful for testing or read-only modes.

    Args:
        context: Permission request context
        connection: ACP connection (unused)

    Returns:
        ToolPermissionResult denying execution
    """
    logger.debug(
        "Auto-denying tool execution (deny-all handler)",
        tool_name=context.tool_name,
        server_name=context.server_name,
    )
    return ToolPermissionResult(allowed=False)


class ToolPermissionManager:
    """
    Manages tool permissions with caching for allow_always/reject_always decisions.
    """

    def __init__(
        self,
        handler: ToolPermissionHandlerFnT,
        connection: "AgentSideConnection",
    ):
        """
        Initialize the permission manager.

        Args:
            handler: Permission handler function
            connection: ACP connection
        """
        self.handler = handler
        self.connection = connection

        # Cache for remembered decisions: {tool_name: bool}
        self._allowed_cache: set[str] = set()
        self._denied_cache: set[str] = set()

    async def check_permission(
        self,
        context: ToolPermissionContext,
    ) -> ToolPermissionResult:
        """
        Check if a tool execution is permitted.

        Checks cached decisions first, then delegates to the handler.

        Args:
            context: Permission request context

        Returns:
            ToolPermissionResult with the decision
        """
        tool_key = f"{context.server_name}:{context.tool_name}"

        # Check cached allow_always decisions
        if tool_key in self._allowed_cache:
            logger.debug(
                "Tool allowed via cached decision",
                tool_name=context.tool_name,
                server_name=context.server_name,
            )
            return ToolPermissionResult(allowed=True, remember=True)

        # Check cached reject_always decisions
        if tool_key in self._denied_cache:
            logger.debug(
                "Tool denied via cached decision",
                tool_name=context.tool_name,
                server_name=context.server_name,
            )
            return ToolPermissionResult(allowed=False, remember=True)

        # No cached decision - ask the handler
        result = await self.handler(context, self.connection)

        # Cache the decision if remember=True
        if result.remember:
            if result.allowed:
                self._allowed_cache.add(tool_key)
                logger.info(
                    "Cached allow_always decision",
                    tool_name=context.tool_name,
                    server_name=context.server_name,
                )
            else:
                self._denied_cache.add(tool_key)
                logger.info(
                    "Cached reject_always decision",
                    tool_name=context.tool_name,
                    server_name=context.server_name,
                )

        return result

    def clear_cache(self) -> None:
        """Clear all cached permission decisions."""
        self._allowed_cache.clear()
        self._denied_cache.clear()
        logger.info("Cleared tool permission cache")
