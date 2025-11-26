"""
ACP Tool Permission Manager

Implements the ToolPermissionHandler protocol for ACP sessions.
Requests tool execution permission from the ACP client when needed,
with support for persistent permissions via .fast-agent/auths.md.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from acp.schema import PermissionOption, RequestPermissionRequest, ToolCall

from fast_agent.acp.permission_store import PermissionStore
from fast_agent.acp.tool_permission_handler import (
    PERMISSION_DENIED_MESSAGE,
    PermissionResult,
)
from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)


class ACPToolPermissionManager:
    """
    Manages tool execution permission requests via ACP.

    Implements ToolPermissionHandler protocol. Checks persistent permissions
    first, then requests permission from the ACP client if needed.

    On error (e.g., client disconnect), defaults to DENY for safety.
    """

    def __init__(
        self,
        connection: "AgentSideConnection",
        session_id: str,
        store: PermissionStore | None = None,
        base_path: Path | None = None,
    ) -> None:
        """
        Initialize the permission manager.

        Args:
            connection: The ACP connection to send permission requests on
            session_id: The ACP session ID
            store: Optional permission store (creates one if not provided)
            base_path: Base path for permission store (defaults to cwd)
        """
        self._connection = connection
        self._session_id = session_id
        self._store = store or PermissionStore(base_path)

    def _make_permission_key(self, server_name: str, tool_name: str) -> str:
        """Create a display key for logging."""
        return f"{server_name}/{tool_name}"

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None,
    ) -> PermissionResult:
        """
        Check if tool execution is permitted.

        Implements ToolPermissionHandler.check_permission.

        1. Checks persistent store for remembered permissions
        2. If not found, requests permission from ACP client
        3. On error, returns DENY (fail-safe)

        Args:
            tool_name: Name of the tool
            server_name: Name of the server/runtime
            arguments: Tool arguments

        Returns:
            PermissionResult indicating whether to proceed
        """
        permission_key = self._make_permission_key(server_name, tool_name)

        # 1. Check persistent store first
        stored = await self._store.get(server_name, tool_name)
        if stored == "allow_always":
            logger.debug(
                f"Using stored permission for {permission_key}: allow_always",
                name="acp_permission_stored",
            )
            return PermissionResult.allow(remember=True)
        if stored == "reject_always":
            logger.debug(
                f"Using stored permission for {permission_key}: reject_always",
                name="acp_permission_stored",
            )
            return PermissionResult.deny(remember=True, message=PERMISSION_DENIED_MESSAGE)

        # 2. Request permission from ACP client
        return await self._request_permission(tool_name, server_name, arguments)

    async def _request_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None,
    ) -> PermissionResult:
        """
        Request permission from the ACP client.

        Args:
            tool_name: Name of the tool
            server_name: Name of the server/runtime
            arguments: Tool arguments

        Returns:
            PermissionResult based on user's decision
        """
        permission_key = self._make_permission_key(server_name, tool_name)

        # Build tool call info for the permission request
        tool_call = ToolCall(
            toolCallId=f"perm_{server_name}_{tool_name}",
            title=f"{server_name}/{tool_name}",
            rawInput=arguments,
        )

        # Create permission options
        options = [
            PermissionOption(
                optionId="allow_once",
                kind="allow_once",
                name="Allow Once",
            ),
            PermissionOption(
                optionId="allow_always",
                kind="allow_always",
                name="Always Allow",
            ),
            PermissionOption(
                optionId="reject_once",
                kind="reject_once",
                name="Reject Once",
            ),
            PermissionOption(
                optionId="reject_always",
                kind="reject_always",
                name="Never Allow",
            ),
        ]

        request = RequestPermissionRequest(
            sessionId=self._session_id,
            toolCall=tool_call,
            options=options,
        )

        try:
            logger.info(
                f"Requesting permission for {permission_key}",
                name="acp_permission_request",
                tool_name=tool_name,
                server_name=server_name,
            )

            # Send permission request to client
            response = await self._connection.requestPermission(request)

            # Handle response
            return await self._handle_permission_response(
                response, server_name, tool_name, permission_key
            )

        except Exception as e:
            # FAIL SAFE: deny on error
            logger.error(
                f"Error requesting permission for {permission_key}: {e}",
                name="acp_permission_error",
                exc_info=True,
            )
            return PermissionResult.deny(message=f"Permission check failed: {e}")

    async def _handle_permission_response(
        self,
        response: Any,
        server_name: str,
        tool_name: str,
        permission_key: str,
    ) -> PermissionResult:
        """
        Handle the permission response from the client.

        Args:
            response: The RequestPermissionResponse from the client
            server_name: Name of the server/runtime
            tool_name: Name of the tool
            permission_key: Display key for logging

        Returns:
            PermissionResult based on user's decision
        """
        outcome = response.outcome

        if hasattr(outcome, "outcome"):
            outcome_type = outcome.outcome

            if outcome_type == "cancelled":
                logger.info(
                    f"Permission cancelled for {permission_key}",
                    name="acp_permission_cancelled",
                )
                return PermissionResult.deny(message="Permission request was cancelled.")

            elif outcome_type == "selected":
                option_id = getattr(outcome, "optionId", None)

                if option_id == "allow_once":
                    logger.info(
                        f"Permission granted (once) for {permission_key}",
                        name="acp_permission_allow_once",
                    )
                    return PermissionResult.allow(remember=False)

                elif option_id == "allow_always":
                    # Store persistent permission
                    await self._store.set(server_name, tool_name, "allow_always")
                    logger.info(
                        f"Permission granted (always) for {permission_key}",
                        name="acp_permission_allow_always",
                    )
                    return PermissionResult.allow(remember=True)

                elif option_id == "reject_once":
                    logger.info(
                        f"Permission denied (once) for {permission_key}",
                        name="acp_permission_reject_once",
                    )
                    return PermissionResult.deny(message=PERMISSION_DENIED_MESSAGE)

                elif option_id == "reject_always":
                    # Store persistent permission
                    await self._store.set(server_name, tool_name, "reject_always")
                    logger.info(
                        f"Permission denied (always) for {permission_key}",
                        name="acp_permission_reject_always",
                    )
                    return PermissionResult.deny(
                        remember=True, message=PERMISSION_DENIED_MESSAGE
                    )

        # Unknown response - fail safe to deny
        logger.warning(
            f"Unknown permission response for {permission_key}, denying",
            name="acp_permission_unknown",
        )
        return PermissionResult.deny(message="Unknown permission response.")
