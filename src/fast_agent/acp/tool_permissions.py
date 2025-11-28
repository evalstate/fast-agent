"""
ACP Tool Call Permissions

Provides a permission handler that requests tool execution permission from the ACP client.
This follows the same pattern as elicitation handlers but for tool execution authorization.

Key behaviors:
- Requests permission from ACP client before executing tools
- Supports allow_once, allow_always, reject_once, reject_always
- Persists "always" decisions in .fast-agent/auths.md
- Fail-safe: defaults to DENY on any error (never silently allow on failure)
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from acp.schema import (
    PermissionOption,
    RequestPermissionRequest,
    ToolCall,
)

from fast_agent.acp.permission_store import (
    PermissionDecision,
    PermissionResult,
    PermissionStore,
    infer_tool_kind,
)
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


@runtime_checkable
class ToolPermissionHandler(Protocol):
    """Protocol for tool permission handlers."""

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> PermissionResult:
        """
        Check if tool execution is permitted.

        Args:
            tool_name: Name of the tool to execute
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments
            tool_call_id: Optional tool call ID for tracking

        Returns:
            PermissionResult indicating whether execution is allowed
        """
        ...


class NoOpPermissionHandler:
    """Permission handler that allows all tool calls (used when permissions are disabled)."""

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> PermissionResult:
        """Always allow tool execution."""
        return PermissionResult.allow_once()


class ACPToolPermissionManager:
    """
    Manages tool execution permission requests via ACP.

    This class provides a handler that can be used to request permission
    from the ACP client before executing tools.

    Key behaviors:
    - Checks PermissionStore for persisted permissions first
    - Sends session/request_permission to client if not cached
    - Persists "always" decisions
    - FAIL-SAFE: Defaults to DENY on any error
    """

    def __init__(
        self,
        connection: "AgentSideConnection",
        session_id: str,
        working_directory: str | None = None,
    ) -> None:
        """
        Initialize the permission manager.

        Args:
            connection: The ACP connection to send permission requests on
            session_id: The ACP session ID
            working_directory: The session's working directory for persisting permissions
        """
        self._connection = connection
        self._session_id = session_id
        self._store = PermissionStore(working_directory)
        # In-memory cache for session (fast lookup, synced with store)
        self._session_cache: dict[str, PermissionDecision] = {}
        self._lock = asyncio.Lock()

    def _get_permission_key(self, tool_name: str, server_name: str) -> str:
        """Get a unique key for remembering permissions."""
        return f"{server_name}/{tool_name}"

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> PermissionResult:
        """
        Check if tool execution is permitted.

        First checks persistent store, then session cache, then requests from client.

        Args:
            tool_name: Name of the tool to execute
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments
            tool_call_id: Optional tool call ID for tracking

        Returns:
            PermissionResult indicating whether execution is allowed
        """
        permission_key = self._get_permission_key(tool_name, server_name)

        try:
            # Check session cache first (fastest)
            async with self._lock:
                if permission_key in self._session_cache:
                    cached = self._session_cache[permission_key]
                    allowed = cached == PermissionDecision.ALLOW_ALWAYS
                    logger.debug(
                        f"Using session-cached permission for {permission_key}: {cached.value}",
                        name="acp_tool_permission_cache",
                    )
                    return PermissionResult(allowed=allowed, remember=True)

            # Check persistent store
            stored = await self._store.check(server_name, tool_name)
            if stored is not None:
                allowed = stored == PermissionDecision.ALLOW_ALWAYS
                # Update session cache
                async with self._lock:
                    self._session_cache[permission_key] = stored
                logger.debug(
                    f"Using persisted permission for {permission_key}: {stored.value}",
                    name="acp_tool_permission_persisted",
                )
                return PermissionResult(allowed=allowed, remember=True)

            # Request permission from client
            return await self._request_permission_from_client(
                tool_name, server_name, arguments, tool_call_id
            )

        except Exception as e:
            # FAIL-SAFE: Default to DENY on any error
            logger.error(
                f"Error checking tool permission for {permission_key}: {e}",
                name="acp_tool_permission_error",
                exc_info=True,
            )
            return PermissionResult.denied()

    async def _request_permission_from_client(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None,
        tool_call_id: str | None,
    ) -> PermissionResult:
        """
        Request permission from the ACP client.

        Args:
            tool_name: Name of the tool
            server_name: Name of the server
            arguments: Tool arguments
            tool_call_id: Optional tool call ID

        Returns:
            PermissionResult based on client response
        """
        permission_key = self._get_permission_key(tool_name, server_name)

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

        # Infer tool kind for the ToolCall object
        kind = infer_tool_kind(tool_name, arguments)

        # Create ToolCall object per ACP spec
        tool_call = ToolCall(
            toolCallId=tool_call_id or "pending",
            title=f"{server_name}/{tool_name}",
            kind=kind,
            status="pending",
            rawInput=arguments,
        )

        # Create permission request with options using SDK's PermissionOption type
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
            prompt=prompt,
            options=options,
            toolCall=tool_call,
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
            return await self._process_permission_response(
                response, tool_name, server_name, permission_key
            )

        except Exception as e:
            # FAIL-SAFE: Default to DENY on any error
            logger.error(
                f"Error requesting tool permission: {e}",
                name="acp_tool_permission_request_error",
                exc_info=True,
            )
            return PermissionResult.denied()

    async def _process_permission_response(
        self,
        response: Any,
        tool_name: str,
        server_name: str,
        permission_key: str,
    ) -> PermissionResult:
        """
        Process the permission response from the client.

        Args:
            response: The response from the client
            tool_name: Name of the tool
            server_name: Name of the server
            permission_key: The permission cache key

        Returns:
            PermissionResult based on the response
        """
        try:
            outcome = response.outcome
            if hasattr(outcome, "outcome"):
                outcome_type = outcome.outcome

                if outcome_type == "cancelled":
                    logger.info(
                        f"Permission request cancelled for {permission_key}",
                        name="acp_tool_permission_cancelled",
                    )
                    return PermissionResult.cancelled_result()

                elif outcome_type == "selected":
                    option_id = getattr(outcome, "optionId", None)

                    if option_id == "allow_once":
                        logger.info(
                            f"Permission allow_once for {permission_key}",
                            name="acp_tool_permission_allow_once",
                        )
                        return PermissionResult.allow_once()

                    elif option_id == "allow_always":
                        # Store in session cache and persistent store
                        async with self._lock:
                            self._session_cache[permission_key] = PermissionDecision.ALLOW_ALWAYS
                        await self._store.store(
                            server_name, tool_name, PermissionDecision.ALLOW_ALWAYS
                        )
                        logger.info(
                            f"Permission allow_always for {permission_key}",
                            name="acp_tool_permission_allow_always",
                        )
                        return PermissionResult.allow_always()

                    elif option_id == "reject_once":
                        logger.info(
                            f"Permission reject_once for {permission_key}",
                            name="acp_tool_permission_reject_once",
                        )
                        return PermissionResult.reject_once()

                    elif option_id == "reject_always":
                        # Store in session cache and persistent store
                        async with self._lock:
                            self._session_cache[permission_key] = PermissionDecision.REJECT_ALWAYS
                        await self._store.store(
                            server_name, tool_name, PermissionDecision.REJECT_ALWAYS
                        )
                        logger.info(
                            f"Permission reject_always for {permission_key}",
                            name="acp_tool_permission_reject_always",
                        )
                        return PermissionResult.reject_always()

            # Unknown response format - FAIL-SAFE to DENY
            logger.warning(
                f"Unknown permission response for {permission_key}, defaulting to deny",
                name="acp_tool_permission_unknown_response",
            )
            return PermissionResult.denied()

        except Exception as e:
            # FAIL-SAFE: Default to DENY on any error
            logger.error(
                f"Error processing permission response: {e}",
                name="acp_tool_permission_process_error",
                exc_info=True,
            )
            return PermissionResult.denied()

    async def clear_session_cache(self) -> None:
        """Clear the session-specific permission cache."""
        async with self._lock:
            count = len(self._session_cache)
            self._session_cache.clear()
            logger.debug(
                f"Cleared {count} session-cached permissions",
                name="acp_tool_permission_cache_cleared",
            )

    async def clear_persisted_permissions(
        self, server_name: str | None = None, tool_name: str | None = None
    ) -> None:
        """
        Clear persisted permissions.

        Args:
            server_name: If provided with tool_name, clears specific permission.
                        If provided alone, clears all permissions for that server.
            tool_name: Must be provided with server_name.
        """
        await self._store.clear(server_name, tool_name)

        # Also clear from session cache
        async with self._lock:
            if server_name and tool_name:
                key = self._get_permission_key(tool_name, server_name)
                self._session_cache.pop(key, None)
            elif server_name:
                keys_to_remove = [
                    k for k in self._session_cache if k.startswith(f"{server_name}/")
                ]
                for k in keys_to_remove:
                    del self._session_cache[k]
            else:
                self._session_cache.clear()


def create_acp_permission_handler(
    permission_manager: ACPToolPermissionManager,
) -> ToolPermissionHandler:
    """
    Create a tool permission handler for ACP integration.

    This creates a handler that implements the ToolPermissionHandler protocol.

    Args:
        permission_manager: The ACPToolPermissionManager instance

    Returns:
        A permission handler that delegates to the manager
    """

    class _ACPPermissionHandler:
        """Handler that delegates to ACPToolPermissionManager."""

        async def check_permission(
            self,
            tool_name: str,
            server_name: str,
            arguments: dict[str, Any] | None = None,
            tool_call_id: str | None = None,
        ) -> PermissionResult:
            return await permission_manager.check_permission(
                tool_name, server_name, arguments, tool_call_id
            )

    return _ACPPermissionHandler()
