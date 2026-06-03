"""
ACP Tool Call Permissions

Provides a permission handler that requests tool execution permission from the ACP client.
This follows the same pattern as elicitation handlers but for tool execution authorization.

Key features:
- Requests user permission before tool execution via ACP session/request_permission
- Supports persistent permissions (allow_always, reject_always) stored in the fast-agent environment
- Fail-safe: defaults to DENY on any error
- In-memory caching for remembered permissions within a session
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, runtime_checkable

from acp.schema import (
    AllowedOutcome,
    DeniedOutcome,
    PermissionOption,
    RequestPermissionResponse,
    ToolCallProgress,
    ToolCallUpdate,
)

from fast_agent.acp.permission_store import (
    PermissionDecision,
    PermissionEntry,
    PermissionResult,
    PermissionStore,
)
from fast_agent.acp.tool_kind_inference import infer_tool_kind
from fast_agent.acp.tool_titles import build_tool_title
from fast_agent.core.logging.logger import get_logger
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)


def _is_acp_tool_call_id(value: str | None) -> TypeGuard[str]:
    if value is None or len(value) != 32:
        return False
    return all(ch in "0123456789abcdef" for ch in strip_casefold(value))


@dataclass
class ToolPermissionRequest:
    """Request for tool execution permission."""

    tool_name: str
    server_name: str
    arguments: dict[str, Any] | None
    tool_call_id: str | None = None


# Type for permission handler callbacks
ToolPermissionHandlerT = Callable[[ToolPermissionRequest], Awaitable[PermissionResult]]


def _permission_options() -> list[PermissionOption]:
    return [
        PermissionOption(
            option_id="allow_once",
            kind="allow_once",
            name="Allow Once",
        ),
        PermissionOption(
            option_id="allow_always",
            kind="allow_always",
            name="Always Allow This Tool",
        ),
        PermissionOption(
            option_id="reject_once",
            kind="reject_once",
            name="Reject Once",
        ),
        PermissionOption(
            option_id="reject_always",
            kind="reject_always",
            name="Never Allow This Tool",
        ),
    ]


@runtime_checkable
class ToolPermissionChecker(Protocol):
    """
    Protocol for checking tool execution permissions.

    This allows permission checking to be injected into the MCP aggregator
    without tight coupling to ACP.
    """

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


class ACPToolPermissionManager:
    """
    Manages tool execution permission requests via ACP.

    This class provides a handler that can be used to request permission
    from the ACP client before executing tools. It implements the
    ToolPermissionChecker protocol for integration with the MCP aggregator.

    Features:
    - Checks persistent permissions from PermissionStore first
    - Falls back to ACP client permission request
    - Caches session-level permissions in memory
    - Fail-safe: defaults to DENY on any error
    """

    def __init__(
        self,
        connection: "AgentSideConnection",
        session_id: str,
        store: PermissionStore | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        """
        Initialize the permission manager.

        Args:
            connection: The ACP connection to send permission requests on
            session_id: The ACP session ID
            store: Optional PermissionStore for persistence (created if not provided)
            cwd: Working directory for the store (only used if store not provided)
        """
        self._connection = connection
        self._session_id = session_id
        self._store = store or PermissionStore(cwd=cwd)
        # In-memory cache for session-level permissions (cleared on session end)
        self._session_cache: dict[PermissionEntry, bool] = {}
        self._lock = asyncio.Lock()

    def _get_permission_key(self, tool_name: str, server_name: str) -> PermissionEntry:
        """Get a unique key for remembering permissions."""
        return PermissionEntry(server_name=server_name, tool_name=tool_name)

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> PermissionResult:
        """
        Check if tool execution is permitted.

        Order of checks:
        1. Session-level cache (for allow_once/reject_once remembered within session)
        2. Persistent store (for allow_always/reject_always)
        3. ACP client permission request

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
            # 1. Check session-level cache
            async with self._lock:
                if permission_key in self._session_cache:
                    allowed = self._session_cache[permission_key]
                    logger.debug(
                        f"Using session-cached permission for {permission_key.display_key}: {allowed}",
                        name="acp_tool_permission_session_cache",
                    )
                    return PermissionResult(allowed=allowed, remember=True)

            # 2. Check persistent store
            stored_decision = await self._store.get(server_name, tool_name)
            if stored_decision is not None:
                allowed = stored_decision == PermissionDecision.ALLOW_ALWAYS
                logger.debug(
                    f"Using stored permission for {permission_key.display_key}: {stored_decision.value}",
                    name="acp_tool_permission_stored",
                )
                # Cache in session for faster subsequent lookups
                async with self._lock:
                    self._session_cache[permission_key] = allowed
                return PermissionResult(allowed=allowed, remember=True)

            # 3. Request permission from ACP client
            return await self._request_permission_from_client(
                tool_name=tool_name,
                server_name=server_name,
                arguments=arguments,
                tool_call_id=tool_call_id,
                permission_key=permission_key,
            )

        except Exception as e:
            logger.error(
                f"Error checking tool permission: {e}",
                name="acp_tool_permission_error",
                exc_info=True,
            )
            # FAIL-SAFE: Default to DENY on any error
            return PermissionResult(allowed=False, remember=False)

    async def _request_permission_from_client(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None,
        tool_call_id: str | None,
        permission_key: PermissionEntry,
    ) -> PermissionResult:
        """
        Request permission from the ACP client.

        Args:
            tool_name: Name of the tool
            server_name: Name of the server
            arguments: Tool arguments
            tool_call_id: Tool call ID
            permission_key: Cache key for this tool

        Returns:
            PermissionResult from the client
        """
        title = build_tool_title(tool_name=tool_name, server_name=server_name)

        # If we have an ACP toolCallId already (e.g. from streaming tool notifications),
        # proactively update the tool call title so the client UI matches the permission prompt.
        if _is_acp_tool_call_id(tool_call_id):
            with suppress(Exception):
                await self._connection.session_update(
                    session_id=self._session_id,
                    update=ToolCallProgress(
                        tool_call_id=tool_call_id,
                        title=title,
                        status="pending",
                        session_update="tool_call_update",
                    ),
                )

        # Create ToolCallUpdate object per ACP spec with raw_input for full argument visibility
        tool_kind = infer_tool_kind(tool_name, arguments)

        tool_call = ToolCallUpdate(
            tool_call_id=tool_call_id or "pending",
            title=title,
            kind=tool_kind,
            status="pending",
            raw_input=arguments,  # Include full arguments so client can display them
        )

        options = _permission_options()

        try:
            logger.info(
                f"Requesting permission for {permission_key.display_key}",
                name="acp_tool_permission_request",
                tool_name=tool_name,
                server_name=server_name,
            )

            # Send permission request to client using flattened parameters
            response = await self._connection.request_permission(
                options=options,
                session_id=self._session_id,
                tool_call=tool_call,
            )

            # Handle response
            return await self._handle_permission_response(
                response, permission_key, server_name, tool_name
            )

        except Exception as e:
            logger.error(
                f"Error requesting tool permission from client: {e}",
                name="acp_tool_permission_request_error",
                exc_info=True,
                tool_name=tool_name,
                server_name=server_name,
                session_id=self._session_id,
                tool_call_id=tool_call_id,
            )
            # FAIL-SAFE: Default to DENY on any error
            return PermissionResult(allowed=False, remember=False)

    async def _handle_permission_response(
        self,
        response: RequestPermissionResponse,
        permission_key: PermissionEntry,
        server_name: str,
        tool_name: str,
    ) -> PermissionResult:
        """
        Handle the permission response from the client.

        Args:
            response: The response from requestPermission
            permission_key: Cache key
            server_name: Server name
            tool_name: Tool name

        Returns:
            PermissionResult based on client response
        """
        outcome = response.outcome
        if isinstance(outcome, DeniedOutcome):
            logger.info(
                f"Permission request cancelled for {permission_key.display_key}",
                name="acp_tool_permission_cancelled",
            )
            return PermissionResult.cancelled()

        if isinstance(outcome, AllowedOutcome):
            option_id = outcome.option_id
            option_handlers = {
                "allow_once": self._handle_allow_once,
                "allow_always": self._handle_allow_always,
                "reject_once": self._handle_reject_once,
                "reject_always": self._handle_reject_always,
            }
            option_handler = option_handlers.get(option_id)
            if option_handler is not None:
                return await option_handler(permission_key, server_name, tool_name)

            logger.warning(
                f"Unknown permission option '{option_id}' for {permission_key.display_key}, defaulting to reject",
                name="acp_tool_permission_unknown_option",
            )
            return PermissionResult(allowed=False, remember=False)

        logger.warning(
            f"Unknown permission response format for {permission_key.display_key}, defaulting to reject",
            name="acp_tool_permission_unknown_format",
        )
        return PermissionResult(allowed=False, remember=False)

    async def _handle_allow_once(
        self,
        permission_key: PermissionEntry,
        server_name: str,
        tool_name: str,
    ) -> PermissionResult:
        del server_name, tool_name
        logger.info(
            f"Permission granted once for {permission_key.display_key}",
            name="acp_tool_permission_allow_once",
        )
        return PermissionResult.allow_once()

    async def _handle_allow_always(
        self,
        permission_key: PermissionEntry,
        server_name: str,
        tool_name: str,
    ) -> PermissionResult:
        await self._store.set(server_name, tool_name, PermissionDecision.ALLOW_ALWAYS)
        async with self._lock:
            self._session_cache[permission_key] = True
        logger.info(
            f"Permission granted always for {permission_key.display_key}",
            name="acp_tool_permission_allow_always",
        )
        return PermissionResult.allow_always()

    async def _handle_reject_once(
        self,
        permission_key: PermissionEntry,
        server_name: str,
        tool_name: str,
    ) -> PermissionResult:
        del server_name, tool_name
        logger.info(
            f"Permission rejected once for {permission_key.display_key}",
            name="acp_tool_permission_reject_once",
        )
        return PermissionResult.reject_once()

    async def _handle_reject_always(
        self,
        permission_key: PermissionEntry,
        server_name: str,
        tool_name: str,
    ) -> PermissionResult:
        await self._store.set(server_name, tool_name, PermissionDecision.REJECT_ALWAYS)
        async with self._lock:
            self._session_cache[permission_key] = False
        logger.info(
            f"Permission rejected always for {permission_key.display_key}",
            name="acp_tool_permission_reject_always",
        )
        return PermissionResult.reject_always()

    async def clear_session_cache(self) -> None:
        """Clear the session-level permission cache."""
        async with self._lock:
            self._session_cache.clear()
            logger.debug(
                "Cleared session permission cache",
                name="acp_tool_permission_cache_cleared",
            )


class NoOpToolPermissionChecker:
    """
    No-op permission checker that always allows tool execution.

    Used when --no-permissions flag is set or when not running in ACP mode.
    """

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> PermissionResult:
        """Always allows tool execution."""
        return PermissionResult.allow_once()


def create_acp_permission_handler(
    permission_manager: ACPToolPermissionManager,
) -> ToolPermissionHandlerT:
    """
    Create a tool permission handler for ACP integration.

    This creates a handler that can be injected into the tool execution
    pipeline to request permission before executing tools.

    Args:
        permission_manager: The ACPToolPermissionManager instance

    Returns:
        A permission handler function
    """

    async def handler(request: ToolPermissionRequest) -> PermissionResult:
        """Handle tool permission request."""
        return await permission_manager.check_permission(
            tool_name=request.tool_name,
            server_name=request.server_name,
            arguments=request.arguments,
            tool_call_id=request.tool_call_id,
        )

    return handler
