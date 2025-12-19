"""
ACP Tool Call Permissions

Provides a permission handler that requests tool execution permission from the ACP client.
This follows the same pattern as elicitation handlers but for tool execution authorization.

Key features:
- Requests user permission before tool execution via ACP session/request_permission
- Supports persistent permissions (allow_always, reject_always) stored in .fast-agent/auths.md
- Fail-safe: defaults to DENY on any error
- In-memory caching for remembered permissions within a session
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol, runtime_checkable

from acp.schema import (
    PermissionOption,
    ToolCallProgress,
    ToolCallUpdate,
    ToolKind,
)

from fast_agent.acp.permission_store import PermissionDecision, PermissionResult, PermissionStore
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


# Type for permission handler callbacks
ToolPermissionHandlerT = Callable[[ToolPermissionRequest], Awaitable[PermissionResult]]


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


def _infer_tool_kind(tool_name: str, arguments: dict[str, Any] | None = None) -> ToolKind:
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
    if any(word in name_lower for word in ["read", "get", "fetch", "list", "show", "cat"]):
        return "read"
    elif any(
        word in name_lower for word in ["write", "edit", "update", "modify", "patch", "create"]
    ):
        return "edit"
    elif any(word in name_lower for word in ["delete", "remove", "clear", "clean", "rm"]):
        return "delete"
    elif any(word in name_lower for word in ["move", "rename", "mv", "copy", "cp"]):
        return "move"
    elif any(word in name_lower for word in ["search", "find", "query", "grep", "locate"]):
        return "search"
    elif any(word in name_lower for word in ["execute", "run", "exec", "command", "bash", "shell"]):
        return "execute"
    elif any(word in name_lower for word in ["think", "plan", "reason", "analyze"]):
        return "think"
    elif any(word in name_lower for word in ["fetch", "download", "http", "request", "curl"]):
        return "fetch"

    return "other"


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
        self._session_cache: dict[str, bool] = {}
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
                        f"Using session-cached permission for {permission_key}: {allowed}",
                        name="acp_tool_permission_session_cache",
                    )
                    return PermissionResult(allowed=allowed, remember=True)

            # 2. Check persistent store
            stored_decision = await self._store.get(server_name, tool_name)
            if stored_decision is not None:
                allowed = stored_decision == PermissionDecision.ALLOW_ALWAYS
                logger.debug(
                    f"Using stored permission for {permission_key}: {stored_decision.value}",
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
        permission_key: str,
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
        # Create descriptive title with argument summary
        title = f"{server_name}/{tool_name}"
        if arguments:
            # Include trimmed arg list info in title for user context
            arg_str = ", ".join(f"{k}={v}" for k, v in list(arguments.items()))
            if len(arg_str) > 50:
                arg_str = arg_str[:47] + "..."
            title = f"{title}({arg_str})"

        # If we have an ACP toolCallId already (e.g. from streaming tool notifications),
        # proactively update the tool call title so the client UI matches the permission prompt.
        if tool_call_id and len(tool_call_id) == 32:
            lowered = tool_call_id.lower()
            if all(ch in "0123456789abcdef" for ch in lowered):
                try:
                    await self._connection.session_update(
                        session_id=self._session_id,
                        update=ToolCallProgress(
                            tool_call_id=tool_call_id,
                            title=title,
                            status="pending",
                            session_update="tool_call_update",
                        ),
                    )
                except Exception:
                    pass

        # Create ToolCallUpdate object per ACP spec with raw_input for full argument visibility
        tool_kind = _infer_tool_kind(tool_name, arguments)
        tool_call = ToolCallUpdate(
            tool_call_id=tool_call_id or "pending",
            title=title,
            kind=tool_kind,
            status="pending",
            raw_input=arguments,  # Include full arguments so client can display them
        )

        # Create permission request with options
        options = [
            PermissionOption(
                option_id="allow_once",
                kind="allow_once",
                name="Allow Once",
            ),
            PermissionOption(
                option_id="allow_always",
                kind="allow_always",
                name="Always Allow",
            ),
            PermissionOption(
                option_id="reject_once",
                kind="reject_once",
                name="Reject Once",
            ),
            PermissionOption(
                option_id="reject_always",
                kind="reject_always",
                name="Never Allow",
            ),
        ]

        try:
            logger.info(
                f"Requesting permission for {permission_key}",
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
            )
            # FAIL-SAFE: Default to DENY on any error
            return PermissionResult(allowed=False, remember=False)

    async def _handle_permission_response(
        self,
        response: Any,
        permission_key: str,
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
        if not hasattr(outcome, "outcome"):
            logger.warning(
                f"Unknown permission response format for {permission_key}, defaulting to reject",
                name="acp_tool_permission_unknown_format",
            )
            return PermissionResult(allowed=False, remember=False)

        outcome_type = outcome.outcome

        if outcome_type == "cancelled":
            logger.info(
                f"Permission request cancelled for {permission_key}",
                name="acp_tool_permission_cancelled",
            )
            return PermissionResult.cancelled()

        if outcome_type == "selected":
            option_id = getattr(outcome, "optionId", None)

            if option_id == "allow_once":
                logger.info(
                    f"Permission granted once for {permission_key}",
                    name="acp_tool_permission_allow_once",
                )
                return PermissionResult.allow_once()

            elif option_id == "allow_always":
                # Store in persistent store
                await self._store.set(server_name, tool_name, PermissionDecision.ALLOW_ALWAYS)
                # Also cache in session
                async with self._lock:
                    self._session_cache[permission_key] = True
                logger.info(
                    f"Permission granted always for {permission_key}",
                    name="acp_tool_permission_allow_always",
                )
                return PermissionResult.allow_always()

            elif option_id == "reject_once":
                logger.info(
                    f"Permission rejected once for {permission_key}",
                    name="acp_tool_permission_reject_once",
                )
                return PermissionResult.reject_once()

            elif option_id == "reject_always":
                # Store in persistent store
                await self._store.set(server_name, tool_name, PermissionDecision.REJECT_ALWAYS)
                # Also cache in session
                async with self._lock:
                    self._session_cache[permission_key] = False
                logger.info(
                    f"Permission rejected always for {permission_key}",
                    name="acp_tool_permission_reject_always",
                )
                return PermissionResult.reject_always()

        # Unknown response type - FAIL-SAFE: DENY
        logger.warning(
            f"Unknown permission option for {permission_key}, defaulting to reject",
            name="acp_tool_permission_unknown_option",
        )
        return PermissionResult(allowed=False, remember=False)

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
