"""
ACP Tool Call Permissions

Provides a permission handler that requests tool execution permission from the ACP client.
This follows the same pattern as elicitation handlers but for tool execution authorization.

Persistent permissions are stored in `.fast-agent/auths.md` in the session's working directory.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from acp.schema import PermissionOption, RequestPermissionRequest, ToolCall, ToolKind

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)

# Permission kinds that should be persisted to disk
PERSISTENT_PERMISSION_KINDS = {"allow_always", "reject_always"}


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
ToolPermissionHandlerT = Callable[[ToolPermissionRequest], Awaitable[ToolPermissionResponse]]


class PermissionFileStore:
    """
    Persistent storage for tool permissions in `.fast-agent/auths.md`.

    Only stores `allow_always` and `reject_always` decisions. The file is only
    created when a persistent permission is first granted.

    File format (simple markdown):
    ```
    # Fast-Agent Tool Permissions

    ## Allowed Tools
    - server_name/tool_name

    ## Rejected Tools
    - server_name/tool_name
    ```
    """

    def __init__(self, base_dir: str | Path) -> None:
        """
        Initialize the permission file store.

        Args:
            base_dir: Base directory (session cwd) where .fast-agent/auths.md will be stored
        """
        self._base_dir = Path(base_dir)
        self._file_path = self._base_dir / ".fast-agent" / "auths.md"
        self._allowed: set[str] = set()  # permission keys that are always allowed
        self._rejected: set[str] = set()  # permission keys that are always rejected
        self._loaded = False

    def _get_permission_key(self, tool_name: str, server_name: str) -> str:
        """Get a unique key for a tool permission."""
        return f"{server_name}/{tool_name}"

    def _ensure_loaded(self) -> None:
        """Load permissions from file if not already loaded."""
        if self._loaded:
            return

        self._loaded = True

        if not self._file_path.exists():
            return

        try:
            content = self._file_path.read_text(encoding="utf-8")
            current_section = None

            for line in content.splitlines():
                line = line.strip()

                if line == "## Allowed Tools":
                    current_section = "allowed"
                elif line == "## Rejected Tools":
                    current_section = "rejected"
                elif line.startswith("- ") and current_section:
                    permission_key = line[2:].strip()
                    if permission_key:
                        if current_section == "allowed":
                            self._allowed.add(permission_key)
                        elif current_section == "rejected":
                            self._rejected.add(permission_key)

            logger.info(
                f"Loaded {len(self._allowed)} allowed and {len(self._rejected)} rejected permissions",
                name="permission_file_loaded",
                file_path=str(self._file_path),
            )
        except Exception as e:
            logger.warning(
                f"Failed to load permissions file: {e}",
                name="permission_file_load_error",
            )

    def _save(self) -> None:
        """Save permissions to file."""
        # Only create file if there are permissions to save
        if not self._allowed and not self._rejected:
            return

        try:
            # Ensure directory exists
            self._file_path.parent.mkdir(parents=True, exist_ok=True)

            lines = ["# Fast-Agent Tool Permissions", ""]

            if self._allowed:
                lines.append("## Allowed Tools")
                for key in sorted(self._allowed):
                    lines.append(f"- {key}")
                lines.append("")

            if self._rejected:
                lines.append("## Rejected Tools")
                for key in sorted(self._rejected):
                    lines.append(f"- {key}")
                lines.append("")

            self._file_path.write_text("\n".join(lines), encoding="utf-8")

            logger.info(
                f"Saved permissions to {self._file_path}",
                name="permission_file_saved",
            )
        except Exception as e:
            logger.error(
                f"Failed to save permissions file: {e}",
                name="permission_file_save_error",
            )

    def get_permission(self, tool_name: str, server_name: str) -> bool | None:
        """
        Get the stored permission for a tool.

        Args:
            tool_name: Name of the tool
            server_name: Name of the server

        Returns:
            True if allowed, False if rejected, None if no stored permission
        """
        self._ensure_loaded()
        key = self._get_permission_key(tool_name, server_name)

        if key in self._allowed:
            return True
        if key in self._rejected:
            return False
        return None

    def set_permission(self, tool_name: str, server_name: str, allowed: bool) -> None:
        """
        Store a persistent permission for a tool.

        Args:
            tool_name: Name of the tool
            server_name: Name of the server
            allowed: Whether the tool is allowed
        """
        self._ensure_loaded()
        key = self._get_permission_key(tool_name, server_name)

        # Remove from opposite set if present
        if allowed:
            self._rejected.discard(key)
            self._allowed.add(key)
        else:
            self._allowed.discard(key)
            self._rejected.add(key)

        self._save()

    def clear_permission(self, tool_name: str, server_name: str) -> None:
        """
        Clear a stored permission for a tool.

        Args:
            tool_name: Name of the tool
            server_name: Name of the server
        """
        self._ensure_loaded()
        key = self._get_permission_key(tool_name, server_name)
        self._allowed.discard(key)
        self._rejected.discard(key)
        self._save()

    def clear_all(self) -> None:
        """Clear all stored permissions."""
        self._allowed.clear()
        self._rejected.clear()

        # Remove the file if it exists
        if self._file_path.exists():
            try:
                self._file_path.unlink()
                logger.info(
                    f"Removed permissions file {self._file_path}",
                    name="permission_file_removed",
                )
            except Exception as e:
                logger.error(
                    f"Failed to remove permissions file: {e}",
                    name="permission_file_remove_error",
                )


def infer_tool_kind(tool_name: str) -> ToolKind:
    """
    Infer the tool kind from the tool name for UI hints.

    Args:
        tool_name: Name of the tool

    Returns:
        The inferred ToolKind
    """
    name_lower = tool_name.lower()

    if any(word in name_lower for word in ["read", "get", "list", "show", "view", "cat"]):
        return "read"
    if any(word in name_lower for word in ["write", "edit", "update", "modify", "patch", "set"]):
        return "edit"
    if any(word in name_lower for word in ["delete", "remove", "rm"]):
        return "delete"
    if any(word in name_lower for word in ["move", "rename", "mv", "copy", "cp"]):
        return "move"
    if any(word in name_lower for word in ["search", "find", "grep", "query"]):
        return "search"
    if any(word in name_lower for word in ["execute", "run", "exec", "shell", "bash", "command"]):
        return "execute"
    if any(word in name_lower for word in ["think", "plan", "reason", "analyze"]):
        return "think"
    if any(word in name_lower for word in ["fetch", "download", "http", "request", "url"]):
        return "fetch"

    return "other"


class ACPToolPermissionManager:
    """
    Manages tool execution permission requests via ACP.

    This class provides a handler that can be used to request permission
    from the ACP client before executing tools. Supports both in-memory
    session permissions and persistent file-based permissions.
    """

    def __init__(
        self,
        connection: "AgentSideConnection",
        session_id: str,
        *,
        file_store: PermissionFileStore | None = None,
        permissions_enabled: bool = True,
    ) -> None:
        """
        Initialize the permission manager.

        Args:
            connection: The ACP connection to send permission requests on
            session_id: The ACP session ID
            file_store: Optional file store for persistent permissions
            permissions_enabled: Whether permission requests are enabled (default: True)
        """
        self._connection = connection
        self._session_id = session_id
        self._file_store = file_store
        self._permissions_enabled = permissions_enabled

        # In-memory session permissions (for allow_once/reject_once tracking within session)
        self._session_permissions: dict[str, bool] = {}
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
    ) -> ToolPermissionResponse:
        """
        Check permission to execute a tool, requesting from user if needed.

        Args:
            tool_name: Name of the tool to execute
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments (for display to user)
            tool_call_id: Optional tool call ID for tracking

        Returns:
            ToolPermissionResponse indicating whether execution is allowed
        """
        # If permissions are disabled, always allow
        if not self._permissions_enabled:
            return ToolPermissionResponse(allowed=True, remember=False)

        permission_key = self._get_permission_key(tool_name, server_name)

        # Check file store first (persistent permissions)
        if self._file_store:
            stored = self._file_store.get_permission(tool_name, server_name)
            if stored is not None:
                logger.debug(
                    f"Using stored permission for {permission_key}: {stored}",
                    name="acp_tool_permission_stored",
                )
                return ToolPermissionResponse(allowed=stored, remember=True)

        # Check session permissions (in-memory)
        async with self._lock:
            if permission_key in self._session_permissions:
                allowed = self._session_permissions[permission_key]
                logger.debug(
                    f"Using session permission for {permission_key}: {allowed}",
                    name="acp_tool_permission_session",
                )
                return ToolPermissionResponse(allowed=allowed, remember=True)

        # Need to request permission from user
        return await self._request_permission(
            tool_name=tool_name,
            server_name=server_name,
            arguments=arguments,
            tool_call_id=tool_call_id,
        )

    async def _request_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> ToolPermissionResponse:
        """
        Request permission from the ACP client.

        Args:
            tool_name: Name of the tool to execute
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments
            tool_call_id: Optional tool call ID for tracking

        Returns:
            ToolPermissionResponse indicating whether execution is allowed
        """
        permission_key = self._get_permission_key(tool_name, server_name)

        # Generate tool call ID if not provided
        if not tool_call_id:
            import uuid
            tool_call_id = f"perm_{uuid.uuid4().hex[:8]}"

        # Build a human-readable title
        title = f"Execute {server_name}/{tool_name}"

        # Infer tool kind for UI hints
        kind = infer_tool_kind(tool_name)

        # Build the ToolCall object per ACP spec
        tool_call = ToolCall(
            toolCallId=tool_call_id,
            title=title,
            kind=kind,
            status="pending",
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

        # Build the request
        request = RequestPermissionRequest(
            sessionId=self._session_id,
            toolCall=tool_call,
            options=options,
        )

        try:
            logger.info(
                f"Requesting permission for {permission_key}",
                name="acp_tool_permission_request",
                tool_name=tool_name,
                server_name=server_name,
                tool_call_id=tool_call_id,
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
                        logger.info(
                            f"Permission allowed once for {permission_key}",
                            name="acp_tool_permission_allow_once",
                        )
                        return ToolPermissionResponse(allowed=True, remember=False)

                    elif option_id == "allow_always":
                        # Store in file for persistence
                        if self._file_store:
                            self._file_store.set_permission(tool_name, server_name, True)
                        # Also store in session for quick lookup
                        async with self._lock:
                            self._session_permissions[permission_key] = True
                        logger.info(
                            f"Permission always allowed for {permission_key}",
                            name="acp_tool_permission_allow_always",
                        )
                        return ToolPermissionResponse(allowed=True, remember=True)

                    elif option_id == "reject_once":
                        logger.info(
                            f"Permission rejected once for {permission_key}",
                            name="acp_tool_permission_reject_once",
                        )
                        return ToolPermissionResponse(allowed=False, remember=False)

                    elif option_id == "reject_always":
                        # Store in file for persistence
                        if self._file_store:
                            self._file_store.set_permission(tool_name, server_name, False)
                        # Also store in session for quick lookup
                        async with self._lock:
                            self._session_permissions[permission_key] = False
                        logger.info(
                            f"Permission always rejected for {permission_key}",
                            name="acp_tool_permission_reject_always",
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
            # This is configurable behavior - some may prefer to reject on error
            return ToolPermissionResponse(allowed=True, remember=False)

    async def clear_session_permissions(
        self,
        tool_name: str | None = None,
        server_name: str | None = None,
    ) -> None:
        """
        Clear in-memory session permissions.

        Args:
            tool_name: Optional tool name to clear (clears all if None)
            server_name: Optional server name to clear (clears all if None)
        """
        async with self._lock:
            if tool_name and server_name:
                permission_key = self._get_permission_key(tool_name, server_name)
                self._session_permissions.pop(permission_key, None)
                logger.info(
                    f"Cleared session permission for {permission_key}",
                    name="acp_tool_permission_cleared",
                )
            else:
                self._session_permissions.clear()
                logger.info(
                    "Cleared all session permissions",
                    name="acp_tool_permissions_cleared_all",
                )


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

    async def handler(request: ToolPermissionRequest) -> ToolPermissionResponse:
        """Handle tool permission request."""
        return await permission_manager.check_permission(
            tool_name=request.tool_name,
            server_name=request.server_name,
            arguments=request.arguments,
            tool_call_id=request.tool_call_id,
        )

    return handler
