"""
ACP Tool Call Permissions

Provides a permission handler that requests tool execution permission from the ACP client.
This follows the same pattern as elicitation handlers but for tool execution authorization.

Persistent permissions are stored in `.fast-agent/auths.md` in the session's working directory.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from acp.schema import PermissionOption, RequestPermissionRequest, ToolCall

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)

# File path for persistent permissions
AUTHS_DIR = ".fast-agent"
AUTHS_FILE = "auths.md"


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
    error_message: str | None = None  # Message to return when denied/error


# Type for permission handler callbacks
ToolPermissionHandlerT = Callable[[ToolPermissionRequest], asyncio.Future[ToolPermissionResponse]]


def _infer_tool_kind(tool_name: str) -> str:
    """Infer the ACP ToolKind from the tool name."""
    name_lower = tool_name.lower()
    if any(kw in name_lower for kw in ["read", "get", "list", "show", "cat"]):
        return "read"
    elif any(kw in name_lower for kw in ["write", "edit", "update", "set", "modify"]):
        return "edit"
    elif any(kw in name_lower for kw in ["delete", "remove", "rm"]):
        return "delete"
    elif any(kw in name_lower for kw in ["move", "rename", "mv"]):
        return "move"
    elif any(kw in name_lower for kw in ["search", "find", "grep"]):
        return "search"
    elif any(kw in name_lower for kw in ["exec", "run", "shell", "bash", "command"]):
        return "execute"
    elif any(kw in name_lower for kw in ["fetch", "download", "http", "request"]):
        return "fetch"
    elif any(kw in name_lower for kw in ["think", "plan", "reason"]):
        return "think"
    return "other"


class PermissionFileManager:
    """
    Manages persistent permission storage in `.fast-agent/auths.md`.

    File format:
    ```markdown
    # Fast-Agent Tool Authorizations

    ## always_allow
    - server_name/tool_name
    - other_server/other_tool

    ## always_reject
    - dangerous_server/dangerous_tool
    ```
    """

    def __init__(self, cwd: str | Path) -> None:
        """Initialize with the working directory for this session."""
        self._cwd = Path(cwd)
        self._file_path = self._cwd / AUTHS_DIR / AUTHS_FILE

    def load(self) -> dict[str, bool]:
        """
        Load persistent permissions from file.

        Returns:
            Dict mapping permission_key to allowed (True/False)
        """
        permissions: dict[str, bool] = {}

        if not self._file_path.exists():
            return permissions

        try:
            content = self._file_path.read_text()
            current_section: str | None = None

            for line in content.splitlines():
                line = line.strip()
                if line.startswith("## always_allow"):
                    current_section = "allow"
                elif line.startswith("## always_reject"):
                    current_section = "reject"
                elif line.startswith("- ") and current_section:
                    key = line[2:].strip()
                    if key:
                        permissions[key] = current_section == "allow"

            logger.info(
                f"Loaded {len(permissions)} persistent permissions from {self._file_path}",
                name="acp_permission_file_loaded",
            )
        except Exception as e:
            logger.warning(
                f"Failed to load permissions file: {e}",
                name="acp_permission_file_error",
            )

        return permissions

    def save(self, permissions: dict[str, bool]) -> None:
        """
        Save persistent permissions to file.

        Only saves allow_always and reject_always permissions.

        Args:
            permissions: Dict mapping permission_key to allowed (True/False)
        """
        if not permissions:
            return

        # Separate into allow and reject lists
        allow_list = sorted(k for k, v in permissions.items() if v)
        reject_list = sorted(k for k, v in permissions.items() if not v)

        # Build file content
        lines = ["# Fast-Agent Tool Authorizations", ""]

        if allow_list:
            lines.append("## always_allow")
            for key in allow_list:
                lines.append(f"- {key}")
            lines.append("")

        if reject_list:
            lines.append("## always_reject")
            for key in reject_list:
                lines.append(f"- {key}")
            lines.append("")

        # Ensure directory exists
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_path.write_text("\n".join(lines))
            logger.info(
                f"Saved {len(permissions)} permissions to {self._file_path}",
                name="acp_permission_file_saved",
            )
        except Exception as e:
            logger.error(
                f"Failed to save permissions file: {e}",
                name="acp_permission_file_save_error",
            )


@dataclass
class ACPToolPermissionManager:
    """
    Manages tool execution permission requests via ACP.

    This class provides a handler that can be used to request permission
    from the ACP client before executing tools.

    Permissions can be persisted to `.fast-agent/auths.md` in the session's
    working directory when users select "Always Allow" or "Never Allow".
    """

    _connection: "AgentSideConnection"
    _session_id: str
    _cwd: str | Path | None = None
    _enabled: bool = True
    _remembered_permissions: dict[str, bool] = field(default_factory=dict)
    _file_manager: PermissionFileManager | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        """Load persistent permissions if cwd is provided."""
        if self._cwd:
            self._file_manager = PermissionFileManager(self._cwd)
            self._remembered_permissions = self._file_manager.load()

    def _get_permission_key(self, tool_name: str, server_name: str) -> str:
        """Get a unique key for remembering permissions."""
        return f"{server_name}/{tool_name}"

    def _save_permissions(self) -> None:
        """Save current permissions to file if file manager is available."""
        if self._file_manager:
            self._file_manager.save(self._remembered_permissions)

    async def request_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> ToolPermissionResponse:
        """
        Request permission to execute a tool.

        Args:
            tool_name: Name of the tool to execute
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments
            tool_call_id: Optional tool call ID for tracking

        Returns:
            ToolPermissionResponse indicating whether execution is allowed
        """
        # If permissions are disabled, allow all
        if not self._enabled:
            return ToolPermissionResponse(allowed=True, remember=False)

        permission_key = self._get_permission_key(tool_name, server_name)

        # Check remembered permissions
        async with self._lock:
            if permission_key in self._remembered_permissions:
                allowed = self._remembered_permissions[permission_key]
                logger.debug(
                    f"Using remembered permission for {permission_key}: {allowed}",
                    name="acp_tool_permission_remembered",
                )
                if allowed:
                    return ToolPermissionResponse(allowed=True, remember=True)
                else:
                    return ToolPermissionResponse(
                        allowed=False,
                        remember=True,
                        error_message="The User declined this operation",
                    )

        # Build tool call object for ACP request
        tool_call = ToolCall(
            toolCallId=tool_call_id or f"perm_{permission_key}",
            title=f"{server_name}/{tool_name}",
            kind=_infer_tool_kind(tool_name),
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
            toolCall=tool_call,
            options=options,
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
                    return ToolPermissionResponse(
                        allowed=False,
                        remember=False,
                        cancelled=True,
                        error_message="The User declined this operation",
                    )

                elif outcome_type == "selected":
                    option_id = getattr(outcome, "optionId", None)

                    if option_id == "allow_once":
                        return ToolPermissionResponse(allowed=True, remember=False)

                    elif option_id == "allow_always":
                        async with self._lock:
                            self._remembered_permissions[permission_key] = True
                            self._save_permissions()
                        logger.info(
                            f"Remembering allow for {permission_key}",
                            name="acp_tool_permission_remember_allow",
                        )
                        return ToolPermissionResponse(allowed=True, remember=True)

                    elif option_id == "reject_once":
                        return ToolPermissionResponse(
                            allowed=False,
                            remember=False,
                            error_message="The User declined this operation",
                        )

                    elif option_id == "reject_always":
                        async with self._lock:
                            self._remembered_permissions[permission_key] = False
                            self._save_permissions()
                        logger.info(
                            f"Remembering reject for {permission_key}",
                            name="acp_tool_permission_remember_reject",
                        )
                        return ToolPermissionResponse(
                            allowed=False,
                            remember=True,
                            error_message="The User declined this operation",
                        )

            # Default to rejection if we can't parse the response
            logger.warning(
                f"Unknown permission response for {permission_key}, defaulting to reject",
                name="acp_tool_permission_unknown",
            )
            return ToolPermissionResponse(
                allowed=False,
                remember=False,
                error_message="The User declined this operation",
            )

        except Exception as e:
            logger.error(
                f"Error requesting tool permission: {e}",
                name="acp_tool_permission_error",
                exc_info=True,
            )
            # Deny on error with specific message
            return ToolPermissionResponse(
                allowed=False,
                remember=False,
                error_message="An error occurred requesting permission for the call",
            )

    async def clear_remembered_permissions(
        self, tool_name: str | None = None, server_name: str | None = None
    ) -> None:
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
            self._save_permissions()


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
        return await permission_manager.request_permission(
            tool_name=request.tool_name,
            server_name=request.server_name,
            arguments=request.arguments,
            tool_call_id=request.tool_call_id,
        )

    return handler
