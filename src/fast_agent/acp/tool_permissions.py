"""
ACP Tool Call Permissions

Provides a permission handler that requests tool execution permission from the ACP client.
This follows the same pattern as elicitation handlers but for tool execution authorization.

Features:
- Permission requests via ACP session/request_permission
- File persistence in .fast-agent/auths.md (markdown table format)
- Fail-safe behavior: DENY on any error
- Support for allow_once, allow_always, reject_once, reject_always
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from acp.schema import (
    PermissionOption,
    RequestPermissionRequest,
    ToolCall,
)

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from acp import AgentSideConnection

logger = get_logger(__name__)


# Permission file location
PERMISSION_FILE_NAME = "auths.md"
PERMISSION_DIR_NAME = ".fast-agent"


class PermissionDecision(str, Enum):
    """Stored permission decisions."""

    ALLOW_ALWAYS = "allow_always"
    REJECT_ALWAYS = "reject_always"


@dataclass(frozen=True)
class PermissionResult:
    """
    Result from a permission request.

    This is the outcome after checking stored permissions or requesting
    permission from the ACP client.
    """

    allowed: bool
    remember: bool = False
    was_cancelled: bool = False
    error: str | None = None

    @property
    def cancelled(self) -> bool:
        """Check if the permission request was cancelled."""
        return self.was_cancelled

    @classmethod
    def allow_once(cls) -> "PermissionResult":
        """Create a result for allow once (don't persist)."""
        return cls(allowed=True, remember=False)

    @classmethod
    def allow_always(cls) -> "PermissionResult":
        """Create a result for allow always (persist)."""
        return cls(allowed=True, remember=True)

    @classmethod
    def reject_once(cls) -> "PermissionResult":
        """Create a result for reject once (don't persist)."""
        return cls(allowed=False, remember=False)

    @classmethod
    def reject_always(cls) -> "PermissionResult":
        """Create a result for reject always (persist)."""
        return cls(allowed=False, remember=True)

    @classmethod
    def create_cancelled(cls) -> "PermissionResult":
        """Create a result for cancelled request."""
        return cls(allowed=False, remember=False, was_cancelled=True)

    @classmethod
    def denied_with_error(cls, error: str) -> "PermissionResult":
        """Create a denial result due to an error (fail-safe)."""
        return cls(allowed=False, remember=False, error=error)


@dataclass
class ToolPermissionRequest:
    """Request for tool execution permission."""

    tool_name: str
    server_name: str
    arguments: dict[str, Any] | None = None
    tool_call_id: str | None = None


# Type for permission handler callbacks
ToolPermissionHandlerT = Callable[
    [ToolPermissionRequest], "asyncio.Future[PermissionResult]"
]


class PermissionStore:
    """
    Persistent storage for tool permissions.

    Stores permission decisions in a markdown file at .fast-agent/auths.md
    in the session's working directory. The file uses a human-readable
    markdown table format that can be manually edited.

    Format:
    ```markdown
    # Fast-Agent Tool Permissions

    | Server | Tool | Permission |
    |--------|------|------------|
    | mcp_server | write_file | allow_always |
    | mcp_server | delete_file | reject_always |
    ```
    """

    def __init__(self, cwd: str | Path | None = None) -> None:
        """
        Initialize the permission store.

        Args:
            cwd: Working directory for the session. If None, uses current directory.
        """
        self._cwd = Path(cwd) if cwd else Path.cwd()
        self._permissions: dict[str, PermissionDecision] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    @property
    def file_path(self) -> Path:
        """Get the path to the permissions file."""
        return self._cwd / PERMISSION_DIR_NAME / PERMISSION_FILE_NAME

    def _get_key(self, server_name: str, tool_name: str) -> str:
        """Get a unique key for a server/tool combination."""
        return f"{server_name}/{tool_name}"

    def _parse_key(self, key: str) -> tuple[str, str]:
        """Parse a key back into server_name and tool_name."""
        parts = key.split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "", key

    def _load_from_file(self) -> None:
        """Load permissions from the markdown file."""
        if not self.file_path.exists():
            self._loaded = True
            return

        try:
            content = self.file_path.read_text(encoding="utf-8")
            self._permissions = self._parse_markdown_table(content)
            self._loaded = True
            logger.debug(
                f"Loaded {len(self._permissions)} permissions from {self.file_path}",
                name="permission_store_loaded",
            )
        except Exception as e:
            logger.warning(
                f"Failed to load permissions file: {e}",
                name="permission_store_load_error",
            )
            # Continue without persistence
            self._loaded = True

    def _parse_markdown_table(self, content: str) -> dict[str, PermissionDecision]:
        """Parse the markdown table format."""
        permissions: dict[str, PermissionDecision] = {}

        lines = content.strip().split("\n")
        in_table = False

        for line in lines:
            line = line.strip()
            if not line.startswith("|"):
                continue

            # Skip header and separator lines
            if "Server" in line or "---" in line:
                in_table = True
                continue

            if not in_table:
                continue

            # Parse table row: | server | tool | permission |
            parts = [p.strip() for p in line.split("|")]
            # Filter out empty parts (from leading/trailing |)
            parts = [p for p in parts if p]

            if len(parts) >= 3:
                server_name = parts[0]
                tool_name = parts[1]
                permission_str = parts[2]

                try:
                    permission = PermissionDecision(permission_str)
                    key = self._get_key(server_name, tool_name)
                    permissions[key] = permission
                except ValueError:
                    logger.warning(
                        f"Invalid permission value: {permission_str}",
                        name="permission_store_invalid_value",
                    )

        return permissions

    def _save_to_file(self) -> None:
        """Save permissions to the markdown file."""
        if not self._permissions:
            # Don't create file if there are no permissions
            return

        try:
            # Create directory if it doesn't exist
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            content = self._generate_markdown_table()
            self.file_path.write_text(content, encoding="utf-8")

            logger.debug(
                f"Saved {len(self._permissions)} permissions to {self.file_path}",
                name="permission_store_saved",
            )
        except Exception as e:
            logger.warning(
                f"Failed to save permissions file: {e}",
                name="permission_store_save_error",
            )
            # Continue without persistence - don't break the workflow

    def _generate_markdown_table(self) -> str:
        """Generate the markdown table content."""
        lines = [
            "# Fast-Agent Tool Permissions",
            "",
            "This file stores persistent tool permission decisions.",
            "You can manually edit this file to modify permissions.",
            "",
            "| Server | Tool | Permission |",
            "|--------|------|------------|",
        ]

        for key, permission in sorted(self._permissions.items()):
            server_name, tool_name = self._parse_key(key)
            lines.append(f"| {server_name} | {tool_name} | {permission.value} |")

        lines.append("")
        return "\n".join(lines)

    async def get(self, server_name: str, tool_name: str) -> PermissionDecision | None:
        """
        Get the stored permission for a server/tool combination.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool

        Returns:
            The stored PermissionDecision, or None if not found
        """
        async with self._lock:
            if not self._loaded:
                self._load_from_file()

            key = self._get_key(server_name, tool_name)
            return self._permissions.get(key)

    async def set(
        self, server_name: str, tool_name: str, decision: PermissionDecision
    ) -> None:
        """
        Store a permission decision.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool
            decision: The permission decision to store
        """
        async with self._lock:
            if not self._loaded:
                self._load_from_file()

            key = self._get_key(server_name, tool_name)
            self._permissions[key] = decision
            self._save_to_file()

    async def remove(self, server_name: str, tool_name: str) -> None:
        """
        Remove a stored permission.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool
        """
        async with self._lock:
            if not self._loaded:
                self._load_from_file()

            key = self._get_key(server_name, tool_name)
            if key in self._permissions:
                del self._permissions[key]
                self._save_to_file()

    async def clear(self) -> None:
        """Clear all stored permissions."""
        async with self._lock:
            self._permissions.clear()
            # Don't delete the file, just save empty state
            if self.file_path.exists():
                self._save_to_file()


def _infer_tool_kind(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
    """
    Infer the ACP tool kind from the tool name and arguments.

    This is used to provide appropriate categorization for permission requests
    and tool call notifications.

    Args:
        tool_name: Name of the tool being called
        arguments: Optional tool arguments (may provide additional context)

    Returns:
        The inferred tool kind: read, edit, delete, move, search, execute, think, fetch, or other
    """
    name_lower = tool_name.lower()

    # Search operations - check before read since "locate" contains "cat"
    if any(word in name_lower for word in ["search", "find", "query", "grep", "locate", "lookup"]):
        return "search"

    # Fetch operations - check before read since patterns might overlap
    if any(word in name_lower for word in ["fetch", "download", "http", "request", "curl", "wget"]):
        return "fetch"

    # Execute operations
    if any(word in name_lower for word in ["execute", "run", "exec", "command", "bash", "shell", "spawn"]):
        return "execute"

    # Delete operations
    if any(word in name_lower for word in ["delete", "remove", "clear", "clean", "rm", "unlink", "drop"]):
        return "delete"

    # Move operations
    if any(word in name_lower for word in ["move", "rename", "mv", "copy", "cp"]):
        return "move"

    # Edit operations - check before read since "create" and "add" are more specific
    if any(word in name_lower for word in ["write", "edit", "update", "modify", "patch", "set", "create", "add", "append"]):
        return "edit"

    # Think operations
    if any(word in name_lower for word in ["think", "plan", "reason", "analyze", "evaluate"]):
        return "think"

    # Read operations - check last as these are very common patterns
    if any(word in name_lower for word in ["read", "get", "list", "show", "cat", "head", "tail", "view"]):
        return "read"

    # API operations - catch any "api" or "call" related tools
    if any(word in name_lower for word in ["api", "call"]):
        return "fetch"

    return "other"


class ACPToolPermissionManager:
    """
    Manages tool execution permission requests via ACP.

    This class provides a handler that can be used to request permission
    from the ACP client before executing tools.

    Key behaviors:
    - Checks PermissionStore first for persisted decisions
    - Sends session/request_permission to client if no stored decision
    - Persists allow_always/reject_always decisions
    - Fail-safe: DENY on any error (never silently allow on failure)
    """

    def __init__(
        self,
        connection: "AgentSideConnection",
        cwd: str | Path | None = None,
        permissions_enabled: bool = True,
    ) -> None:
        """
        Initialize the permission manager.

        Args:
            connection: The ACP connection to send permission requests on
            cwd: Working directory for permission persistence
            permissions_enabled: If False, all permissions are auto-granted (--no-permissions mode)
        """
        self._connection = connection
        self._store = PermissionStore(cwd)
        self._permissions_enabled = permissions_enabled
        # In-memory cache for session (faster than file I/O for repeated checks)
        self._session_cache: dict[str, bool] = {}
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
    ) -> PermissionResult:
        """
        Request permission to execute a tool.

        This method implements the full permission flow:
        1. If permissions disabled, immediately allow
        2. Check PermissionStore for persisted decision
        3. Check session cache for this session's decisions
        4. Send session/request_permission to client
        5. Process response and persist if needed

        Fail-safe: On ANY error, returns DENY.

        Args:
            session_id: The ACP session ID
            tool_name: Name of the tool to execute
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments (shown to user)
            tool_call_id: Optional tool call ID for tracking

        Returns:
            PermissionResult indicating whether execution is allowed
        """
        # Fast path: if permissions are disabled, allow all
        if not self._permissions_enabled:
            logger.debug(
                f"Permissions disabled, allowing {server_name}/{tool_name}",
                name="acp_permission_disabled",
            )
            return PermissionResult.allow_once()

        permission_key = self._get_permission_key(tool_name, server_name)

        try:
            # Check persisted permissions first
            stored_decision = await self._store.get(server_name, tool_name)
            if stored_decision:
                if stored_decision == PermissionDecision.ALLOW_ALWAYS:
                    logger.debug(
                        f"Using stored allow_always for {permission_key}",
                        name="acp_permission_stored_allow",
                    )
                    return PermissionResult.allow_always()
                elif stored_decision == PermissionDecision.REJECT_ALWAYS:
                    logger.debug(
                        f"Using stored reject_always for {permission_key}",
                        name="acp_permission_stored_reject",
                    )
                    return PermissionResult.reject_always()

            # Check session cache
            async with self._lock:
                if permission_key in self._session_cache:
                    allowed = self._session_cache[permission_key]
                    logger.debug(
                        f"Using session cache for {permission_key}: {allowed}",
                        name="acp_permission_session_cache",
                    )
                    return PermissionResult(allowed=allowed, remember=True)

            # Need to request permission from client
            return await self._request_from_client(
                session_id=session_id,
                tool_name=tool_name,
                server_name=server_name,
                arguments=arguments,
                tool_call_id=tool_call_id,
            )

        except Exception as e:
            # FAIL-SAFE: On any error, DENY
            logger.error(
                f"Error during permission check for {permission_key}: {e}",
                name="acp_permission_error",
                exc_info=True,
            )
            return PermissionResult.denied_with_error(str(e))

    async def _request_from_client(
        self,
        session_id: str,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None,
        tool_call_id: str | None,
    ) -> PermissionResult:
        """
        Send a permission request to the ACP client.

        Args:
            session_id: The ACP session ID
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
            arg_items = list(arguments.items())[:5]
            arg_lines = []
            for k, v in arg_items:
                v_str = str(v)
                if len(v_str) > 100:
                    v_str = v_str[:97] + "..."
                arg_lines.append(f"  {k}: {v_str}")
            if len(arguments) > 5:
                arg_lines.append(f"  ... and {len(arguments) - 5} more arguments")
            prompt_parts.append("Arguments:")
            prompt_parts.extend(arg_lines)

        prompt = "\n".join(prompt_parts)

        # Infer tool kind for context
        kind = _infer_tool_kind(tool_name, arguments)

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

        # Build ToolCall object per ACP spec (always required)
        # Generate a tool_call_id if not provided
        actual_tool_call_id = tool_call_id
        if not actual_tool_call_id:
            import uuid
            actual_tool_call_id = f"perm-{uuid.uuid4().hex[:8]}"

        tool_call_obj = ToolCall(
            toolCallId=actual_tool_call_id,
            title=f"{server_name}/{tool_name}",
            kind=kind,
            status="pending",
            rawInput=arguments,
        )

        request = RequestPermissionRequest(
            sessionId=session_id,
            prompt=prompt,
            options=options,
            toolCall=tool_call_obj,
        )

        try:
            logger.info(
                f"Requesting permission for {permission_key}",
                name="acp_permission_request",
                tool_name=tool_name,
                server_name=server_name,
                kind=kind,
            )

            # Send permission request to client
            response = await self._connection.requestPermission(request)

            # Handle response
            return await self._process_response(
                response=response,
                server_name=server_name,
                tool_name=tool_name,
                permission_key=permission_key,
            )

        except asyncio.CancelledError:
            # Re-raise cancellation
            raise
        except Exception as e:
            # FAIL-SAFE: On any error during client communication, DENY
            logger.error(
                f"Error requesting tool permission from client: {e}",
                name="acp_permission_client_error",
                exc_info=True,
            )
            return PermissionResult.denied_with_error(f"Client communication error: {e}")

    async def _process_response(
        self,
        response: Any,
        server_name: str,
        tool_name: str,
        permission_key: str,
    ) -> PermissionResult:
        """
        Process the permission response from the client.

        Args:
            response: The RequestPermissionResponse from the client
            server_name: Name of the server
            tool_name: Name of the tool
            permission_key: The unique key for this permission

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
                        name="acp_permission_cancelled",
                    )
                    return PermissionResult.create_cancelled()

                elif outcome_type == "selected":
                    option_id = getattr(outcome, "optionId", None)

                    if option_id == "allow_once":
                        logger.info(
                            f"Permission granted (once) for {permission_key}",
                            name="acp_permission_allow_once",
                        )
                        return PermissionResult.allow_once()

                    elif option_id == "allow_always":
                        logger.info(
                            f"Permission granted (always) for {permission_key}",
                            name="acp_permission_allow_always",
                        )
                        # Persist the decision
                        await self._store.set(
                            server_name, tool_name, PermissionDecision.ALLOW_ALWAYS
                        )
                        # Also cache in session
                        async with self._lock:
                            self._session_cache[permission_key] = True
                        return PermissionResult.allow_always()

                    elif option_id == "reject_once":
                        logger.info(
                            f"Permission rejected (once) for {permission_key}",
                            name="acp_permission_reject_once",
                        )
                        return PermissionResult.reject_once()

                    elif option_id == "reject_always":
                        logger.info(
                            f"Permission rejected (always) for {permission_key}",
                            name="acp_permission_reject_always",
                        )
                        # Persist the decision
                        await self._store.set(
                            server_name, tool_name, PermissionDecision.REJECT_ALWAYS
                        )
                        # Also cache in session
                        async with self._lock:
                            self._session_cache[permission_key] = False
                        return PermissionResult.reject_always()

            # FAIL-SAFE: Unknown response format, DENY
            logger.warning(
                f"Unknown permission response for {permission_key}, defaulting to reject",
                name="acp_permission_unknown_response",
            )
            return PermissionResult.denied_with_error("Unknown response format from client")

        except Exception as e:
            # FAIL-SAFE: Error processing response, DENY
            logger.error(
                f"Error processing permission response: {e}",
                name="acp_permission_response_error",
                exc_info=True,
            )
            return PermissionResult.denied_with_error(f"Error processing response: {e}")

    async def clear_remembered_permissions(
        self, tool_name: str | None = None, server_name: str | None = None
    ) -> None:
        """
        Clear remembered permissions.

        Args:
            tool_name: Optional tool name to clear (clears all if None)
            server_name: Optional server name to clear (clears all if None)
        """
        if tool_name and server_name:
            await self._store.remove(server_name, tool_name)
            permission_key = self._get_permission_key(tool_name, server_name)
            async with self._lock:
                self._session_cache.pop(permission_key, None)
            logger.info(
                f"Cleared permission for {server_name}/{tool_name}",
                name="acp_permission_cleared",
            )
        else:
            await self._store.clear()
            async with self._lock:
                self._session_cache.clear()
            logger.info(
                "Cleared all remembered permissions",
                name="acp_permissions_cleared_all",
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

    async def handler(request: ToolPermissionRequest) -> PermissionResult:
        """Handle tool permission request."""
        return await permission_manager.request_permission(
            session_id=session_id,
            tool_name=request.tool_name,
            server_name=request.server_name,
            arguments=request.arguments,
            tool_call_id=request.tool_call_id,
        )

    return handler


# Import ToolPermissionHandler protocol for type checking
from fast_agent.mcp.tool_execution_handler import (
    ToolPermissionCheckResult,
    ToolPermissionHandler,
)


class ACPPermissionHandlerAdapter:
    """
    Adapter that wraps ACPToolPermissionManager to implement ToolPermissionHandler protocol.

    This allows the permission manager to be used directly with the MCP aggregator's
    permission checking system.
    """

    def __init__(
        self,
        permission_manager: ACPToolPermissionManager,
        session_id: str,
    ) -> None:
        """
        Initialize the adapter.

        Args:
            permission_manager: The ACPToolPermissionManager instance
            session_id: The ACP session ID
        """
        self._permission_manager = permission_manager
        self._session_id = session_id

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None,
        tool_call_id: str | None = None,
    ) -> ToolPermissionCheckResult:
        """
        Check if a tool execution is permitted.

        Args:
            tool_name: Name of the tool to execute
            server_name: Name of the MCP server providing the tool
            arguments: Tool arguments
            tool_call_id: Optional tool call ID for tracking

        Returns:
            ToolPermissionCheckResult indicating whether to proceed
        """
        result = await self._permission_manager.request_permission(
            session_id=self._session_id,
            tool_name=tool_name,
            server_name=server_name,
            arguments=arguments,
            tool_call_id=tool_call_id,
        )

        if result.allowed:
            return ToolPermissionCheckResult.allow()
        else:
            # Build informative error message
            if result.cancelled:
                message = f"Tool execution cancelled: {server_name}/{tool_name}"
            elif result.error:
                message = f"Permission check failed: {result.error}"
            else:
                message = f"Tool execution denied: {server_name}/{tool_name}"
            return ToolPermissionCheckResult.deny(message)
