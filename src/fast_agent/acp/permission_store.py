"""
ACP Tool Permission Store

Provides persistent storage for tool permission decisions.
Permissions are stored in a markdown file for human readability and editability.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)

# Permission file location relative to session's working directory
PERMISSIONS_DIR = ".fast-agent"
PERMISSIONS_FILE = "auths.md"


class PermissionDecision(Enum):
    """Represents a stored permission decision."""

    ALLOW_ALWAYS = "allow_always"
    REJECT_ALWAYS = "reject_always"


@dataclass
class PermissionResult:
    """Result of a permission check or request."""

    allowed: bool
    remember: bool
    cancelled: bool = False

    @classmethod
    def allow_once(cls) -> "PermissionResult":
        """Create an allow-once result."""
        return cls(allowed=True, remember=False)

    @classmethod
    def allow_always(cls) -> "PermissionResult":
        """Create an allow-always result."""
        return cls(allowed=True, remember=True)

    @classmethod
    def reject_once(cls) -> "PermissionResult":
        """Create a reject-once result."""
        return cls(allowed=False, remember=False)

    @classmethod
    def reject_always(cls) -> "PermissionResult":
        """Create a reject-always result."""
        return cls(allowed=False, remember=True)

    @classmethod
    def cancelled_result(cls) -> "PermissionResult":
        """Create a cancelled result."""
        return cls(allowed=False, remember=False, cancelled=True)

    @classmethod
    def denied(cls) -> "PermissionResult":
        """Create a denied result (default fail-safe)."""
        return cls(allowed=False, remember=False)


ToolKindT = Literal["read", "edit", "delete", "move", "search", "execute", "think", "fetch", "other"]


class PermissionStore:
    """
    Stores and persists tool permission decisions.

    Permissions are stored in .fast-agent/auths.md in the session's working directory.
    The file is only created when the first "always" permission is stored.

    Format is a markdown table:
    | Server | Tool | Permission |
    |--------|------|------------|
    | server_name | tool_name | allow_always |
    """

    def __init__(self, working_directory: str | Path | None = None) -> None:
        """
        Initialize the permission store.

        Args:
            working_directory: The session's working directory for storing permissions.
                              If None, file persistence is disabled.
        """
        self._permissions: dict[str, PermissionDecision] = {}
        self._working_directory = Path(working_directory) if working_directory else None
        self._lock = asyncio.Lock()
        self._loaded = False

    @staticmethod
    def _get_permission_key(server_name: str, tool_name: str) -> str:
        """Get a unique key for a server/tool combination."""
        return f"{server_name}/{tool_name}"

    @property
    def _permissions_path(self) -> Path | None:
        """Get the path to the permissions file."""
        if not self._working_directory:
            return None
        return self._working_directory / PERMISSIONS_DIR / PERMISSIONS_FILE

    async def _ensure_loaded(self) -> None:
        """Load permissions from file if not already loaded."""
        if self._loaded or not self._permissions_path:
            return

        async with self._lock:
            if self._loaded:
                return
            await self._load_from_file()
            self._loaded = True

    async def _load_from_file(self) -> None:
        """Load permissions from the markdown file."""
        if not self._permissions_path or not self._permissions_path.exists():
            return

        try:
            content = self._permissions_path.read_text(encoding="utf-8")
            self._parse_markdown_table(content)
            logger.debug(
                f"Loaded {len(self._permissions)} permissions from {self._permissions_path}",
                name="permission_store_loaded",
            )
        except Exception as e:
            logger.warning(
                f"Failed to load permissions file: {e}",
                name="permission_store_load_error",
                exc_info=True,
            )
            # Continue without persisted permissions - don't fail

    def _parse_markdown_table(self, content: str) -> None:
        """Parse the markdown table format."""
        lines = content.strip().split("\n")

        # Skip header and separator lines
        # Format: | Server | Tool | Permission |
        #         |--------|------|------------|
        #         | server | tool | allow_always |
        for line in lines:
            line = line.strip()
            if not line.startswith("|") or line.startswith("| Server") or "---" in line:
                continue

            # Parse table row
            parts = [p.strip() for p in line.split("|")]
            # parts[0] is empty (before first |), parts[-1] is empty (after last |)
            if len(parts) >= 4:  # |server|tool|permission|
                server_name = parts[1]
                tool_name = parts[2]
                permission_str = parts[3]

                if not server_name or not tool_name or not permission_str:
                    continue

                try:
                    decision = PermissionDecision(permission_str)
                    key = self._get_permission_key(server_name, tool_name)
                    self._permissions[key] = decision
                except ValueError:
                    logger.warning(
                        f"Unknown permission value '{permission_str}' for {server_name}/{tool_name}",
                        name="permission_store_parse_warning",
                    )

    async def _save_to_file(self) -> None:
        """Save permissions to the markdown file."""
        if not self._permissions_path:
            return

        # Only save if we have permissions to persist
        if not self._permissions:
            return

        try:
            # Ensure directory exists
            self._permissions_path.parent.mkdir(parents=True, exist_ok=True)

            # Build markdown table
            lines = [
                "# Tool Permissions",
                "",
                "This file stores persistent tool permission decisions for fast-agent ACP sessions.",
                "You can edit this file manually to change permissions.",
                "",
                "| Server | Tool | Permission |",
                "|--------|------|------------|",
            ]

            for key, decision in sorted(self._permissions.items()):
                server_name, tool_name = key.split("/", 1)
                lines.append(f"| {server_name} | {tool_name} | {decision.value} |")

            lines.append("")  # Trailing newline

            self._permissions_path.write_text("\n".join(lines), encoding="utf-8")
            logger.debug(
                f"Saved {len(self._permissions)} permissions to {self._permissions_path}",
                name="permission_store_saved",
            )
        except Exception as e:
            logger.warning(
                f"Failed to save permissions file: {e}",
                name="permission_store_save_error",
                exc_info=True,
            )
            # Continue - persistence failure shouldn't break functionality

    async def check(self, server_name: str, tool_name: str) -> PermissionDecision | None:
        """
        Check if there's a stored permission for a server/tool combination.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool

        Returns:
            PermissionDecision if found, None otherwise
        """
        await self._ensure_loaded()

        key = self._get_permission_key(server_name, tool_name)
        async with self._lock:
            return self._permissions.get(key)

    async def store(
        self, server_name: str, tool_name: str, decision: PermissionDecision
    ) -> None:
        """
        Store a permission decision.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool
            decision: The permission decision to store
        """
        await self._ensure_loaded()

        key = self._get_permission_key(server_name, tool_name)
        async with self._lock:
            self._permissions[key] = decision
            await self._save_to_file()

        logger.info(
            f"Stored permission {decision.value} for {server_name}/{tool_name}",
            name="permission_store_stored",
            server_name=server_name,
            tool_name=tool_name,
            decision=decision.value,
        )

    async def clear(self, server_name: str | None = None, tool_name: str | None = None) -> None:
        """
        Clear stored permissions.

        Args:
            server_name: If provided with tool_name, clears specific permission.
                        If provided alone, clears all permissions for that server.
            tool_name: Must be provided with server_name.
        """
        await self._ensure_loaded()

        async with self._lock:
            if server_name and tool_name:
                # Clear specific permission
                key = self._get_permission_key(server_name, tool_name)
                self._permissions.pop(key, None)
            elif server_name:
                # Clear all permissions for a server
                keys_to_remove = [
                    k for k in self._permissions.keys() if k.startswith(f"{server_name}/")
                ]
                for k in keys_to_remove:
                    del self._permissions[k]
            else:
                # Clear all permissions
                self._permissions.clear()

            await self._save_to_file()


def infer_tool_kind(tool_name: str, arguments: dict | None = None) -> ToolKindT:
    """
    Infer the tool kind from the tool name and arguments.

    This is used to categorize tools for permission UI and logging.

    Args:
        tool_name: Name of the tool being called
        arguments: Tool arguments (not currently used)

    Returns:
        The inferred tool kind
    """
    name_lower = tool_name.lower()

    # Common patterns for tool categorization
    if any(word in name_lower for word in ["read", "get", "fetch", "list", "show"]):
        return "read"
    elif any(word in name_lower for word in ["write", "edit", "update", "modify", "patch"]):
        return "edit"
    elif any(word in name_lower for word in ["delete", "remove", "clear", "clean", "rm"]):
        return "delete"
    elif any(word in name_lower for word in ["move", "rename", "mv"]):
        return "move"
    elif any(word in name_lower for word in ["search", "find", "query", "grep"]):
        return "search"
    elif any(
        word in name_lower for word in ["execute", "run", "exec", "command", "bash", "shell"]
    ):
        return "execute"
    elif any(word in name_lower for word in ["think", "plan", "reason"]):
        return "think"
    elif any(word in name_lower for word in ["fetch", "download", "http", "request"]):
        return "fetch"

    return "other"
