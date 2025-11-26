"""
File-based permission storage for .fast-agent/auths.md

Manages persistent tool permissions in a human-readable markdown format.
Only creates the file when an "always" permission is set, avoiding
unnecessary file creation for single-use permissions.
"""

import asyncio
from pathlib import Path
from typing import Literal

from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)

PermissionType = Literal["allow_always", "reject_always"]

# File header for auths.md
_FILE_HEADER = """# Fast-Agent Tool Permissions

This file stores persistent tool permissions set during ACP sessions.
Edit or delete entries to change permissions.

| Server | Tool | Permission |
|--------|------|------------|
"""


class PermissionStore:
    """
    Manages persistent tool permissions in .fast-agent/auths.md

    Permissions are stored in a simple markdown table format for human
    readability and easy manual editing. The file is only created when
    a persistent permission (allow_always or reject_always) is set.

    Thread-safe via asyncio lock.
    """

    def __init__(self, base_path: Path | None = None):
        """
        Initialize the permission store.

        Args:
            base_path: Base directory for .fast-agent folder.
                      Defaults to current working directory.
        """
        self._base_path = base_path or Path.cwd()
        self._cache: dict[str, PermissionType] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    @property
    def _file_path(self) -> Path:
        """Path to the auths.md file."""
        return self._base_path / ".fast-agent" / "auths.md"

    def _make_key(self, server_name: str, tool_name: str) -> str:
        """Create a cache key from server and tool names."""
        return f"{server_name}/{tool_name}"

    def _load_sync(self) -> None:
        """Load permissions from file (synchronous, called under lock)."""
        if self._loaded:
            return

        self._cache.clear()

        if not self._file_path.exists():
            self._loaded = True
            return

        try:
            content = self._file_path.read_text(encoding="utf-8")
            # Parse markdown table
            in_table = False
            for line in content.splitlines():
                line = line.strip()
                # Skip header rows
                if line.startswith("| Server"):
                    in_table = True
                    continue
                if line.startswith("|---"):
                    continue
                if not line.startswith("|") or not in_table:
                    continue

                # Parse table row: | server | tool | permission |
                parts = [p.strip() for p in line.split("|")]
                # parts will be ['', 'server', 'tool', 'permission', '']
                if len(parts) >= 4:
                    server = parts[1]
                    tool = parts[2]
                    permission = parts[3]

                    if permission in ("allow_always", "reject_always"):
                        key = self._make_key(server, tool)
                        self._cache[key] = permission  # type: ignore[assignment]
                        logger.debug(
                            f"Loaded permission: {key} -> {permission}",
                            name="permission_store_load",
                        )

            self._loaded = True
            logger.info(
                f"Loaded {len(self._cache)} permissions from {self._file_path}",
                name="permission_store_loaded",
            )

        except Exception as e:
            logger.error(
                f"Error loading permissions from {self._file_path}: {e}",
                name="permission_store_load_error",
                exc_info=True,
            )
            self._loaded = True  # Mark loaded to avoid repeated failures

    def _save_sync(self) -> None:
        """Save permissions to file (synchronous, called under lock)."""
        if not self._cache:
            # No permissions to save, don't create file
            return

        try:
            # Ensure .fast-agent directory exists
            self._file_path.parent.mkdir(parents=True, exist_ok=True)

            # Build file content
            lines = [_FILE_HEADER.rstrip()]

            for key, permission in sorted(self._cache.items()):
                server, tool = key.split("/", 1)
                lines.append(f"| {server} | {tool} | {permission} |")

            content = "\n".join(lines) + "\n"
            self._file_path.write_text(content, encoding="utf-8")

            logger.info(
                f"Saved {len(self._cache)} permissions to {self._file_path}",
                name="permission_store_saved",
            )

        except Exception as e:
            logger.error(
                f"Error saving permissions to {self._file_path}: {e}",
                name="permission_store_save_error",
                exc_info=True,
            )

    async def load(self) -> None:
        """Load permissions from file."""
        async with self._lock:
            self._load_sync()

    async def get(self, server_name: str, tool_name: str) -> PermissionType | None:
        """
        Get stored permission for a tool.

        Args:
            server_name: Name of the server/runtime
            tool_name: Name of the tool

        Returns:
            The stored permission type, or None if no permission is stored.
        """
        async with self._lock:
            self._load_sync()
            key = self._make_key(server_name, tool_name)
            return self._cache.get(key)

    async def set(
        self, server_name: str, tool_name: str, permission: PermissionType
    ) -> None:
        """
        Store permission for a tool.

        Creates the .fast-agent/auths.md file if it doesn't exist.

        Args:
            server_name: Name of the server/runtime
            tool_name: Name of the tool
            permission: The permission to store
        """
        async with self._lock:
            self._load_sync()
            key = self._make_key(server_name, tool_name)
            self._cache[key] = permission
            self._save_sync()

            logger.info(
                f"Stored permission: {key} -> {permission}",
                name="permission_store_set",
            )

    async def remove(self, server_name: str, tool_name: str) -> bool:
        """
        Remove a stored permission.

        Args:
            server_name: Name of the server/runtime
            tool_name: Name of the tool

        Returns:
            True if permission was removed, False if it didn't exist.
        """
        async with self._lock:
            self._load_sync()
            key = self._make_key(server_name, tool_name)
            if key in self._cache:
                del self._cache[key]
                self._save_sync()
                logger.info(
                    f"Removed permission: {key}",
                    name="permission_store_remove",
                )
                return True
            return False

    async def clear(self) -> None:
        """Clear all stored permissions."""
        async with self._lock:
            self._cache.clear()
            # Delete the file if it exists
            if self._file_path.exists():
                try:
                    self._file_path.unlink()
                    logger.info(
                        f"Deleted permissions file: {self._file_path}",
                        name="permission_store_cleared",
                    )
                except Exception as e:
                    logger.error(
                        f"Error deleting permissions file: {e}",
                        name="permission_store_clear_error",
                    )
            self._loaded = False

    def get_all_sync(self) -> dict[str, PermissionType]:
        """
        Get all stored permissions (synchronous, for display).

        Returns:
            Dict mapping "server/tool" keys to permission types.
        """
        return dict(self._cache)
