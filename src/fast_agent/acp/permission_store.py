"""
ACP Tool Permission Store

Provides persistent storage for tool execution permissions.
Stores permissions in a human-readable markdown file within the fast-agent environment.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from fast_agent.constants import DEFAULT_ENVIRONMENT_DIR
from fast_agent.core.logging.logger import get_logger
from fast_agent.paths import resolve_environment_paths
from fast_agent.utils.async_utils import run_in_thread
from fast_agent.utils.markdown import escape_markdown_table_cell

logger = get_logger(__name__)


class PermissionDecision(str, Enum):
    """Stored permission decisions (only 'always' variants are persisted)."""

    ALLOW_ALWAYS = "allow_always"
    REJECT_ALWAYS = "reject_always"


@dataclass
class PermissionResult:
    """Result of a permission check or request."""

    allowed: bool
    remember: bool = False
    is_cancelled: bool = False

    @classmethod
    def allow_once(cls) -> "PermissionResult":
        """Create an allow-once result (not persisted)."""
        return cls(allowed=True, remember=False)

    @classmethod
    def allow_always(cls) -> "PermissionResult":
        """Create an allow-always result (persisted)."""
        return cls(allowed=True, remember=True)

    @classmethod
    def reject_once(cls) -> "PermissionResult":
        """Create a reject-once result (not persisted)."""
        return cls(allowed=False, remember=False)

    @classmethod
    def reject_always(cls) -> "PermissionResult":
        """Create a reject-always result (persisted)."""
        return cls(allowed=False, remember=True)

    @classmethod
    def cancelled(cls) -> "PermissionResult":
        """Create a cancelled result (rejected, not persisted)."""
        return cls(allowed=False, remember=False, is_cancelled=True)


@dataclass(frozen=True, slots=True)
class PermissionEntry:
    """Stored permission identity for one MCP server/tool pair."""

    server_name: str
    tool_name: str

    @property
    def display_key(self) -> str:
        return f"{self.server_name}/{self.tool_name}"


def _unescape_markdown_table_cell(value: str) -> str:
    result: list[str] = []
    escaped = False
    for char in value:
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        result.append(char)
    if escaped:
        result.append("\\")
    return "".join(result)


def _split_markdown_table_row(line: str) -> list[str]:
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in line.strip()[1:-1]:
        if escaped:
            current.append("\\")
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "|":
            cells.append(_unescape_markdown_table_cell("".join(current).strip()))
            current = []
            continue
        current.append(char)
    if escaped:
        current.append("\\")
    cells.append(_unescape_markdown_table_cell("".join(current).strip()))
    return cells


class PermissionStore:
    """
    Persistent storage for tool execution permissions.

    Stores allow_always and reject_always decisions in a markdown file
    that is human-readable and editable. The file is only created when
    the first 'always' permission is set.

    Thread-safe for concurrent access using asyncio locks.
    """

    def __init__(self, cwd: str | Path | None = None) -> None:
        """
        Initialize the permission store.

        Args:
            cwd: Working directory for the session. If None, uses current directory.
        """
        self._cwd = Path(cwd) if cwd else Path.cwd()
        override = DEFAULT_ENVIRONMENT_DIR if cwd is not None else None
        env_paths = resolve_environment_paths(cwd=self._cwd, override=override)
        self._file_path = env_paths.permissions_file
        self._cache: dict[PermissionEntry, PermissionDecision] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    @property
    def file_path(self) -> Path:
        """Get the path to the permissions file."""
        return self._file_path

    def _get_permission_key(self, server_name: str, tool_name: str) -> PermissionEntry:
        """Get a unique key for a server/tool combination."""
        return PermissionEntry(server_name, tool_name)

    async def _ensure_loaded(self) -> None:
        """Ensure permissions are loaded from disk (lazy loading)."""
        if self._loaded:
            return

        if self._file_path.exists():
            try:
                await self._load_from_file()
            except Exception as e:
                logger.warning(
                    f"Failed to load permissions file: {e}",
                    name="permission_store_load_error",
                )
                # Continue without persisted permissions
        self._loaded = True

    async def _load_from_file(self) -> None:
        """Load permissions from the markdown file."""
        content = await run_in_thread(self._file_path.read_text, encoding="utf-8")

        # Parse markdown table format:
        # | Server | Tool | Permission |
        # |--------|------|------------|
        # | server1 | tool1 | allow_always |

        in_table = False
        for line_number, line in enumerate(content.splitlines(), start=1):
            stripped_line = line.strip()

            # Skip empty lines and header
            if not stripped_line:
                continue
            if stripped_line.startswith("# "):
                continue
            if stripped_line.startswith(("|--", "| --")):
                in_table = True
                continue
            if stripped_line.startswith("| Server"):
                continue

            # Parse table rows
            if in_table and stripped_line.startswith("|") and stripped_line.endswith("|"):
                parts = _split_markdown_table_row(stripped_line)
                if len(parts) >= 3:
                    server_name, tool_name, permission = parts[0], parts[1], parts[2]
                    if not server_name or not tool_name:
                        continue
                    key = self._get_permission_key(server_name, tool_name)
                    try:
                        self._cache[key] = PermissionDecision(permission)
                    except ValueError:
                        logger.warning(
                            f"Invalid permission value in auths.md: {permission}",
                            name="permission_store_parse_error",
                            file_path=str(self._file_path),
                            line_number=line_number,
                            server_name=server_name,
                            tool_name=tool_name,
                        )

    async def _delete_file(self) -> None:
        if self._file_path.exists():
            await run_in_thread(self._file_path.unlink)

    async def _save_to_file(self) -> None:
        """Save permissions to the markdown file."""
        if not self._cache:
            await self._delete_file()
            return

        # Ensure directory exists
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        # Build markdown content
        lines = [
            "# fast-agent Tool Permissions",
            "",
            "This file stores persistent tool execution permissions.",
            "You can edit this file manually to add or remove permissions.",
            "",
            "| Server | Tool | Permission |",
            "|--------|------|------------|",
        ]

        for key, decision in sorted(self._cache.items(), key=lambda item: item[0].display_key):
            server_name = escape_markdown_table_cell(key.server_name)
            tool_name = escape_markdown_table_cell(key.tool_name)
            lines.append(f"| {server_name} | {tool_name} | {decision.value} |")

        lines.append("")  # Trailing newline
        content = "\n".join(lines)

        await run_in_thread(self._file_path.write_text, content, encoding="utf-8")

        logger.debug(
            f"Saved {len(self._cache)} permissions to {self._file_path}",
            name="permission_store_saved",
        )

    async def get(self, server_name: str, tool_name: str) -> PermissionDecision | None:
        """
        Get stored permission for a server/tool.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool

        Returns:
            PermissionDecision if stored, None if not found
        """
        async with self._lock:
            await self._ensure_loaded()
            key = self._get_permission_key(server_name, tool_name)
            return self._cache.get(key)

    async def set(self, server_name: str, tool_name: str, decision: PermissionDecision) -> None:
        """
        Store a permission decision.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool
            decision: The permission decision to store
        """
        async with self._lock:
            await self._ensure_loaded()
            key = self._get_permission_key(server_name, tool_name)
            self._cache[key] = decision
            try:
                await self._save_to_file()
            except Exception as e:
                logger.warning(
                    f"Failed to save permissions file: {e}",
                    name="permission_store_save_error",
                )
                # Continue - in-memory cache is still valid

    async def remove(self, server_name: str, tool_name: str) -> bool:
        """
        Remove a stored permission.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool

        Returns:
            True if permission was removed, False if not found
        """
        async with self._lock:
            await self._ensure_loaded()
            key = self._get_permission_key(server_name, tool_name)
            if key in self._cache:
                del self._cache[key]
                try:
                    await self._save_to_file()
                except Exception as e:
                    logger.warning(
                        f"Failed to save permissions file after removal: {e}",
                        name="permission_store_save_error",
                    )
                return True
            return False

    async def clear(self) -> None:
        """Clear all stored permissions."""
        async with self._lock:
            self._cache.clear()
            try:
                await self._delete_file()
            except Exception as e:
                logger.warning(
                    f"Failed to delete permissions file: {e}",
                    name="permission_store_delete_error",
                )
