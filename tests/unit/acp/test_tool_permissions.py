"""
Unit tests for ACP tool permission components.

Tests for:
- PermissionStore file persistence
- PermissionResult factory methods
- _infer_tool_kind function
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from fast_agent.acp.permission_store import (
    DEFAULT_PERMISSIONS_FILE,
    PermissionDecision,
    PermissionResult,
    PermissionStore,
)
from fast_agent.acp.tool_permissions import _infer_tool_kind


class TestPermissionResult:
    """Tests for PermissionResult dataclass."""

    def test_allow_once(self) -> None:
        """allow_once creates allowed=True, remember=False."""
        result = PermissionResult.allow_once()
        assert result.allowed is True
        assert result.remember is False
        assert result.is_cancelled is False

    def test_allow_always(self) -> None:
        """allow_always creates allowed=True, remember=True."""
        result = PermissionResult.allow_always()
        assert result.allowed is True
        assert result.remember is True
        assert result.is_cancelled is False

    def test_reject_once(self) -> None:
        """reject_once creates allowed=False, remember=False."""
        result = PermissionResult.reject_once()
        assert result.allowed is False
        assert result.remember is False
        assert result.is_cancelled is False

    def test_reject_always(self) -> None:
        """reject_always creates allowed=False, remember=True."""
        result = PermissionResult.reject_always()
        assert result.allowed is False
        assert result.remember is True
        assert result.is_cancelled is False

    def test_cancelled(self) -> None:
        """cancelled creates allowed=False, is_cancelled=True."""
        result = PermissionResult.cancelled()
        assert result.allowed is False
        assert result.remember is False
        assert result.is_cancelled is True


class TestPermissionStore:
    """Tests for PermissionStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_tools(self, temp_dir: Path) -> None:
        """get() returns None for tools without stored permissions."""
        store = PermissionStore(cwd=temp_dir)
        result = await store.get("unknown_server", "unknown_tool")
        assert result is None

    @pytest.mark.asyncio
    async def test_stores_and_retrieves_allow_always(self, temp_dir: Path) -> None:
        """Stores and retrieves allow_always decisions."""
        store = PermissionStore(cwd=temp_dir)
        await store.set("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)

        result = await store.get("server1", "tool1")
        assert result == PermissionDecision.ALLOW_ALWAYS

    @pytest.mark.asyncio
    async def test_stores_and_retrieves_reject_always(self, temp_dir: Path) -> None:
        """Stores and retrieves reject_always decisions."""
        store = PermissionStore(cwd=temp_dir)
        await store.set("server1", "tool1", PermissionDecision.REJECT_ALWAYS)

        result = await store.get("server1", "tool1")
        assert result == PermissionDecision.REJECT_ALWAYS

    @pytest.mark.asyncio
    async def test_persists_across_instances(self, temp_dir: Path) -> None:
        """Permissions persist across store instances (file I/O)."""
        # First instance - set permission
        store1 = PermissionStore(cwd=temp_dir)
        await store1.set("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)

        # Second instance - should load from file
        store2 = PermissionStore(cwd=temp_dir)
        result = await store2.get("server1", "tool1")
        assert result == PermissionDecision.ALLOW_ALWAYS

    @pytest.mark.asyncio
    async def test_only_creates_file_when_permission_set(self, temp_dir: Path) -> None:
        """File is only created when first permission is set."""
        store = PermissionStore(cwd=temp_dir)

        # Initially, no file
        assert not (temp_dir / DEFAULT_PERMISSIONS_FILE).exists()

        # Just reading doesn't create file
        await store.get("server1", "tool1")
        assert not (temp_dir / DEFAULT_PERMISSIONS_FILE).exists()

        # Setting permission creates file
        await store.set("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)
        assert (temp_dir / DEFAULT_PERMISSIONS_FILE).exists()

    @pytest.mark.asyncio
    async def test_handles_missing_file_gracefully(self, temp_dir: Path) -> None:
        """get() works when file doesn't exist."""
        store = PermissionStore(cwd=temp_dir)

        # Should not raise
        result = await store.get("server1", "tool1")
        assert result is None

    @pytest.mark.asyncio
    async def test_removes_permission(self, temp_dir: Path) -> None:
        """remove() deletes stored permission."""
        store = PermissionStore(cwd=temp_dir)

        # Set and verify
        await store.set("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)
        assert await store.get("server1", "tool1") == PermissionDecision.ALLOW_ALWAYS

        # Remove
        removed = await store.remove("server1", "tool1")
        assert removed is True

        # Verify removed
        assert await store.get("server1", "tool1") is None

    @pytest.mark.asyncio
    async def test_remove_returns_false_for_missing(self, temp_dir: Path) -> None:
        """remove() returns False for non-existent permissions."""
        store = PermissionStore(cwd=temp_dir)
        removed = await store.remove("server1", "tool1")
        assert removed is False

    @pytest.mark.asyncio
    async def test_clear_removes_all_permissions(self, temp_dir: Path) -> None:
        """clear() removes all stored permissions."""
        store = PermissionStore(cwd=temp_dir)

        # Set multiple permissions
        await store.set("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)
        await store.set("server2", "tool2", PermissionDecision.REJECT_ALWAYS)

        # Clear all
        await store.clear()

        # Verify all removed
        assert await store.get("server1", "tool1") is None
        assert await store.get("server2", "tool2") is None

    @pytest.mark.asyncio
    async def test_list_all_returns_all_permissions(self, temp_dir: Path) -> None:
        """list_all() returns all stored permissions."""
        store = PermissionStore(cwd=temp_dir)

        # Set multiple permissions
        await store.set("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)
        await store.set("server2", "tool2", PermissionDecision.REJECT_ALWAYS)

        all_perms = await store.list_all()
        assert len(all_perms) == 2
        assert all_perms["server1/tool1"] == PermissionDecision.ALLOW_ALWAYS
        assert all_perms["server2/tool2"] == PermissionDecision.REJECT_ALWAYS

    @pytest.mark.asyncio
    async def test_file_format_is_human_readable(self, temp_dir: Path) -> None:
        """The permissions file is human-readable markdown."""
        store = PermissionStore(cwd=temp_dir)
        await store.set("my_server", "my_tool", PermissionDecision.ALLOW_ALWAYS)

        # Read the file content
        file_path = temp_dir / DEFAULT_PERMISSIONS_FILE
        content = file_path.read_text()

        # Check it contains markdown table elements
        assert "| Server | Tool | Permission |" in content
        assert "| my_server | my_tool | allow_always |" in content

    @pytest.mark.asyncio
    async def test_concurrent_access_is_safe(self, temp_dir: Path) -> None:
        """Concurrent access to store is thread-safe."""
        store = PermissionStore(cwd=temp_dir)

        async def set_permission(i: int):
            await store.set(f"server{i}", f"tool{i}", PermissionDecision.ALLOW_ALWAYS)

        # Run many concurrent sets
        await asyncio.gather(*[set_permission(i) for i in range(10)])

        # All should be stored
        all_perms = await store.list_all()
        assert len(all_perms) == 10


class TestInferToolKind:
    """Tests for _infer_tool_kind function."""

    def test_read_tools(self) -> None:
        """Tools with read-like names are classified as 'read'."""
        assert _infer_tool_kind("read_file") == "read"
        assert _infer_tool_kind("get_data") == "read"
        assert _infer_tool_kind("list_files") == "read"
        assert _infer_tool_kind("show_status") == "read"
        # Note: "fetch" is in the "read" list, so fetch_X -> "read" (not "fetch")
        # The "fetch" category is for tools with only "fetch" pattern after read check

    def test_edit_tools(self) -> None:
        """Tools with edit-like names are classified as 'edit'."""
        assert _infer_tool_kind("write_file") == "edit"
        assert _infer_tool_kind("edit_document") == "edit"
        assert _infer_tool_kind("update_config") == "edit"
        assert _infer_tool_kind("modify_settings") == "edit"
        assert _infer_tool_kind("create_file") == "edit"

    def test_delete_tools(self) -> None:
        """Tools with delete-like names are classified as 'delete'."""
        assert _infer_tool_kind("delete_file") == "delete"
        assert _infer_tool_kind("remove_item") == "delete"
        assert _infer_tool_kind("clear_cache") == "delete"
        assert _infer_tool_kind("clean_temp") == "delete"

    def test_execute_tools(self) -> None:
        """Tools with execute-like names are classified as 'execute'."""
        assert _infer_tool_kind("execute_command") == "execute"
        assert _infer_tool_kind("run_script") == "execute"
        assert _infer_tool_kind("exec_sql") == "execute"
        assert _infer_tool_kind("bash_command") == "execute"

    def test_search_tools(self) -> None:
        """Tools with search-like names are classified as 'search'."""
        assert _infer_tool_kind("search_files") == "search"
        assert _infer_tool_kind("find_pattern") == "search"
        assert _infer_tool_kind("query_database") == "search"
        assert _infer_tool_kind("grep_content") == "search"

    def test_move_tools(self) -> None:
        """Tools with move-like names are classified as 'move'."""
        assert _infer_tool_kind("move_file") == "move"
        assert _infer_tool_kind("rename_item") == "move"
        assert _infer_tool_kind("copy_document") == "move"

    def test_unknown_tools_return_other(self) -> None:
        """Tools without matching patterns return 'other'."""
        assert _infer_tool_kind("foo_bar") == "other"
        assert _infer_tool_kind("my_custom_tool") == "other"
        assert _infer_tool_kind("process_data") == "other"

    def test_case_insensitive(self) -> None:
        """Pattern matching is case-insensitive."""
        assert _infer_tool_kind("READ_FILE") == "read"
        assert _infer_tool_kind("Delete_Item") == "delete"
        assert _infer_tool_kind("EXECUTE_CMD") == "execute"
