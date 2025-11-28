"""
Unit tests for ACP tool permissions.

Tests the permission store, permission results, tool kind inference,
and the ACPToolPermissionManager.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from fast_agent.acp.permission_store import (
    PermissionDecision,
    PermissionResult,
    PermissionStore,
    infer_tool_kind,
)
from fast_agent.acp.tool_permissions import (
    ACPToolPermissionManager,
    NoOpPermissionHandler,
    create_acp_permission_handler,
)


class TestPermissionResult:
    """Test PermissionResult factory methods and properties."""

    def test_allow_once(self):
        """Test allow_once factory method."""
        result = PermissionResult.allow_once()
        assert result.allowed is True
        assert result.remember is False
        assert result.cancelled is False

    def test_allow_always(self):
        """Test allow_always factory method."""
        result = PermissionResult.allow_always()
        assert result.allowed is True
        assert result.remember is True
        assert result.cancelled is False

    def test_reject_once(self):
        """Test reject_once factory method."""
        result = PermissionResult.reject_once()
        assert result.allowed is False
        assert result.remember is False
        assert result.cancelled is False

    def test_reject_always(self):
        """Test reject_always factory method."""
        result = PermissionResult.reject_always()
        assert result.allowed is False
        assert result.remember is True
        assert result.cancelled is False

    def test_cancelled(self):
        """Test cancelled factory method."""
        result = PermissionResult.cancelled_result()
        assert result.allowed is False
        assert result.remember is False
        assert result.cancelled is True

    def test_denied(self):
        """Test denied factory method (fail-safe)."""
        result = PermissionResult.denied()
        assert result.allowed is False
        assert result.remember is False
        assert result.cancelled is False


class TestInferToolKind:
    """Test tool kind inference from tool names."""

    def test_read_tools(self):
        """Test tools inferred as 'read'."""
        assert infer_tool_kind("read_file") == "read"
        assert infer_tool_kind("get_resource") == "read"
        assert infer_tool_kind("fetch_data") == "read"
        assert infer_tool_kind("list_files") == "read"
        assert infer_tool_kind("show_content") == "read"

    def test_edit_tools(self):
        """Test tools inferred as 'edit'."""
        assert infer_tool_kind("write_file") == "edit"
        assert infer_tool_kind("edit_document") == "edit"
        assert infer_tool_kind("update_config") == "edit"
        assert infer_tool_kind("modify_settings") == "edit"
        assert infer_tool_kind("patch_file") == "edit"

    def test_delete_tools(self):
        """Test tools inferred as 'delete'."""
        assert infer_tool_kind("delete_file") == "delete"
        assert infer_tool_kind("remove_item") == "delete"
        assert infer_tool_kind("clear_cache") == "delete"
        assert infer_tool_kind("clean_temp") == "delete"
        assert infer_tool_kind("rm_file") == "delete"

    def test_move_tools(self):
        """Test tools inferred as 'move'."""
        assert infer_tool_kind("move_file") == "move"
        assert infer_tool_kind("rename_document") == "move"
        assert infer_tool_kind("mv_item") == "move"

    def test_search_tools(self):
        """Test tools inferred as 'search'."""
        assert infer_tool_kind("search_code") == "search"
        assert infer_tool_kind("find_files") == "search"
        assert infer_tool_kind("query_database") == "search"
        assert infer_tool_kind("grep_pattern") == "search"

    def test_execute_tools(self):
        """Test tools inferred as 'execute'."""
        assert infer_tool_kind("execute_command") == "execute"
        assert infer_tool_kind("run_script") == "execute"
        assert infer_tool_kind("exec_process") == "execute"
        assert infer_tool_kind("command_line") == "execute"
        assert infer_tool_kind("bash_execute") == "execute"
        assert infer_tool_kind("shell_command") == "execute"

    def test_think_tools(self):
        """Test tools inferred as 'think'."""
        assert infer_tool_kind("think_step") == "think"
        assert infer_tool_kind("plan_action") == "think"
        assert infer_tool_kind("reason_about") == "think"

    def test_fetch_tools(self):
        """Test tools inferred as 'fetch'."""
        # Note: fetch is also matched by read pattern, read comes first
        assert infer_tool_kind("download_file") == "fetch"
        assert infer_tool_kind("http_request") == "fetch"

    def test_other_tools(self):
        """Test unknown tools return 'other'."""
        assert infer_tool_kind("unknown_tool") == "other"
        assert infer_tool_kind("custom_action") == "other"
        assert infer_tool_kind("something_random") == "other"


class TestPermissionStore:
    """Test PermissionStore persistence and retrieval."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_tools(self, temp_dir):
        """Test that check returns None for tools without stored permissions."""
        store = PermissionStore(temp_dir)
        result = await store.check("server", "unknown_tool")
        assert result is None

    @pytest.mark.asyncio
    async def test_store_and_retrieve_allow_always(self, temp_dir):
        """Test storing and retrieving allow_always permission."""
        store = PermissionStore(temp_dir)

        await store.store("test_server", "test_tool", PermissionDecision.ALLOW_ALWAYS)
        result = await store.check("test_server", "test_tool")

        assert result == PermissionDecision.ALLOW_ALWAYS

    @pytest.mark.asyncio
    async def test_store_and_retrieve_reject_always(self, temp_dir):
        """Test storing and retrieving reject_always permission."""
        store = PermissionStore(temp_dir)

        await store.store("test_server", "test_tool", PermissionDecision.REJECT_ALWAYS)
        result = await store.check("test_server", "test_tool")

        assert result == PermissionDecision.REJECT_ALWAYS

    @pytest.mark.asyncio
    async def test_persists_across_instances(self, temp_dir):
        """Test that permissions persist across store instances (file I/O)."""
        # First store instance
        store1 = PermissionStore(temp_dir)
        await store1.store("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)
        await store1.store("server2", "tool2", PermissionDecision.REJECT_ALWAYS)

        # New store instance should load persisted data
        store2 = PermissionStore(temp_dir)
        result1 = await store2.check("server1", "tool1")
        result2 = await store2.check("server2", "tool2")

        assert result1 == PermissionDecision.ALLOW_ALWAYS
        assert result2 == PermissionDecision.REJECT_ALWAYS

    @pytest.mark.asyncio
    async def test_only_creates_file_when_permission_set(self, temp_dir):
        """Test that the permissions file is only created when a permission is stored."""
        permissions_path = temp_dir / ".fast-agent" / "auths.md"

        # Just creating the store shouldn't create the file
        store = PermissionStore(temp_dir)
        await store.check("server", "tool")  # This triggers load
        assert not permissions_path.exists()

        # Storing a permission should create the file
        await store.store("server", "tool", PermissionDecision.ALLOW_ALWAYS)
        assert permissions_path.exists()

    @pytest.mark.asyncio
    async def test_handles_missing_file_gracefully(self, temp_dir):
        """Test that missing file is handled gracefully."""
        store = PermissionStore(temp_dir)

        # Should not raise and should return None
        result = await store.check("server", "tool")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear_specific_permission(self, temp_dir):
        """Test clearing a specific permission."""
        store = PermissionStore(temp_dir)

        await store.store("server", "tool1", PermissionDecision.ALLOW_ALWAYS)
        await store.store("server", "tool2", PermissionDecision.REJECT_ALWAYS)

        await store.clear("server", "tool1")

        assert await store.check("server", "tool1") is None
        assert await store.check("server", "tool2") == PermissionDecision.REJECT_ALWAYS

    @pytest.mark.asyncio
    async def test_clear_all_server_permissions(self, temp_dir):
        """Test clearing all permissions for a server."""
        store = PermissionStore(temp_dir)

        await store.store("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)
        await store.store("server1", "tool2", PermissionDecision.REJECT_ALWAYS)
        await store.store("server2", "tool3", PermissionDecision.ALLOW_ALWAYS)

        await store.clear("server1")

        assert await store.check("server1", "tool1") is None
        assert await store.check("server1", "tool2") is None
        assert await store.check("server2", "tool3") == PermissionDecision.ALLOW_ALWAYS

    @pytest.mark.asyncio
    async def test_no_persistence_without_working_directory(self):
        """Test that store works without persistence when no working_directory."""
        store = PermissionStore(None)

        # Should work in-memory only
        await store.store("server", "tool", PermissionDecision.ALLOW_ALWAYS)
        result = await store.check("server", "tool")
        assert result == PermissionDecision.ALLOW_ALWAYS


class TestNoOpPermissionHandler:
    """Test NoOpPermissionHandler behavior."""

    @pytest.mark.asyncio
    async def test_always_allows(self):
        """Test that NoOpPermissionHandler always allows tool execution."""
        handler = NoOpPermissionHandler()

        result = await handler.check_permission("any_tool", "any_server", {"arg": "value"})

        assert result.allowed is True
        assert result.remember is False


class TestACPToolPermissionManager:
    """Test ACPToolPermissionManager behavior."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock ACP connection."""
        connection = MagicMock()
        connection.requestPermission = AsyncMock()
        return connection

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_uses_cached_permission(self, mock_connection, temp_dir):
        """Test that cached permissions are used without making requests."""
        manager = ACPToolPermissionManager(mock_connection, "session-1", temp_dir)

        # Pre-populate the session cache
        manager._session_cache["test_server/test_tool"] = PermissionDecision.ALLOW_ALWAYS

        result = await manager.check_permission("test_tool", "test_server")

        assert result.allowed is True
        # Should not have called requestPermission
        mock_connection.requestPermission.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_persisted_permission(self, mock_connection, temp_dir):
        """Test that persisted permissions are loaded and used."""
        # First, store a permission
        store = PermissionStore(temp_dir)
        await store.store("test_server", "test_tool", PermissionDecision.ALLOW_ALWAYS)

        # Create manager - should load persisted permission
        manager = ACPToolPermissionManager(mock_connection, "session-1", temp_dir)

        result = await manager.check_permission("test_tool", "test_server")

        assert result.allowed is True
        mock_connection.requestPermission.assert_not_called()

    @pytest.mark.asyncio
    async def test_requests_permission_when_not_cached(self, mock_connection, temp_dir):
        """Test that permission is requested when not cached."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.outcome = MagicMock()
        mock_response.outcome.outcome = "selected"
        mock_response.outcome.optionId = "allow_once"
        mock_connection.requestPermission.return_value = mock_response

        manager = ACPToolPermissionManager(mock_connection, "session-1", temp_dir)

        result = await manager.check_permission("test_tool", "test_server", {"arg": "value"})

        assert result.allowed is True
        assert result.remember is False
        mock_connection.requestPermission.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_allow_always_response(self, mock_connection, temp_dir):
        """Test handling of allow_always response."""
        mock_response = MagicMock()
        mock_response.outcome = MagicMock()
        mock_response.outcome.outcome = "selected"
        mock_response.outcome.optionId = "allow_always"
        mock_connection.requestPermission.return_value = mock_response

        manager = ACPToolPermissionManager(mock_connection, "session-1", temp_dir)

        result = await manager.check_permission("test_tool", "test_server")

        assert result.allowed is True
        assert result.remember is True

        # Should be cached for future calls
        mock_connection.requestPermission.reset_mock()
        result2 = await manager.check_permission("test_tool", "test_server")
        assert result2.allowed is True
        mock_connection.requestPermission.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_reject_once_response(self, mock_connection, temp_dir):
        """Test handling of reject_once response."""
        mock_response = MagicMock()
        mock_response.outcome = MagicMock()
        mock_response.outcome.outcome = "selected"
        mock_response.outcome.optionId = "reject_once"
        mock_connection.requestPermission.return_value = mock_response

        manager = ACPToolPermissionManager(mock_connection, "session-1", temp_dir)

        result = await manager.check_permission("test_tool", "test_server")

        assert result.allowed is False
        assert result.remember is False

    @pytest.mark.asyncio
    async def test_handles_reject_always_response(self, mock_connection, temp_dir):
        """Test handling of reject_always response."""
        mock_response = MagicMock()
        mock_response.outcome = MagicMock()
        mock_response.outcome.outcome = "selected"
        mock_response.outcome.optionId = "reject_always"
        mock_connection.requestPermission.return_value = mock_response

        manager = ACPToolPermissionManager(mock_connection, "session-1", temp_dir)

        result = await manager.check_permission("test_tool", "test_server")

        assert result.allowed is False
        assert result.remember is True

        # Should be cached for future calls
        mock_connection.requestPermission.reset_mock()
        result2 = await manager.check_permission("test_tool", "test_server")
        assert result2.allowed is False
        mock_connection.requestPermission.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_cancelled_response(self, mock_connection, temp_dir):
        """Test handling of cancelled response."""
        mock_response = MagicMock()
        mock_response.outcome = MagicMock()
        mock_response.outcome.outcome = "cancelled"
        mock_connection.requestPermission.return_value = mock_response

        manager = ACPToolPermissionManager(mock_connection, "session-1", temp_dir)

        result = await manager.check_permission("test_tool", "test_server")

        assert result.allowed is False
        assert result.cancelled is True

    @pytest.mark.asyncio
    async def test_fails_safe_on_error(self, mock_connection, temp_dir):
        """Test that errors result in DENY (fail-safe)."""
        mock_connection.requestPermission.side_effect = Exception("Connection error")

        manager = ACPToolPermissionManager(mock_connection, "session-1", temp_dir)

        result = await manager.check_permission("test_tool", "test_server")

        assert result.allowed is False
        assert result.remember is False

    @pytest.mark.asyncio
    async def test_fails_safe_on_unknown_response(self, mock_connection, temp_dir):
        """Test that unknown response format results in DENY."""
        mock_response = MagicMock()
        mock_response.outcome = MagicMock()
        mock_response.outcome.outcome = "unknown_outcome_type"
        mock_connection.requestPermission.return_value = mock_response

        manager = ACPToolPermissionManager(mock_connection, "session-1", temp_dir)

        result = await manager.check_permission("test_tool", "test_server")

        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_clear_session_cache(self, mock_connection, temp_dir):
        """Test clearing session cache."""
        manager = ACPToolPermissionManager(mock_connection, "session-1", temp_dir)
        manager._session_cache["test_server/test_tool"] = PermissionDecision.ALLOW_ALWAYS

        await manager.clear_session_cache()

        assert len(manager._session_cache) == 0


class TestCreateACPPermissionHandler:
    """Test the create_acp_permission_handler factory function."""

    @pytest.mark.asyncio
    async def test_creates_handler_that_delegates(self):
        """Test that the factory creates a handler that delegates to manager."""
        mock_connection = MagicMock()
        mock_response = MagicMock()
        mock_response.outcome = MagicMock()
        mock_response.outcome.outcome = "selected"
        mock_response.outcome.optionId = "allow_once"
        mock_connection.requestPermission = AsyncMock(return_value=mock_response)

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ACPToolPermissionManager(mock_connection, "session-1", temp_dir)
            handler = create_acp_permission_handler(manager)

            result = await handler.check_permission("tool", "server", {"arg": 1})

            assert result.allowed is True
            mock_connection.requestPermission.assert_called_once()
