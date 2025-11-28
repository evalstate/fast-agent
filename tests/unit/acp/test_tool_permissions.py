"""
Unit tests for ACP tool permissions.

Tests cover:
- PermissionStore: persistence, loading, saving, clearing
- PermissionResult: factory methods and properties
- _infer_tool_kind: tool name to kind mapping
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from fast_agent.acp.tool_permissions import (
    PERMISSION_DIR_NAME,
    PERMISSION_FILE_NAME,
    ACPPermissionHandlerAdapter,
    ACPToolPermissionManager,
    PermissionDecision,
    PermissionResult,
    PermissionStore,
    _infer_tool_kind,
)


class TestPermissionResult:
    """Tests for PermissionResult dataclass and factory methods."""

    def test_allow_once(self):
        """Test allow_once factory method."""
        result = PermissionResult.allow_once()
        assert result.allowed is True
        assert result.remember is False
        assert result.cancelled is False
        assert result.error is None

    def test_allow_always(self):
        """Test allow_always factory method."""
        result = PermissionResult.allow_always()
        assert result.allowed is True
        assert result.remember is True
        assert result.cancelled is False
        assert result.error is None

    def test_reject_once(self):
        """Test reject_once factory method."""
        result = PermissionResult.reject_once()
        assert result.allowed is False
        assert result.remember is False
        assert result.cancelled is False
        assert result.error is None

    def test_reject_always(self):
        """Test reject_always factory method."""
        result = PermissionResult.reject_always()
        assert result.allowed is False
        assert result.remember is True
        assert result.cancelled is False
        assert result.error is None

    def test_cancelled(self):
        """Test create_cancelled factory method."""
        result = PermissionResult.create_cancelled()
        assert result.allowed is False
        assert result.remember is False
        assert result.cancelled is True
        assert result.error is None

    def test_denied_with_error(self):
        """Test denied_with_error factory method."""
        error_msg = "Something went wrong"
        result = PermissionResult.denied_with_error(error_msg)
        assert result.allowed is False
        assert result.remember is False
        assert result.cancelled is False
        assert result.error == error_msg


class TestPermissionStore:
    """Tests for PermissionStore file persistence."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_get_returns_none_for_unknown_tools(self, temp_dir):
        """Test that get returns None for tools not in the store."""
        store = PermissionStore(temp_dir)
        result = await store.get("unknown_server", "unknown_tool")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_allow_always(self, temp_dir):
        """Test storing and retrieving allow_always permission."""
        store = PermissionStore(temp_dir)
        await store.set("my_server", "my_tool", PermissionDecision.ALLOW_ALWAYS)
        result = await store.get("my_server", "my_tool")
        assert result == PermissionDecision.ALLOW_ALWAYS

    @pytest.mark.asyncio
    async def test_set_and_get_reject_always(self, temp_dir):
        """Test storing and retrieving reject_always permission."""
        store = PermissionStore(temp_dir)
        await store.set("my_server", "my_tool", PermissionDecision.REJECT_ALWAYS)
        result = await store.get("my_server", "my_tool")
        assert result == PermissionDecision.REJECT_ALWAYS

    @pytest.mark.asyncio
    async def test_persists_across_instances(self, temp_dir):
        """Test that permissions persist across PermissionStore instances."""
        # Store a permission
        store1 = PermissionStore(temp_dir)
        await store1.set("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)

        # Create new instance and verify persistence
        store2 = PermissionStore(temp_dir)
        result = await store2.get("server1", "tool1")
        assert result == PermissionDecision.ALLOW_ALWAYS

    @pytest.mark.asyncio
    async def test_only_creates_file_when_permission_set(self, temp_dir):
        """Test that the file is only created when a permission is set."""
        store = PermissionStore(temp_dir)

        # Just getting should not create the file
        await store.get("server", "tool")
        assert not store.file_path.exists()

        # Setting should create the file
        await store.set("server", "tool", PermissionDecision.ALLOW_ALWAYS)
        assert store.file_path.exists()

    @pytest.mark.asyncio
    async def test_handles_missing_file_gracefully(self, temp_dir):
        """Test that a missing file is handled gracefully."""
        store = PermissionStore(temp_dir)
        # This should not raise an exception
        result = await store.get("any_server", "any_tool")
        assert result is None

    @pytest.mark.asyncio
    async def test_remove_permission(self, temp_dir):
        """Test removing a stored permission."""
        store = PermissionStore(temp_dir)
        await store.set("server", "tool", PermissionDecision.ALLOW_ALWAYS)

        # Verify it's stored
        result = await store.get("server", "tool")
        assert result == PermissionDecision.ALLOW_ALWAYS

        # Remove it
        await store.remove("server", "tool")

        # Verify it's gone
        result = await store.get("server", "tool")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear_all_permissions(self, temp_dir):
        """Test clearing all stored permissions."""
        store = PermissionStore(temp_dir)
        await store.set("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)
        await store.set("server2", "tool2", PermissionDecision.REJECT_ALWAYS)

        # Clear all
        await store.clear()

        # Verify all are gone
        assert await store.get("server1", "tool1") is None
        assert await store.get("server2", "tool2") is None

    @pytest.mark.asyncio
    async def test_file_path_location(self, temp_dir):
        """Test that file path is correctly computed."""
        store = PermissionStore(temp_dir)
        expected_path = temp_dir / PERMISSION_DIR_NAME / PERMISSION_FILE_NAME
        assert store.file_path == expected_path

    @pytest.mark.asyncio
    async def test_markdown_file_format(self, temp_dir):
        """Test that the file is written in markdown table format."""
        store = PermissionStore(temp_dir)
        await store.set("test_server", "test_tool", PermissionDecision.ALLOW_ALWAYS)

        content = store.file_path.read_text(encoding="utf-8")

        # Check markdown table structure
        assert "| Server | Tool | Permission |" in content
        assert "|--------|------|------------|" in content
        assert "| test_server | test_tool | allow_always |" in content

    @pytest.mark.asyncio
    async def test_multiple_permissions_stored(self, temp_dir):
        """Test storing multiple permissions."""
        store = PermissionStore(temp_dir)
        await store.set("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)
        await store.set("server1", "tool2", PermissionDecision.REJECT_ALWAYS)
        await store.set("server2", "tool1", PermissionDecision.ALLOW_ALWAYS)

        assert await store.get("server1", "tool1") == PermissionDecision.ALLOW_ALWAYS
        assert await store.get("server1", "tool2") == PermissionDecision.REJECT_ALWAYS
        assert await store.get("server2", "tool1") == PermissionDecision.ALLOW_ALWAYS


class TestInferToolKind:
    """Tests for _infer_tool_kind function."""

    def test_read_operations(self):
        """Test that read-like tool names are identified."""
        assert _infer_tool_kind("read_file") == "read"
        assert _infer_tool_kind("get_user") == "read"
        assert _infer_tool_kind("list_files") == "read"
        assert _infer_tool_kind("show_status") == "read"
        assert _infer_tool_kind("cat_file") == "read"
        assert _infer_tool_kind("head") == "read"
        assert _infer_tool_kind("tail") == "read"
        assert _infer_tool_kind("view_document") == "read"

    def test_edit_operations(self):
        """Test that edit-like tool names are identified."""
        assert _infer_tool_kind("write_file") == "edit"
        assert _infer_tool_kind("edit_document") == "edit"
        assert _infer_tool_kind("update_user") == "edit"
        assert _infer_tool_kind("modify_settings") == "edit"
        assert _infer_tool_kind("patch_config") == "edit"
        assert _infer_tool_kind("set_value") == "edit"
        assert _infer_tool_kind("create_file") == "edit"
        assert _infer_tool_kind("add_entry") == "edit"
        assert _infer_tool_kind("append_log") == "edit"

    def test_delete_operations(self):
        """Test that delete-like tool names are identified."""
        assert _infer_tool_kind("delete_file") == "delete"
        assert _infer_tool_kind("remove_user") == "delete"
        assert _infer_tool_kind("clear_cache") == "delete"
        assert _infer_tool_kind("clean_temp") == "delete"
        assert _infer_tool_kind("rm") == "delete"
        assert _infer_tool_kind("unlink_file") == "delete"
        assert _infer_tool_kind("drop_table") == "delete"

    def test_move_operations(self):
        """Test that move-like tool names are identified."""
        assert _infer_tool_kind("move_file") == "move"
        assert _infer_tool_kind("rename_document") == "move"
        assert _infer_tool_kind("mv") == "move"
        assert _infer_tool_kind("copy_file") == "move"
        assert _infer_tool_kind("cp") == "move"

    def test_search_operations(self):
        """Test that search-like tool names are identified."""
        assert _infer_tool_kind("search_files") == "search"
        assert _infer_tool_kind("find_user") == "search"
        assert _infer_tool_kind("query_database") == "search"
        assert _infer_tool_kind("grep") == "search"
        assert _infer_tool_kind("locate_file") == "search"
        assert _infer_tool_kind("lookup_value") == "search"

    def test_execute_operations(self):
        """Test that execute-like tool names are identified."""
        assert _infer_tool_kind("execute_command") == "execute"
        assert _infer_tool_kind("run_script") == "execute"
        assert _infer_tool_kind("exec") == "execute"
        assert _infer_tool_kind("command") == "execute"
        assert _infer_tool_kind("bash") == "execute"
        assert _infer_tool_kind("shell") == "execute"
        assert _infer_tool_kind("spawn_process") == "execute"
        # Note: call_api returns "fetch" since "api" is checked for fetch operations

    def test_think_operations(self):
        """Test that think-like tool names are identified."""
        assert _infer_tool_kind("think") == "think"
        assert _infer_tool_kind("plan_task") == "think"
        assert _infer_tool_kind("reason_about") == "think"
        assert _infer_tool_kind("analyze_data") == "think"
        assert _infer_tool_kind("evaluate_options") == "think"

    def test_fetch_operations(self):
        """Test that fetch-like tool names are identified."""
        assert _infer_tool_kind("fetch_url") == "fetch"
        assert _infer_tool_kind("download_file") == "fetch"
        assert _infer_tool_kind("http_get") == "fetch"
        assert _infer_tool_kind("request_data") == "fetch"
        assert _infer_tool_kind("curl") == "fetch"
        assert _infer_tool_kind("wget") == "fetch"
        assert _infer_tool_kind("api_call") == "fetch"

    def test_unknown_operations(self):
        """Test that unknown tool names return 'other'."""
        assert _infer_tool_kind("foo") == "other"
        assert _infer_tool_kind("bar_baz") == "other"
        assert _infer_tool_kind("my_custom_tool") == "other"
        assert _infer_tool_kind("xyz123") == "other"

    def test_case_insensitivity(self):
        """Test that matching is case insensitive."""
        assert _infer_tool_kind("READ_FILE") == "read"
        assert _infer_tool_kind("Write_File") == "edit"
        assert _infer_tool_kind("DELETE_USER") == "delete"


class TestACPToolPermissionManager:
    """Tests for ACPToolPermissionManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_connection(self):
        """Create a mock ACP connection."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_permissions_disabled_allows_all(self, temp_dir, mock_connection):
        """Test that when permissions are disabled, all tools are allowed."""
        manager = ACPToolPermissionManager(
            connection=mock_connection,
            cwd=temp_dir,
            permissions_enabled=False,
        )

        result = await manager.request_permission(
            session_id="test-session",
            tool_name="dangerous_tool",
            server_name="any_server",
        )

        assert result.allowed is True
        # Connection should not be called when permissions are disabled
        mock_connection.requestPermission.assert_not_called()

    @pytest.mark.asyncio
    async def test_stored_allow_always_grants_permission(self, temp_dir, mock_connection):
        """Test that stored allow_always grants permission without client request."""
        manager = ACPToolPermissionManager(
            connection=mock_connection,
            cwd=temp_dir,
            permissions_enabled=True,
        )

        # Pre-populate the store
        await manager._store.set("test_server", "test_tool", PermissionDecision.ALLOW_ALWAYS)

        result = await manager.request_permission(
            session_id="test-session",
            tool_name="test_tool",
            server_name="test_server",
        )

        assert result.allowed is True
        # Connection should not be called when we have stored permission
        mock_connection.requestPermission.assert_not_called()

    @pytest.mark.asyncio
    async def test_stored_reject_always_denies_permission(self, temp_dir, mock_connection):
        """Test that stored reject_always denies permission without client request."""
        manager = ACPToolPermissionManager(
            connection=mock_connection,
            cwd=temp_dir,
            permissions_enabled=True,
        )

        # Pre-populate the store
        await manager._store.set("test_server", "test_tool", PermissionDecision.REJECT_ALWAYS)

        result = await manager.request_permission(
            session_id="test-session",
            tool_name="test_tool",
            server_name="test_server",
        )

        assert result.allowed is False
        # Connection should not be called when we have stored permission
        mock_connection.requestPermission.assert_not_called()


class TestACPPermissionHandlerAdapter:
    """Tests for ACPPermissionHandlerAdapter."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock permission manager."""
        manager = MagicMock()
        manager.request_permission = AsyncMock()
        return manager

    @pytest.mark.asyncio
    async def test_adapter_converts_allow_to_check_result(self, mock_manager):
        """Test that adapter converts allow result to ToolPermissionCheckResult."""
        mock_manager.request_permission.return_value = PermissionResult.allow_once()

        adapter = ACPPermissionHandlerAdapter(
            permission_manager=mock_manager,
            session_id="test-session",
        )

        result = await adapter.check_permission(
            tool_name="test_tool",
            server_name="test_server",
            arguments={"arg": "value"},
        )

        assert result.allowed is True
        mock_manager.request_permission.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_converts_deny_to_check_result(self, mock_manager):
        """Test that adapter converts deny result to ToolPermissionCheckResult."""
        mock_manager.request_permission.return_value = PermissionResult.reject_once()

        adapter = ACPPermissionHandlerAdapter(
            permission_manager=mock_manager,
            session_id="test-session",
        )

        result = await adapter.check_permission(
            tool_name="test_tool",
            server_name="test_server",
            arguments=None,
        )

        assert result.allowed is False
        assert "denied" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_adapter_handles_cancelled(self, mock_manager):
        """Test that adapter handles cancelled permissions."""
        mock_manager.request_permission.return_value = PermissionResult.create_cancelled()

        adapter = ACPPermissionHandlerAdapter(
            permission_manager=mock_manager,
            session_id="test-session",
        )

        result = await adapter.check_permission(
            tool_name="test_tool",
            server_name="test_server",
            arguments=None,
        )

        assert result.allowed is False
        assert "cancelled" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_adapter_handles_error(self, mock_manager):
        """Test that adapter handles error in permission result."""
        mock_manager.request_permission.return_value = PermissionResult.denied_with_error(
            "Connection timeout"
        )

        adapter = ACPPermissionHandlerAdapter(
            permission_manager=mock_manager,
            session_id="test-session",
        )

        result = await adapter.check_permission(
            tool_name="test_tool",
            server_name="test_server",
            arguments=None,
        )

        assert result.allowed is False
        assert "Connection timeout" in result.error_message


class TestACPTerminalRuntimePermissions:
    """Tests for permission checking in ACPTerminalRuntime."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock ACP connection."""
        connection = MagicMock()
        connection._conn = MagicMock()
        connection._conn.send_request = AsyncMock()
        return connection

    @pytest.fixture
    def mock_permission_handler(self):
        """Create a mock permission handler."""
        from fast_agent.mcp.tool_execution_handler import ToolPermissionCheckResult

        handler = MagicMock()
        handler.check_permission = AsyncMock(return_value=ToolPermissionCheckResult.allow())
        return handler

    @pytest.mark.asyncio
    async def test_execute_without_permission_handler(self, mock_connection):
        """Test that execute works without a permission handler."""
        from fast_agent.acp.terminal_runtime import ACPTerminalRuntime

        # Set up the mock to return terminal creation and wait results
        mock_connection._conn.send_request.side_effect = [
            {"terminalId": "term-123"},  # terminal/create
            {"exitCode": 0},  # terminal/wait_for_exit
            {"output": "Hello, World!", "truncated": False},  # terminal/output
            None,  # terminal/release
        ]

        runtime = ACPTerminalRuntime(
            connection=mock_connection,
            session_id="test-session",
            activation_reason="test",
            permission_handler=None,
        )

        result = await runtime.execute({"command": "echo 'Hello, World!'"})

        # Should have executed successfully
        assert result.isError is False
        assert "Hello, World!" in result.content[0].text

    @pytest.mark.asyncio
    async def test_execute_with_permission_granted(self, mock_connection, mock_permission_handler):
        """Test that execute works when permission is granted."""
        from fast_agent.acp.terminal_runtime import ACPTerminalRuntime

        # Set up the mock to return terminal creation and wait results
        mock_connection._conn.send_request.side_effect = [
            {"terminalId": "term-123"},  # terminal/create
            {"exitCode": 0},  # terminal/wait_for_exit
            {"output": "Success", "truncated": False},  # terminal/output
            None,  # terminal/release
        ]

        runtime = ACPTerminalRuntime(
            connection=mock_connection,
            session_id="test-session",
            activation_reason="test",
            permission_handler=mock_permission_handler,
        )

        result = await runtime.execute({"command": "echo 'Success'"})

        # Should have executed successfully
        assert result.isError is False
        # Permission handler should have been called
        mock_permission_handler.check_permission.assert_called_once()
        call_kwargs = mock_permission_handler.check_permission.call_args.kwargs
        assert call_kwargs["tool_name"] == "execute"
        assert call_kwargs["server_name"] == "acp_terminal"

    @pytest.mark.asyncio
    async def test_execute_with_permission_denied(self, mock_connection, mock_permission_handler):
        """Test that execute is blocked when permission is denied."""
        from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
        from fast_agent.mcp.tool_execution_handler import ToolPermissionCheckResult

        # Set permission handler to deny
        mock_permission_handler.check_permission.return_value = ToolPermissionCheckResult.deny(
            "Permission denied for execute tool"
        )

        runtime = ACPTerminalRuntime(
            connection=mock_connection,
            session_id="test-session",
            activation_reason="test",
            permission_handler=mock_permission_handler,
        )

        result = await runtime.execute({"command": "rm -rf /"})

        # Should have been denied
        assert result.isError is True
        assert "Permission denied" in result.content[0].text
        # Terminal commands should NOT have been called
        mock_connection._conn.send_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_permission_check_error_denies(self, mock_connection, mock_permission_handler):
        """Test that permission check errors result in denial (fail-safe)."""
        from fast_agent.acp.terminal_runtime import ACPTerminalRuntime

        # Make permission check raise an exception
        mock_permission_handler.check_permission.side_effect = Exception("Connection error")

        runtime = ACPTerminalRuntime(
            connection=mock_connection,
            session_id="test-session",
            activation_reason="test",
            permission_handler=mock_permission_handler,
        )

        result = await runtime.execute({"command": "echo 'test'"})

        # Should have been denied due to error (fail-safe)
        assert result.isError is True
        assert "Permission check failed" in result.content[0].text
        # Terminal commands should NOT have been called
        mock_connection._conn.send_request.assert_not_called()


class TestACPFilesystemRuntimePermissions:
    """Tests for permission checking in ACPFilesystemRuntime."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock ACP connection."""
        connection = MagicMock()
        connection.readTextFile = AsyncMock()
        connection.writeTextFile = AsyncMock()
        return connection

    @pytest.fixture
    def mock_permission_handler(self):
        """Create a mock permission handler."""
        from fast_agent.mcp.tool_execution_handler import ToolPermissionCheckResult

        handler = MagicMock()
        handler.check_permission = AsyncMock(return_value=ToolPermissionCheckResult.allow())
        return handler

    @pytest.mark.asyncio
    async def test_read_without_permission_handler(self, mock_connection):
        """Test that read works without a permission handler."""
        from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime

        # Set up the mock to return file content
        mock_response = MagicMock()
        mock_response.content = "File content here"
        mock_connection.readTextFile.return_value = mock_response

        runtime = ACPFilesystemRuntime(
            connection=mock_connection,
            session_id="test-session",
            activation_reason="test",
            permission_handler=None,
        )

        result = await runtime.read_text_file({"path": "/test/file.txt"})

        # Should have executed successfully
        assert result.isError is False
        assert "File content here" in result.content[0].text

    @pytest.mark.asyncio
    async def test_read_with_permission_granted(self, mock_connection, mock_permission_handler):
        """Test that read works when permission is granted."""
        from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime

        mock_response = MagicMock()
        mock_response.content = "Secret file content"
        mock_connection.readTextFile.return_value = mock_response

        runtime = ACPFilesystemRuntime(
            connection=mock_connection,
            session_id="test-session",
            activation_reason="test",
            permission_handler=mock_permission_handler,
        )

        result = await runtime.read_text_file({"path": "/etc/passwd"})

        # Should have executed successfully
        assert result.isError is False
        # Permission handler should have been called
        mock_permission_handler.check_permission.assert_called_once()
        call_kwargs = mock_permission_handler.check_permission.call_args.kwargs
        assert call_kwargs["tool_name"] == "read_text_file"
        assert call_kwargs["server_name"] == "acp_filesystem"

    @pytest.mark.asyncio
    async def test_read_with_permission_denied(self, mock_connection, mock_permission_handler):
        """Test that read is blocked when permission is denied."""
        from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
        from fast_agent.mcp.tool_execution_handler import ToolPermissionCheckResult

        # Set permission handler to deny
        mock_permission_handler.check_permission.return_value = ToolPermissionCheckResult.deny(
            "Permission denied for read_text_file tool"
        )

        runtime = ACPFilesystemRuntime(
            connection=mock_connection,
            session_id="test-session",
            activation_reason="test",
            permission_handler=mock_permission_handler,
        )

        result = await runtime.read_text_file({"path": "/etc/shadow"})

        # Should have been denied
        assert result.isError is True
        assert "Permission denied" in result.content[0].text
        # File read should NOT have been called
        mock_connection.readTextFile.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_with_permission_denied(self, mock_connection, mock_permission_handler):
        """Test that write is blocked when permission is denied."""
        from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
        from fast_agent.mcp.tool_execution_handler import ToolPermissionCheckResult

        # Set permission handler to deny
        mock_permission_handler.check_permission.return_value = ToolPermissionCheckResult.deny(
            "Permission denied for write_text_file tool"
        )

        runtime = ACPFilesystemRuntime(
            connection=mock_connection,
            session_id="test-session",
            activation_reason="test",
            permission_handler=mock_permission_handler,
        )

        result = await runtime.write_text_file({"path": "/etc/passwd", "content": "malicious"})

        # Should have been denied
        assert result.isError is True
        assert "Permission denied" in result.content[0].text
        # File write should NOT have been called
        mock_connection.writeTextFile.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_with_permission_granted(self, mock_connection, mock_permission_handler):
        """Test that write works when permission is granted."""
        from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime

        runtime = ACPFilesystemRuntime(
            connection=mock_connection,
            session_id="test-session",
            activation_reason="test",
            permission_handler=mock_permission_handler,
        )

        result = await runtime.write_text_file({"path": "/tmp/test.txt", "content": "Hello"})

        # Should have executed successfully
        assert result.isError is False
        # Permission handler should have been called
        mock_permission_handler.check_permission.assert_called_once()
        call_kwargs = mock_permission_handler.check_permission.call_args.kwargs
        assert call_kwargs["tool_name"] == "write_text_file"
        assert call_kwargs["server_name"] == "acp_filesystem"

    @pytest.mark.asyncio
    async def test_permission_check_error_denies_read(self, mock_connection, mock_permission_handler):
        """Test that permission check errors result in denial for read (fail-safe)."""
        from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime

        # Make permission check raise an exception
        mock_permission_handler.check_permission.side_effect = Exception("Connection error")

        runtime = ACPFilesystemRuntime(
            connection=mock_connection,
            session_id="test-session",
            activation_reason="test",
            permission_handler=mock_permission_handler,
        )

        result = await runtime.read_text_file({"path": "/test/file.txt"})

        # Should have been denied due to error (fail-safe)
        assert result.isError is True
        assert "Permission check failed" in result.content[0].text
        # File read should NOT have been called
        mock_connection.readTextFile.assert_not_called()

    @pytest.mark.asyncio
    async def test_permission_check_error_denies_write(self, mock_connection, mock_permission_handler):
        """Test that permission check errors result in denial for write (fail-safe)."""
        from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime

        # Make permission check raise an exception
        mock_permission_handler.check_permission.side_effect = Exception("Connection error")

        runtime = ACPFilesystemRuntime(
            connection=mock_connection,
            session_id="test-session",
            activation_reason="test",
            permission_handler=mock_permission_handler,
        )

        result = await runtime.write_text_file({"path": "/test/file.txt", "content": "test"})

        # Should have been denied due to error (fail-safe)
        assert result.isError is True
        assert "Permission check failed" in result.content[0].text
        # File write should NOT have been called
        mock_connection.writeTextFile.assert_not_called()
