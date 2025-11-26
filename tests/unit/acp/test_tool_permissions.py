"""Unit tests for ACP tool permissions."""

from pathlib import Path

import pytest

from fast_agent.acp.tool_permissions import (
    PermissionFileStore,
    ToolPermissionRequest,
    ToolPermissionResponse,
    infer_tool_kind,
)


class TestPermissionFileStore:
    """Tests for PermissionFileStore class."""

    def test_get_permission_returns_none_for_unknown_tool(self, tmp_path: Path) -> None:
        """Test that get_permission returns None for unknown tools."""
        store = PermissionFileStore(tmp_path)
        result = store.get_permission("unknown_tool", "unknown_server")
        assert result is None

    def test_set_and_get_allowed_permission(self, tmp_path: Path) -> None:
        """Test setting and getting an allowed permission."""
        store = PermissionFileStore(tmp_path)
        store.set_permission("read_file", "fs_server", allowed=True)

        result = store.get_permission("read_file", "fs_server")
        assert result is True

    def test_set_and_get_rejected_permission(self, tmp_path: Path) -> None:
        """Test setting and getting a rejected permission."""
        store = PermissionFileStore(tmp_path)
        store.set_permission("delete_file", "fs_server", allowed=False)

        result = store.get_permission("delete_file", "fs_server")
        assert result is False

    def test_permission_persists_to_file(self, tmp_path: Path) -> None:
        """Test that permissions are persisted to the auths.md file."""
        store1 = PermissionFileStore(tmp_path)
        store1.set_permission("read_file", "server1", allowed=True)
        store1.set_permission("write_file", "server1", allowed=False)

        # Create a new store instance to force re-reading from file
        store2 = PermissionFileStore(tmp_path)

        assert store2.get_permission("read_file", "server1") is True
        assert store2.get_permission("write_file", "server1") is False

    def test_permission_file_created_only_when_needed(self, tmp_path: Path) -> None:
        """Test that the auths.md file is only created when a permission is saved."""
        store = PermissionFileStore(tmp_path)
        file_path = tmp_path / ".fast-agent" / "auths.md"

        # Reading should not create the file
        store.get_permission("some_tool", "some_server")
        assert not file_path.exists()

        # Setting a permission should create the file
        store.set_permission("some_tool", "some_server", allowed=True)
        assert file_path.exists()

    def test_overwrite_permission_changes_value(self, tmp_path: Path) -> None:
        """Test that setting a permission again overwrites the previous value."""
        store = PermissionFileStore(tmp_path)

        store.set_permission("tool", "server", allowed=True)
        assert store.get_permission("tool", "server") is True

        store.set_permission("tool", "server", allowed=False)
        assert store.get_permission("tool", "server") is False

    def test_clear_permission_removes_entry(self, tmp_path: Path) -> None:
        """Test that clearing a permission removes it."""
        store = PermissionFileStore(tmp_path)

        store.set_permission("tool", "server", allowed=True)
        assert store.get_permission("tool", "server") is True

        store.clear_permission("tool", "server")
        assert store.get_permission("tool", "server") is None

    def test_clear_all_removes_everything(self, tmp_path: Path) -> None:
        """Test that clear_all removes all permissions and the file."""
        store = PermissionFileStore(tmp_path)
        file_path = tmp_path / ".fast-agent" / "auths.md"

        store.set_permission("tool1", "server1", allowed=True)
        store.set_permission("tool2", "server2", allowed=False)
        assert file_path.exists()

        store.clear_all()

        assert store.get_permission("tool1", "server1") is None
        assert store.get_permission("tool2", "server2") is None
        assert not file_path.exists()

    def test_permission_key_format(self, tmp_path: Path) -> None:
        """Test that permission keys use server/tool format."""
        store = PermissionFileStore(tmp_path)
        key = store._get_permission_key("my_tool", "my_server")
        assert key == "my_server/my_tool"


class TestInferToolKind:
    """Tests for the infer_tool_kind function."""

    @pytest.mark.parametrize(
        "tool_name,expected_kind",
        [
            ("read_file", "read"),
            ("get_data", "read"),
            ("list_items", "read"),
            ("show_content", "read"),
            ("view_page", "read"),
            ("cat_file", "read"),
            ("write_file", "edit"),
            ("edit_document", "edit"),
            ("update_record", "edit"),
            ("modify_settings", "edit"),
            ("patch_config", "edit"),
            ("set_value", "edit"),
            ("delete_file", "delete"),
            ("remove_item", "delete"),
            ("rm_file", "delete"),
            ("move_file", "move"),
            ("rename_document", "move"),
            ("mv_item", "move"),
            ("copy_file", "move"),
            ("cp_data", "move"),
            ("search_content", "search"),
            ("find_files", "search"),
            ("grep_text", "search"),
            ("query_database", "search"),
            ("execute_command", "execute"),
            ("run_script", "execute"),
            ("exec_task", "execute"),
            ("shell_cmd", "execute"),
            ("bash_command", "execute"),
            ("think_about", "think"),
            ("plan_strategy", "think"),
            ("reason_through", "think"),
            ("analyze_data", "think"),
            ("fetch_url", "fetch"),
            ("download_file", "fetch"),
            ("http_request", "fetch"),
            ("request_data", "fetch"),
            ("unknown_action", "other"),
            ("custom_tool", "other"),
        ],
    )
    def test_infer_tool_kind(self, tool_name: str, expected_kind: str) -> None:
        """Test that tool kinds are correctly inferred from tool names."""
        assert infer_tool_kind(tool_name) == expected_kind


class TestToolPermissionDataClasses:
    """Tests for the data classes."""

    def test_tool_permission_request(self) -> None:
        """Test ToolPermissionRequest dataclass."""
        request = ToolPermissionRequest(
            tool_name="read_file",
            server_name="fs_server",
            arguments={"path": "/tmp/test.txt"},
            tool_call_id="call_123",
        )

        assert request.tool_name == "read_file"
        assert request.server_name == "fs_server"
        assert request.arguments == {"path": "/tmp/test.txt"}
        assert request.tool_call_id == "call_123"

    def test_tool_permission_request_optional_fields(self) -> None:
        """Test ToolPermissionRequest with optional fields."""
        request = ToolPermissionRequest(
            tool_name="list_files",
            server_name="server",
            arguments=None,
        )

        assert request.arguments is None
        assert request.tool_call_id is None

    def test_tool_permission_response(self) -> None:
        """Test ToolPermissionResponse dataclass."""
        response = ToolPermissionResponse(
            allowed=True,
            remember=True,
            cancelled=False,
        )

        assert response.allowed is True
        assert response.remember is True
        assert response.cancelled is False

    def test_tool_permission_response_defaults(self) -> None:
        """Test ToolPermissionResponse default values."""
        response = ToolPermissionResponse(allowed=False, remember=False)

        assert response.cancelled is False
