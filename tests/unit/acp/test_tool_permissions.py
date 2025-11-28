"""
Unit tests for ACP tool permission components.

Tests for:
- PermissionStore file persistence
- PermissionResult factory methods
- _infer_tool_kind function
- NoOpToolPermissionChecker
- ACPToolPermissionManager (using test doubles)
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest

from fast_agent.acp.permission_store import (
    DEFAULT_PERMISSIONS_FILE,
    PermissionDecision,
    PermissionResult,
    PermissionStore,
)
from fast_agent.acp.tool_permissions import (
    ACPToolPermissionManager,
    NoOpToolPermissionChecker,
    ToolPermissionChecker,
    _infer_tool_kind,
)


# =============================================================================
# Test Doubles for ACPToolPermissionManager Testing
# =============================================================================


class FakeOutcome:
    """Fake outcome object matching ACP schema structure."""

    def __init__(self, outcome: str, optionId: str | None = None):
        self.outcome = outcome
        self.optionId = optionId


class FakePermissionResponse:
    """Fake response matching ACP RequestPermissionResponse structure."""

    def __init__(self, option_id: str):
        if option_id == "cancelled":
            self.outcome = FakeOutcome(outcome="cancelled", optionId=None)
        else:
            self.outcome = FakeOutcome(outcome="selected", optionId=option_id)


class FakeAgentSideConnection:
    """
    Test double for AgentSideConnection.

    Configure responses via constructor, then use in tests.
    No mocking - this is a real class designed for testing.
    """

    def __init__(
        self,
        permission_responses: dict[str, str] | None = None,
        should_raise: Exception | None = None,
    ):
        """
        Args:
            permission_responses: Map of "server/tool" -> option_id response
                                  e.g., {"server1/tool1": "allow_always"}
            should_raise: If set, requestPermission will raise this exception
        """
        self._responses = permission_responses or {}
        self._should_raise = should_raise
        self.permission_requests: list[Any] = []

    async def requestPermission(self, request: Any) -> FakePermissionResponse:
        """Fake implementation that returns configured responses."""
        self.permission_requests.append(request)

        if self._should_raise:
            raise self._should_raise

        # Extract tool info from request to determine response
        tool_call = request.toolCall
        if tool_call:
            key = tool_call.title  # "server/tool" format
        else:
            key = "unknown"

        option_id = self._responses.get(key, "reject_once")
        return FakePermissionResponse(option_id)


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


class TestPermissionStoreEdgeCases:
    """Edge case tests for PermissionStore using real file system."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_handles_malformed_markdown_file(self, temp_dir: Path) -> None:
        """Should handle malformed markdown gracefully without crashing."""
        permissions_file = temp_dir / ".fast-agent" / "auths.md"
        permissions_file.parent.mkdir(parents=True)
        permissions_file.write_text("this is not valid markdown table format\nrandom text")

        store = PermissionStore(cwd=temp_dir)
        result = await store.get("server1", "tool1")

        assert result is None  # Should not crash, just return None

    @pytest.mark.asyncio
    async def test_handles_invalid_permission_values(self, temp_dir: Path) -> None:
        """Should skip invalid permission values in file."""
        permissions_file = temp_dir / ".fast-agent" / "auths.md"
        permissions_file.parent.mkdir(parents=True)
        permissions_file.write_text(
            """# Permissions
| Server | Tool | Permission |
|--------|------|------------|
| server1 | tool1 | invalid_value |
| server2 | tool2 | allow_always |
"""
        )

        store = PermissionStore(cwd=temp_dir)

        # Invalid value should be skipped
        result1 = await store.get("server1", "tool1")
        assert result1 is None

        # Valid value should be loaded
        result2 = await store.get("server2", "tool2")
        assert result2 == PermissionDecision.ALLOW_ALWAYS

    @pytest.mark.asyncio
    async def test_handles_empty_file(self, temp_dir: Path) -> None:
        """Should handle empty permissions file."""
        permissions_file = temp_dir / ".fast-agent" / "auths.md"
        permissions_file.parent.mkdir(parents=True)
        permissions_file.write_text("")

        store = PermissionStore(cwd=temp_dir)
        result = await store.get("server1", "tool1")

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_file_with_only_headers(self, temp_dir: Path) -> None:
        """Should handle file with only table headers."""
        permissions_file = temp_dir / ".fast-agent" / "auths.md"
        permissions_file.parent.mkdir(parents=True)
        permissions_file.write_text(
            """# Permissions
| Server | Tool | Permission |
|--------|------|------------|
"""
        )

        store = PermissionStore(cwd=temp_dir)
        result = await store.get("server1", "tool1")

        assert result is None

    @pytest.mark.asyncio
    async def test_overwrites_existing_permission(self, temp_dir: Path) -> None:
        """Should overwrite existing permission for same server/tool."""
        store = PermissionStore(cwd=temp_dir)

        await store.set("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)
        await store.set("server1", "tool1", PermissionDecision.REJECT_ALWAYS)

        result = await store.get("server1", "tool1")
        assert result == PermissionDecision.REJECT_ALWAYS

    @pytest.mark.asyncio
    async def test_handles_special_characters_in_names(self, temp_dir: Path) -> None:
        """Should handle special characters in server/tool names."""
        store = PermissionStore(cwd=temp_dir)

        await store.set("server-with-dashes", "tool_with_underscores", PermissionDecision.ALLOW_ALWAYS)

        result = await store.get("server-with-dashes", "tool_with_underscores")
        assert result == PermissionDecision.ALLOW_ALWAYS

    @pytest.mark.asyncio
    async def test_handles_mixed_valid_invalid_rows(self, temp_dir: Path) -> None:
        """Should handle files with mix of valid and malformed rows."""
        permissions_file = temp_dir / ".fast-agent" / "auths.md"
        permissions_file.parent.mkdir(parents=True)
        permissions_file.write_text(
            """# Permissions
| Server | Tool | Permission |
|--------|------|------------|
| server1 | tool1 | allow_always |
| malformed row without pipes
| server2 | tool2 | reject_always |
| incomplete |
| server3 | tool3 | allow_always |
"""
        )

        store = PermissionStore(cwd=temp_dir)

        # Valid rows should be loaded
        assert await store.get("server1", "tool1") == PermissionDecision.ALLOW_ALWAYS
        assert await store.get("server2", "tool2") == PermissionDecision.REJECT_ALWAYS
        assert await store.get("server3", "tool3") == PermissionDecision.ALLOW_ALWAYS


class TestNoOpToolPermissionChecker:
    """Tests for NoOpToolPermissionChecker - always allows."""

    @pytest.mark.asyncio
    async def test_always_allows_any_tool(self) -> None:
        """Should always return allowed=True regardless of input."""
        checker = NoOpToolPermissionChecker()

        result = await checker.check_permission(
            tool_name="dangerous_delete_everything",
            server_name="any_server",
            arguments={"recursive": True, "force": True},
        )

        assert result.allowed is True
        assert result.remember is False

    @pytest.mark.asyncio
    async def test_allows_with_no_arguments(self) -> None:
        """Should allow when arguments are None."""
        checker = NoOpToolPermissionChecker()

        result = await checker.check_permission(
            tool_name="some_tool",
            server_name="some_server",
            arguments=None,
        )

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_allows_with_empty_arguments(self) -> None:
        """Should allow when arguments are empty dict."""
        checker = NoOpToolPermissionChecker()

        result = await checker.check_permission(
            tool_name="some_tool",
            server_name="some_server",
            arguments={},
        )

        assert result.allowed is True

    def test_implements_protocol(self) -> None:
        """Should implement ToolPermissionChecker protocol."""
        checker = NoOpToolPermissionChecker()
        assert isinstance(checker, ToolPermissionChecker)


class TestACPToolPermissionManager:
    """Tests for ACPToolPermissionManager using test doubles."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_uses_stored_allow_always_without_client_call(self, temp_dir: Path) -> None:
        """Should return allowed without calling client if store has allow_always."""
        # Pre-populate the store
        store = PermissionStore(cwd=temp_dir)
        await store.set("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)

        connection = FakeAgentSideConnection()
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            store=store,
        )

        result = await manager.check_permission("tool1", "server1")

        assert result.allowed is True
        assert len(connection.permission_requests) == 0  # No client call

    @pytest.mark.asyncio
    async def test_uses_stored_reject_always_without_client_call(self, temp_dir: Path) -> None:
        """Should return rejected without calling client if store has reject_always."""
        store = PermissionStore(cwd=temp_dir)
        await store.set("server1", "tool1", PermissionDecision.REJECT_ALWAYS)

        connection = FakeAgentSideConnection()
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            store=store,
        )

        result = await manager.check_permission("tool1", "server1")

        assert result.allowed is False
        assert len(connection.permission_requests) == 0

    @pytest.mark.asyncio
    async def test_requests_from_client_when_not_stored(self, temp_dir: Path) -> None:
        """Should call client when no stored decision exists."""
        connection = FakeAgentSideConnection(
            permission_responses={"server1/tool1": "allow_once"}
        )
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        result = await manager.check_permission("tool1", "server1", {"arg": "value"})

        assert result.allowed is True
        assert result.remember is False
        assert len(connection.permission_requests) == 1

    @pytest.mark.asyncio
    async def test_persists_allow_always_to_store(self, temp_dir: Path) -> None:
        """Should persist allow_always decisions."""
        connection = FakeAgentSideConnection(
            permission_responses={"server1/tool1": "allow_always"}
        )
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        result = await manager.check_permission("tool1", "server1")

        assert result.allowed is True
        assert result.remember is True

        # Verify persisted
        store = PermissionStore(cwd=temp_dir)
        stored = await store.get("server1", "tool1")
        assert stored == PermissionDecision.ALLOW_ALWAYS

    @pytest.mark.asyncio
    async def test_persists_reject_always_to_store(self, temp_dir: Path) -> None:
        """Should persist reject_always decisions."""
        connection = FakeAgentSideConnection(
            permission_responses={"server1/tool1": "reject_always"}
        )
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        result = await manager.check_permission("tool1", "server1")

        assert result.allowed is False
        assert result.remember is True

        # Verify persisted
        store = PermissionStore(cwd=temp_dir)
        stored = await store.get("server1", "tool1")
        assert stored == PermissionDecision.REJECT_ALWAYS

    @pytest.mark.asyncio
    async def test_handles_cancelled_response(self, temp_dir: Path) -> None:
        """Should handle cancelled permission requests."""
        connection = FakeAgentSideConnection(
            permission_responses={"server1/tool1": "cancelled"}
        )
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        result = await manager.check_permission("tool1", "server1")

        assert result.allowed is False
        assert result.is_cancelled is True

    @pytest.mark.asyncio
    async def test_fail_safe_denies_on_connection_error(self, temp_dir: Path) -> None:
        """FAIL-SAFE: Should DENY when client communication fails."""
        connection = FakeAgentSideConnection(
            should_raise=Exception("Connection failed")
        )
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        result = await manager.check_permission("tool1", "server1")

        assert result.allowed is False  # FAIL-SAFE

    @pytest.mark.asyncio
    async def test_session_cache_avoids_repeated_client_calls(self, temp_dir: Path) -> None:
        """Should cache allow_always in session to avoid repeated client calls."""
        connection = FakeAgentSideConnection(
            permission_responses={"server1/tool1": "allow_always"}
        )
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        # First call - goes to client
        await manager.check_permission("tool1", "server1")
        assert len(connection.permission_requests) == 1

        # Second call - should use cache (either session or store)
        await manager.check_permission("tool1", "server1")
        assert len(connection.permission_requests) == 1  # Still 1, not 2

    @pytest.mark.asyncio
    async def test_clears_session_cache(self, temp_dir: Path) -> None:
        """Should be able to clear session cache."""
        connection = FakeAgentSideConnection(
            permission_responses={"server1/tool1": "allow_always"}
        )
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        await manager.check_permission("tool1", "server1")
        await manager.clear_session_cache()

        # After clearing, should still use persisted store (not call client again)
        await manager.check_permission("tool1", "server1")
        assert len(connection.permission_requests) == 1  # Store has it

    @pytest.mark.asyncio
    async def test_reject_once_does_not_persist(self, temp_dir: Path) -> None:
        """reject_once should not be persisted to store."""
        connection = FakeAgentSideConnection(
            permission_responses={"server1/tool1": "reject_once"}
        )
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        result = await manager.check_permission("tool1", "server1")

        assert result.allowed is False
        assert result.remember is False

        # Verify NOT persisted
        store = PermissionStore(cwd=temp_dir)
        stored = await store.get("server1", "tool1")
        assert stored is None

    @pytest.mark.asyncio
    async def test_allow_once_does_not_persist(self, temp_dir: Path) -> None:
        """allow_once should not be persisted to store."""
        connection = FakeAgentSideConnection(
            permission_responses={"server1/tool1": "allow_once"}
        )
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        result = await manager.check_permission("tool1", "server1")

        assert result.allowed is True
        assert result.remember is False

        # Verify NOT persisted
        store = PermissionStore(cwd=temp_dir)
        stored = await store.get("server1", "tool1")
        assert stored is None
