"""
Unit tests for ACP tool permission components.

Tests for:
- PermissionStore file persistence
- PermissionResult factory methods
- ACPToolPermissionManager (using test doubles)
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest
from acp.schema import AllowedOutcome, DeniedOutcome, RequestPermissionResponse

from fast_agent.acp.permission_store import (
    PermissionDecision,
    PermissionResult,
    PermissionStore,
)
from fast_agent.acp.tool_permission_adapter import ACPToolPermissionAdapter
from fast_agent.acp.tool_permissions import (
    ACPToolPermissionManager,
    _is_acp_tool_call_id,
    _permission_options,
)
from fast_agent.acp.tool_titles import build_tool_title

# =============================================================================
# Test Doubles for ACPToolPermissionManager Testing
# =============================================================================


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
            should_raise: If set, request_permission will raise this exception
        """
        self._responses = permission_responses or {}
        self._should_raise = should_raise
        self.permission_requests: list[Any] = []
        self.session_updates: list[dict[str, Any]] = []

    async def request_permission(
        self,
        options: Any = None,
        session_id: str = "",
        tool_call: Any = None,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        """Fake implementation that returns configured responses (new SDK kwargs style)."""
        # Store the call for assertions
        self.permission_requests.append(
            {
                "options": options,
                "session_id": session_id,
                "tool_call": tool_call,
            }
        )

        if self._should_raise:
            raise self._should_raise

        # Extract tool info from tool_call to determine response
        if tool_call:
            # Title may include args like "server/tool(arg=val)", extract base "server/tool"
            title = tool_call.title
            key = title.split("(")[0] if "(" in title else title
        else:
            key = "unknown"

        option_id = self._responses.get(key, "reject_once")
        if option_id == "cancelled":
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=option_id)
        )

    async def session_update(
        self,
        session_id: str = "",
        update: Any = None,
        **kwargs: Any,
    ) -> None:
        self.session_updates.append(
            {
                "session_id": session_id,
                "update": update,
                "kwargs": kwargs,
            }
        )


class FakeToolProgressManager:
    """Test double for ACP tool-call id lookup."""

    def __init__(self, mapping: dict[str, str | None] | None = None) -> None:
        self._mapping = mapping or {}
        self.lookups: list[str] = []

    async def get_tool_call_id_for_tool_use(self, tool_use_id: str) -> str | None:
        self.lookups.append(tool_use_id)
        return self._mapping.get(tool_use_id)


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
    async def test_persists_names_with_delimiters_across_instances(self, temp_dir: Path) -> None:
        """Server and tool names can contain markdown/key delimiters."""
        store1 = PermissionStore(cwd=temp_dir)
        await store1.set("org/server|alpha", "tools/fetch|url", PermissionDecision.ALLOW_ALWAYS)

        store2 = PermissionStore(cwd=temp_dir)

        assert (
            await store2.get("org/server|alpha", "tools/fetch|url")
            == PermissionDecision.ALLOW_ALWAYS
        )
        assert await store2.get("org/server", "alpha/tools/fetch|url") is None

    @pytest.mark.asyncio
    async def test_only_creates_file_when_permission_set(self, temp_dir: Path) -> None:
        """File is only created when first permission is set."""
        store = PermissionStore(cwd=temp_dir)

        # Initially, no file
        assert not store.file_path.exists()

        # Just reading doesn't create file
        await store.get("server1", "tool1")
        assert not store.file_path.exists()

        # Setting permission creates file
        await store.set("server1", "tool1", PermissionDecision.ALLOW_ALWAYS)
        assert store.file_path.exists()

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
        assert store.file_path.exists()

        # Remove
        removed = await store.remove("server1", "tool1")
        assert removed is True

        # Verify removed
        assert await store.get("server1", "tool1") is None
        assert not store.file_path.exists()

        fresh_store = PermissionStore(cwd=temp_dir)
        assert await fresh_store.get("server1", "tool1") is None

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
    async def test_file_format_is_human_readable(self, temp_dir: Path) -> None:
        """The permissions file is human-readable markdown."""
        store = PermissionStore(cwd=temp_dir)
        await store.set("my_server", "my_tool", PermissionDecision.ALLOW_ALWAYS)

        # Read the file content
        content = store.file_path.read_text()

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
        for i in range(10):
            assert await store.get(f"server{i}", f"tool{i}") == PermissionDecision.ALLOW_ALWAYS


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

        await store.set(
            "server-with-dashes", "tool_with_underscores", PermissionDecision.ALLOW_ALWAYS
        )

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

    @pytest.mark.asyncio
    async def test_skips_rows_with_blank_server_or_tool_names(self, temp_dir: Path) -> None:
        """Should ignore table rows that cannot identify a server/tool pair."""
        permissions_file = temp_dir / ".fast-agent" / "auths.md"
        permissions_file.parent.mkdir(parents=True)
        permissions_file.write_text(
            """# Permissions
| Server | Tool | Permission |
|--------|------|------------|
|        | tool1 | allow_always |
| server1 |      | reject_always |
| server2 | tool2 | allow_always |
"""
        )

        store = PermissionStore(cwd=temp_dir)

        assert await store.get("", "tool1") is None
        assert await store.get("server1", "") is None
        assert await store.get("server2", "tool2") == PermissionDecision.ALLOW_ALWAYS


class TestACPToolPermissionAdapter:
    """Tests for MCP permission-handler to ACP permission-manager adaptation."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_uses_acp_tool_call_id_from_progress_manager(self, temp_dir: Path) -> None:
        """Known progress ids should be forwarded to the permission prompt."""
        acp_tool_call_id = "0123456789abcdef0123456789abcdef"
        connection = FakeAgentSideConnection(permission_responses={"server1/tool1": "allow_once"})
        progress = FakeToolProgressManager({"provider-call-1": acp_tool_call_id})
        adapter = ACPToolPermissionAdapter(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
            tool_handler=progress,
        )

        result = await adapter.check_permission(
            "tool1",
            "server1",
            tool_use_id="provider-call-1",
        )

        assert result.allowed is True
        assert progress.lookups == ["provider-call-1"]
        tool_call = connection.permission_requests[0]["tool_call"]
        assert tool_call.toolCallId == acp_tool_call_id

    @pytest.mark.asyncio
    async def test_does_not_use_provider_tool_use_id_as_acp_tool_call_id(
        self, temp_dir: Path
    ) -> None:
        """Provider tool-use ids are not necessarily valid ACP tool-call ids."""
        connection = FakeAgentSideConnection(permission_responses={"server1/tool1": "allow_once"})
        progress = FakeToolProgressManager({"provider-call-1": None})
        adapter = ACPToolPermissionAdapter(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
            tool_handler=progress,
        )

        result = await adapter.check_permission(
            "tool1",
            "server1",
            tool_use_id="provider-call-1",
        )

        assert result.allowed is True
        assert progress.lookups == ["provider-call-1"]
        tool_call = connection.permission_requests[0]["tool_call"]
        assert tool_call.toolCallId == "pending"


def test_is_acp_tool_call_id_accepts_only_32_hex_chars() -> None:
    assert _is_acp_tool_call_id("0123456789abcdef0123456789abcdef")
    assert _is_acp_tool_call_id("0123456789ABCDEF0123456789ABCDEF")
    assert not _is_acp_tool_call_id(None)
    assert not _is_acp_tool_call_id("provider-call-1")
    assert not _is_acp_tool_call_id("g" * 32)
    assert not _is_acp_tool_call_id("0" * 31)


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
        connection = FakeAgentSideConnection(permission_responses={"server1/tool1": "allow_once"})
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        arguments = {
            "prompt": "lion",
            "quality": "low",
            "tool_result": "image",
        }
        result = await manager.check_permission("tool1", "server1", arguments)

        assert result.allowed is True
        assert result.remember is False
        assert len(connection.permission_requests) == 1

        # Verify tool_call contains rawInput per ACP spec (now stored as dict)
        request = connection.permission_requests[0]
        assert request["tool_call"] is not None
        assert request["tool_call"].rawInput == arguments
        title = request["tool_call"].title
        assert title == "server1/tool1"
        assert "prompt=lion" not in title
        assert "tool_result=image" not in title

    def test_remembered_permission_options_name_tool_scope(self) -> None:
        """Remembered options should make tool-level scope explicit."""
        option_names = {option.option_id: option.name for option in _permission_options()}

        assert option_names["allow_always"] == "Always Allow This Tool"
        assert option_names["reject_always"] == "Never Allow This Tool"

    @pytest.mark.asyncio
    async def test_permission_request_uses_tool_scoped_remembered_labels(
        self,
        temp_dir: Path,
    ) -> None:
        connection = FakeAgentSideConnection(permission_responses={"server1/tool1": "allow_once"})
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        await manager.check_permission("tool1", "server1", {"path": "a.txt"})

        options = connection.permission_requests[0]["options"]
        option_names = {option.option_id: option.name for option in options}
        assert option_names["allow_always"] == "Always Allow This Tool"
        assert option_names["reject_always"] == "Never Allow This Tool"

    @pytest.mark.asyncio
    async def test_allow_always_is_tool_scoped_not_argument_scoped(self, temp_dir: Path) -> None:
        connection = FakeAgentSideConnection(permission_responses={"server1/tool1": "allow_always"})
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        first = await manager.check_permission("tool1", "server1", {"path": "a.txt"})
        second = await manager.check_permission("tool1", "server1", {"path": "b.txt"})

        assert first.allowed is True
        assert second.allowed is True
        assert len(connection.permission_requests) == 1

    @pytest.mark.asyncio
    async def test_long_arguments_stay_in_raw_input_not_title(self, temp_dir: Path) -> None:
        """Long arguments should stay available without bloating the title."""
        connection = FakeAgentSideConnection(permission_responses={"server1/tool1": "allow_once"})
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        long_value = "a" * 120
        arguments = {"payload": long_value}

        result = await manager.check_permission("tool1", "server1", arguments)

        assert result.allowed is True
        request = connection.permission_requests[0]
        title = request["tool_call"].title
        assert title == "server1/tool1"
        assert request["tool_call"].rawInput == arguments

    @pytest.mark.asyncio
    async def test_builtin_server_omits_server_name_in_title(self, temp_dir: Path) -> None:
        """Built-in ACP tools should omit the server name in titles."""
        connection = FakeAgentSideConnection(permission_responses={"execute": "allow_once"})
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        arguments = {"command": "ls"}
        result = await manager.check_permission("execute", "acp_terminal", arguments)

        assert result.allowed is True
        request = connection.permission_requests[0]
        title = request["tool_call"].title
        assert title == "execute"
        assert "acp_terminal" not in title
        assert request["tool_call"].rawInput == arguments

    @pytest.mark.asyncio
    async def test_persists_allow_always_to_store(self, temp_dir: Path) -> None:
        """Should persist allow_always decisions."""
        connection = FakeAgentSideConnection(permission_responses={"server1/tool1": "allow_always"})
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
        connection = FakeAgentSideConnection(permission_responses={"server1/tool1": "cancelled"})
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
        connection = FakeAgentSideConnection(should_raise=Exception("Connection failed"))
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
        connection = FakeAgentSideConnection(permission_responses={"server1/tool1": "allow_always"})
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
    async def test_session_cache_distinguishes_names_with_slashes(self, temp_dir: Path) -> None:
        """Session cache keys should not collide when names contain slashes."""
        connection = FakeAgentSideConnection(
            permission_responses={"org/server/fetch": "allow_always"}
        )
        manager = ACPToolPermissionManager(
            connection=connection,
            session_id="test-session",
            cwd=temp_dir,
        )

        first = await manager.check_permission("fetch", "org/server")
        second = await manager.check_permission("server/fetch", "org")

        assert first.allowed is True
        assert second.allowed is True
        assert len(connection.permission_requests) == 2

    @pytest.mark.asyncio
    async def test_clears_session_cache(self, temp_dir: Path) -> None:
        """Should be able to clear session cache."""
        connection = FakeAgentSideConnection(permission_responses={"server1/tool1": "allow_always"})
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
        connection = FakeAgentSideConnection(permission_responses={"server1/tool1": "reject_once"})
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
        connection = FakeAgentSideConnection(permission_responses={"server1/tool1": "allow_once"})
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


def test_build_tool_title_strips_line_breaks() -> None:
    """Tool titles should strip CR/LF characters for display."""
    title = build_tool_title(
        tool_name="do\nthing",
        server_name="server\r",
    )
    assert "\n" not in title
    assert "\r" not in title
