"""
Integration tests for ACP tool permissions.

Tests the full permission flow including:
- Permission requests sent to client
- allow_once/allow_always behavior
- reject_once/reject_always behavior
- Persistence across sessions
- --no-permissions flag behavior
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

from fast_agent.acp.tool_permissions import (
    PERMISSION_DIR_NAME,
    PERMISSION_FILE_NAME,
    ACPPermissionHandlerAdapter,
    ACPToolPermissionManager,
    PermissionDecision,
    PermissionResult,
    PermissionStore,
)


class TestACPPermissionIntegration:
    """Integration tests for ACP tool permission flow."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def test_client(self):
        """Create a test client instance."""
        return TestClient()

    @pytest.mark.asyncio
    async def test_permission_request_sent_when_tool_called(self, temp_dir, test_client):
        """Test that a permission request is sent to the client when a tool is called."""
        # Queue allow_once response
        test_client.queue_permission_selected("allow_once")

        # Create a mock connection that routes to our test client
        connection = MockAgentSideConnection(test_client)

        # Create permission manager
        manager = ACPToolPermissionManager(
            connection=connection,
            cwd=temp_dir,
            permissions_enabled=True,
        )

        # Request permission
        result = await manager.request_permission(
            session_id="test-session",
            tool_name="dangerous_tool",
            server_name="test_server",
            arguments={"path": "/etc/passwd"},
        )

        # Verify permission was granted
        assert result.allowed is True
        assert result.remember is False  # allow_once doesn't persist

    @pytest.mark.asyncio
    async def test_allow_once_permits_without_persistence(self, temp_dir, test_client):
        """Test that allow_once permits execution without persisting the decision."""
        test_client.queue_permission_selected("allow_once")
        connection = MockAgentSideConnection(test_client)

        manager = ACPToolPermissionManager(
            connection=connection,
            cwd=temp_dir,
            permissions_enabled=True,
        )

        result = await manager.request_permission(
            session_id="test-session",
            tool_name="test_tool",
            server_name="test_server",
        )

        assert result.allowed is True

        # Check nothing was persisted
        store = PermissionStore(temp_dir)
        assert await store.get("test_server", "test_tool") is None

    @pytest.mark.asyncio
    async def test_allow_always_permits_and_persists(self, temp_dir, test_client):
        """Test that allow_always permits execution and persists the decision."""
        test_client.queue_permission_selected("allow_always")
        connection = MockAgentSideConnection(test_client)

        manager = ACPToolPermissionManager(
            connection=connection,
            cwd=temp_dir,
            permissions_enabled=True,
        )

        result = await manager.request_permission(
            session_id="test-session",
            tool_name="test_tool",
            server_name="test_server",
        )

        assert result.allowed is True
        assert result.remember is True

        # Check decision was persisted
        store = PermissionStore(temp_dir)
        decision = await store.get("test_server", "test_tool")
        assert decision == PermissionDecision.ALLOW_ALWAYS

    @pytest.mark.asyncio
    async def test_reject_once_blocks_without_persistence(self, temp_dir, test_client):
        """Test that reject_once blocks execution without persisting the decision."""
        test_client.queue_permission_selected("reject_once")
        connection = MockAgentSideConnection(test_client)

        manager = ACPToolPermissionManager(
            connection=connection,
            cwd=temp_dir,
            permissions_enabled=True,
        )

        result = await manager.request_permission(
            session_id="test-session",
            tool_name="test_tool",
            server_name="test_server",
        )

        assert result.allowed is False

        # Check nothing was persisted
        store = PermissionStore(temp_dir)
        assert await store.get("test_server", "test_tool") is None

    @pytest.mark.asyncio
    async def test_reject_always_blocks_and_persists(self, temp_dir, test_client):
        """Test that reject_always blocks execution and persists the decision."""
        test_client.queue_permission_selected("reject_always")
        connection = MockAgentSideConnection(test_client)

        manager = ACPToolPermissionManager(
            connection=connection,
            cwd=temp_dir,
            permissions_enabled=True,
        )

        result = await manager.request_permission(
            session_id="test-session",
            tool_name="test_tool",
            server_name="test_server",
        )

        assert result.allowed is False
        assert result.remember is True

        # Check decision was persisted
        store = PermissionStore(temp_dir)
        decision = await store.get("test_server", "test_tool")
        assert decision == PermissionDecision.REJECT_ALWAYS

    @pytest.mark.asyncio
    async def test_persisted_permission_skips_client_request(self, temp_dir, test_client):
        """Test that persisted permissions skip the client request."""
        connection = MockAgentSideConnection(test_client)

        # Pre-persist an allow_always decision
        store = PermissionStore(temp_dir)
        await store.set("test_server", "test_tool", PermissionDecision.ALLOW_ALWAYS)

        manager = ACPToolPermissionManager(
            connection=connection,
            cwd=temp_dir,
            permissions_enabled=True,
        )

        # Should use persisted permission without asking client
        result = await manager.request_permission(
            session_id="test-session",
            tool_name="test_tool",
            server_name="test_server",
        )

        assert result.allowed is True
        # Client should not have been called
        assert len(test_client.permission_outcomes) == 0  # No outcomes consumed

    @pytest.mark.asyncio
    async def test_no_permissions_flag_disables_checks(self, temp_dir, test_client):
        """Test that permissions_enabled=False allows all without client requests."""
        connection = MockAgentSideConnection(test_client)

        manager = ACPToolPermissionManager(
            connection=connection,
            cwd=temp_dir,
            permissions_enabled=False,  # --no-permissions mode
        )

        result = await manager.request_permission(
            session_id="test-session",
            tool_name="dangerous_tool",
            server_name="any_server",
        )

        assert result.allowed is True
        # Client should not have been called
        assert len(test_client.permission_outcomes) == 0

    @pytest.mark.asyncio
    async def test_cancelled_permission_denies_execution(self, temp_dir, test_client):
        """Test that cancelled permission requests deny execution."""
        test_client.queue_permission_cancelled()
        connection = MockAgentSideConnection(test_client)

        manager = ACPToolPermissionManager(
            connection=connection,
            cwd=temp_dir,
            permissions_enabled=True,
        )

        result = await manager.request_permission(
            session_id="test-session",
            tool_name="test_tool",
            server_name="test_server",
        )

        assert result.allowed is False
        assert result.cancelled is True

    @pytest.mark.asyncio
    async def test_error_during_permission_check_denies(self, temp_dir):
        """Test that errors during permission check result in denial (fail-safe)."""
        # Create a connection that raises an error
        connection = ErroringMockConnection()

        manager = ACPToolPermissionManager(
            connection=connection,
            cwd=temp_dir,
            permissions_enabled=True,
        )

        result = await manager.request_permission(
            session_id="test-session",
            tool_name="test_tool",
            server_name="test_server",
        )

        # Fail-safe: deny on error
        assert result.allowed is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_permission_file_location(self, temp_dir, test_client):
        """Test that permissions are stored in the correct location."""
        test_client.queue_permission_selected("allow_always")
        connection = MockAgentSideConnection(test_client)

        manager = ACPToolPermissionManager(
            connection=connection,
            cwd=temp_dir,
            permissions_enabled=True,
        )

        await manager.request_permission(
            session_id="test-session",
            tool_name="test_tool",
            server_name="test_server",
        )

        # Check file was created in the right location
        expected_path = temp_dir / PERMISSION_DIR_NAME / PERMISSION_FILE_NAME
        assert expected_path.exists()


class TestACPPermissionHandlerIntegration:
    """Integration tests for ACPPermissionHandlerAdapter."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def test_client(self):
        """Create a test client instance."""
        return TestClient()

    @pytest.mark.asyncio
    async def test_adapter_integrates_with_manager(self, temp_dir, test_client):
        """Test that the adapter properly integrates with the permission manager."""
        test_client.queue_permission_selected("allow_once")
        connection = MockAgentSideConnection(test_client)

        manager = ACPToolPermissionManager(
            connection=connection,
            cwd=temp_dir,
            permissions_enabled=True,
        )

        adapter = ACPPermissionHandlerAdapter(
            permission_manager=manager,
            session_id="test-session",
        )

        # Use the adapter as would be done from MCP aggregator
        result = await adapter.check_permission(
            tool_name="test_tool",
            server_name="test_server",
            arguments={"key": "value"},
            tool_call_id="call-123",
        )

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_adapter_returns_denial_with_message(self, temp_dir, test_client):
        """Test that adapter returns appropriate error messages on denial."""
        test_client.queue_permission_selected("reject_once")
        connection = MockAgentSideConnection(test_client)

        manager = ACPToolPermissionManager(
            connection=connection,
            cwd=temp_dir,
            permissions_enabled=True,
        )

        adapter = ACPPermissionHandlerAdapter(
            permission_manager=manager,
            session_id="test-session",
        )

        result = await adapter.check_permission(
            tool_name="test_tool",
            server_name="test_server",
            arguments=None,
        )

        assert result.allowed is False
        assert result.error_message is not None
        assert "test_server" in result.error_message or "test_tool" in result.error_message


class MockAgentSideConnection:
    """
    Mock ACP connection that routes requests to a TestClient.

    This simulates the ACP connection for testing purposes.
    """

    def __init__(self, test_client: TestClient):
        self._client = test_client

    async def requestPermission(self, request):
        """Route permission request to test client."""
        return await self._client.requestPermission(request)


class ErroringMockConnection:
    """
    Mock connection that raises an error on any request.

    Used to test fail-safe behavior.
    """

    async def requestPermission(self, request):
        """Raise an error to test fail-safe behavior."""
        raise ConnectionError("Simulated connection failure")
