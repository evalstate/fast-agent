"""
Unit tests for ACPToolProgressManager chunking behavior.

Tests for:
- Basic stream event handling (start/delta)
- Chunk streaming without race conditions
- Permission pre-check integration
- Chunks sent while permission pending
"""

import asyncio
from typing import Any

import pytest

from fast_agent.acp.tool_progress import ACPToolProgressManager
from fast_agent.mcp.tool_permission_handler import ToolPermissionHandler, ToolPermissionResult

# =============================================================================
# Test Doubles
# =============================================================================


class FakeAgentSideConnection:
    """
    Test double for AgentSideConnection that captures sessionUpdate notifications.

    No mocking - this is a real class designed for testing.
    """

    def __init__(self):
        self.notifications: list[Any] = []

    async def sessionUpdate(self, notification: Any) -> None:
        """Capture notifications for assertions."""
        self.notifications.append(notification)


class FakePermissionHandler(ToolPermissionHandler):
    """
    Test double for ToolPermissionHandler.

    Allows configuring response, delay, and tracking calls.
    """

    def __init__(
        self,
        result: ToolPermissionResult | None = None,
        delay: float = 0,
    ):
        self._result = result or ToolPermissionResult(allowed=True)
        self._delay = delay
        self.calls: list[dict[str, Any]] = []

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> ToolPermissionResult:
        """Record the call and return configured result after optional delay."""
        self.calls.append({
            "tool_name": tool_name,
            "server_name": server_name,
            "arguments": arguments,
            "tool_use_id": tool_use_id,
        })
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return self._result


# =============================================================================
# Tests for ACPToolProgressManager
# =============================================================================


class TestACPToolProgressManager:
    """Tests for stream event handling and chunking behavior."""

    @pytest.mark.asyncio
    async def test_start_event_sends_notification(self) -> None:
        """Start event should send a tool_call notification with status=pending."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Send start event
        manager.handle_tool_stream_event("start", {
            "tool_name": "server__read_file",
            "tool_use_id": "use-123",
        })

        # Wait for async task to complete
        await asyncio.sleep(0.1)

        # Should have sent one notification
        assert len(connection.notifications) == 1
        notification = connection.notifications[0]

        # Verify it's a tool_call with pending status
        assert notification.update.sessionUpdate == "tool_call"
        assert notification.update.status == "pending"

    @pytest.mark.asyncio
    async def test_delta_event_sends_rawInput_chunk(self) -> None:
        """Delta event after start should send rawInput update with chunk."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Send start then delta
        manager.handle_tool_stream_event("start", {
            "tool_name": "server__read_file",
            "tool_use_id": "use-123",
        })
        manager.handle_tool_stream_event("delta", {
            "tool_use_id": "use-123",
            "chunk": '{"path": "/tmp',
        })

        # Wait for async tasks to complete
        await asyncio.sleep(0.1)

        # Should have two notifications: start + delta
        assert len(connection.notifications) == 2

        # Second notification should be an update with rawInput
        delta_notification = connection.notifications[1]
        assert delta_notification.update.sessionUpdate == "tool_call_update"
        assert delta_notification.update.rawInput == '{"path": "/tmp'

    @pytest.mark.asyncio
    async def test_delta_before_start_is_dropped(self) -> None:
        """Delta event without prior start should be dropped (no notification)."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Send delta without start
        manager.handle_tool_stream_event("delta", {
            "tool_use_id": "use-123",
            "chunk": '{"path": "/tmp',
        })

        # Wait for async task
        await asyncio.sleep(0.1)

        # No notifications should be sent
        assert len(connection.notifications) == 0

    @pytest.mark.asyncio
    async def test_external_id_set_synchronously_allows_immediate_deltas(self) -> None:
        """
        External ID should be set synchronously in handle_tool_stream_event,
        allowing delta events immediately after start without race condition.
        """
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Send start
        manager.handle_tool_stream_event("start", {
            "tool_name": "server__read_file",
            "tool_use_id": "use-123",
        })

        # external_id should be set IMMEDIATELY (synchronously)
        assert "use-123" in manager._stream_tool_use_ids

        # Send delta immediately (no await between)
        manager.handle_tool_stream_event("delta", {
            "tool_use_id": "use-123",
            "chunk": '{"path":',
        })
        manager.handle_tool_stream_event("delta", {
            "tool_use_id": "use-123",
            "chunk": ' "/tmp"}',
        })

        # Wait for all async tasks
        await asyncio.sleep(0.1)

        # Should have 3 notifications: start + 2 deltas
        assert len(connection.notifications) == 3

    @pytest.mark.asyncio
    async def test_permission_precheck_called_on_start(self) -> None:
        """When permission handler is set, check_permission should be called on start."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        permission_handler = FakePermissionHandler()
        manager.set_permission_handler(permission_handler)

        # Send start event with namespaced tool name
        manager.handle_tool_stream_event("start", {
            "tool_name": "myserver__read_file",
            "tool_use_id": "use-123",
        })

        # Wait for async task
        await asyncio.sleep(0.1)

        # Permission should have been checked
        assert len(permission_handler.calls) == 1
        assert permission_handler.calls[0]["tool_name"] == "read_file"
        assert permission_handler.calls[0]["server_name"] == "myserver"
        assert permission_handler.calls[0]["arguments"] is None

    @pytest.mark.asyncio
    async def test_chunks_sent_while_permission_pending(self) -> None:
        """
        Chunks should be sent even while permission check is still pending.
        Permission check is non-blocking.
        """
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Permission handler with delay to simulate slow response
        permission_handler = FakePermissionHandler(delay=0.5)
        manager.set_permission_handler(permission_handler)

        # Send start
        manager.handle_tool_stream_event("start", {
            "tool_name": "myserver__write_file",
            "tool_use_id": "use-456",
        })

        # Immediately send deltas (permission still pending)
        manager.handle_tool_stream_event("delta", {
            "tool_use_id": "use-456",
            "chunk": '{"content":',
        })
        manager.handle_tool_stream_event("delta", {
            "tool_use_id": "use-456",
            "chunk": ' "hello"}',
        })

        # Wait just enough for notifications (but not for full permission delay)
        await asyncio.sleep(0.15)

        # Should have received notifications for start + 2 deltas
        # even though permission check is still pending
        assert len(connection.notifications) >= 3

    @pytest.mark.asyncio
    async def test_multiple_tools_tracked_independently(self) -> None:
        """Multiple concurrent tool streams should be tracked independently."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Start two tools
        manager.handle_tool_stream_event("start", {
            "tool_name": "server__tool_a",
            "tool_use_id": "use-a",
        })
        manager.handle_tool_stream_event("start", {
            "tool_name": "server__tool_b",
            "tool_use_id": "use-b",
        })

        # Send deltas to both
        manager.handle_tool_stream_event("delta", {
            "tool_use_id": "use-a",
            "chunk": "chunk-a",
        })
        manager.handle_tool_stream_event("delta", {
            "tool_use_id": "use-b",
            "chunk": "chunk-b",
        })

        # Wait for async tasks
        await asyncio.sleep(0.1)

        # Should have 4 notifications: 2 starts + 2 deltas
        assert len(connection.notifications) == 4

        # Verify both tools have their own external_id
        assert "use-a" in manager._stream_tool_use_ids
        assert "use-b" in manager._stream_tool_use_ids
        assert manager._stream_tool_use_ids["use-a"] != manager._stream_tool_use_ids["use-b"]
