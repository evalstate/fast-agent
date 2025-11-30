"""
Unit tests for ACPToolProgressManager chunking behavior.

Tests for:
- Basic stream event handling (start/delta)
- Chunk streaming without race conditions
"""

import asyncio
from typing import Any

import pytest

from fast_agent.acp.tool_progress import ACPToolProgressManager

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
    async def test_delta_events_only_notify_after_threshold(self) -> None:
        """Delta notifications are only sent after 20 chunks to reduce UI noise."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Send start
        manager.handle_tool_stream_event("start", {
            "tool_name": "server__read_file",
            "tool_use_id": "use-123",
        })

        # Send 19 deltas - should NOT trigger notifications
        for i in range(19):
            manager.handle_tool_stream_event("delta", {
                "tool_use_id": "use-123",
                "chunk": f"chunk{i}",
            })

        await asyncio.sleep(0.1)

        # Should only have start notification (no delta notifications yet)
        assert len(connection.notifications) == 1

        # Send 20th chunk - should trigger notification
        manager.handle_tool_stream_event("delta", {
            "tool_use_id": "use-123",
            "chunk": "chunk19",
        })

        await asyncio.sleep(0.1)

        # Now should have start + 1 delta notification
        assert len(connection.notifications) == 2

        delta_notification = connection.notifications[1]
        assert delta_notification.update.sessionUpdate == "tool_call_update"
        assert "(streaming: 20 chunks)" in delta_notification.update.title

        # rawInput should NOT be set during streaming
        assert delta_notification.update.rawInput is None

    @pytest.mark.asyncio
    async def test_delta_chunks_accumulate_correctly(self) -> None:
        """Multiple delta events should accumulate into a single content block."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Send start then multiple deltas (need 20+ to trigger notifications)
        manager.handle_tool_stream_event("start", {
            "tool_name": "server__read_file",
            "tool_use_id": "use-123",
        })

        # Send 20 chunks to reach notification threshold
        for i in range(20):
            manager.handle_tool_stream_event("delta", {
                "tool_use_id": "use-123",
                "chunk": f"chunk{i}_",
            })

        # Wait for async tasks to complete
        await asyncio.sleep(0.1)

        # Should have start + 1 delta notification (at chunk 20)
        assert len(connection.notifications) == 2

        # Delta notification should have accumulated content from all chunks
        delta_notification = connection.notifications[1]
        expected_content = "".join(f"chunk{i}_" for i in range(20))
        assert delta_notification.update.content[0].content.text == expected_content

        # Title should show 20 chunks
        assert "(streaming: 20 chunks)" in delta_notification.update.title

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

        # Send deltas immediately (no await between) - chunks are tracked even if not notified
        for i in range(5):
            manager.handle_tool_stream_event("delta", {
                "tool_use_id": "use-123",
                "chunk": f"chunk{i}",
            })

        # Wait for all async tasks
        await asyncio.sleep(0.1)

        # Should have 1 notification (start only, deltas below threshold)
        assert len(connection.notifications) == 1

        # But chunks should still be tracked internally
        assert manager._stream_chunk_counts.get("use-123") == 5

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

        # Send deltas to both (below threshold, so no notifications)
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

        # Should have 2 notifications: 2 starts only (deltas below threshold)
        assert len(connection.notifications) == 2

        # Verify both tools have their own external_id
        assert "use-a" in manager._stream_tool_use_ids
        assert "use-b" in manager._stream_tool_use_ids
        assert manager._stream_tool_use_ids["use-a"] != manager._stream_tool_use_ids["use-b"]

        # Verify chunks are tracked independently
        assert manager._stream_chunk_counts.get("use-a") == 1
        assert manager._stream_chunk_counts.get("use-b") == 1
