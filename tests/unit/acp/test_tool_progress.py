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
    async def test_delta_event_sends_content_with_accumulated_chunk(self) -> None:
        """Delta event after start should accumulate chunk into content and update title."""
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

        # Second notification should be an update with content (not rawInput)
        delta_notification = connection.notifications[1]
        assert delta_notification.update.sessionUpdate == "tool_call_update"

        # Content should contain accumulated text
        assert delta_notification.update.content is not None
        assert len(delta_notification.update.content) == 1
        assert delta_notification.update.content[0].type == "content"
        assert delta_notification.update.content[0].content.text == '{"path": "/tmp'

        # Title should include chunk count
        assert "(streaming: 1 chunks)" in delta_notification.update.title

        # rawInput should NOT be set during streaming
        assert delta_notification.update.rawInput is None

    @pytest.mark.asyncio
    async def test_delta_chunks_accumulate_correctly(self) -> None:
        """Multiple delta events should accumulate into a single content block."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Send start then multiple deltas
        manager.handle_tool_stream_event("start", {
            "tool_name": "server__read_file",
            "tool_use_id": "use-123",
        })
        manager.handle_tool_stream_event("delta", {
            "tool_use_id": "use-123",
            "chunk": '{"path":',
        })
        manager.handle_tool_stream_event("delta", {
            "tool_use_id": "use-123",
            "chunk": ' "/tmp/file.txt"}',
        })

        # Wait for async tasks to complete
        await asyncio.sleep(0.1)

        # Should have three notifications: start + 2 deltas
        assert len(connection.notifications) == 3

        # Third notification should have accumulated content from both chunks
        final_notification = connection.notifications[2]
        assert final_notification.update.content[0].content.text == '{"path": "/tmp/file.txt"}'

        # Title should show 2 chunks
        assert "(streaming: 2 chunks)" in final_notification.update.title

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
