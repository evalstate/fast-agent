"""
Unit tests for ACP Tool Call Tracker.
"""

import pytest
from acp.helpers import session_notification
from acp.schema import SessionUpdate, ToolCall

from fast_agent.acp.tool_call_tracker import ToolCallTracker


class MockConnection:
    """Mock ACP connection for testing."""

    def __init__(self):
        self.notifications = []

    async def sessionUpdate(self, notification):
        """Record session updates."""
        self.notifications.append(notification)


@pytest.mark.asyncio
async def test_create_tool_call():
    """Test creating a tool call sends the initial notification."""
    connection = MockConnection()
    tracker = ToolCallTracker(session_id="test-session", connection=connection)

    tool_call_id = await tracker.create_tool_call(
        title="Test Tool Call",
        kind="execute",
        tool_name="test_tool",
        arguments={"arg1": "value1"},
    )

    # Verify tool call ID was generated
    assert tool_call_id is not None
    assert isinstance(tool_call_id, str)

    # Verify notification was sent
    assert len(connection.notifications) == 1
    notification = connection.notifications[0]

    # Verify notification structure
    assert notification.sessionId == "test-session"
    assert hasattr(notification, "update")
    assert notification.update.sessionUpdate == "tool_call_update"
    assert hasattr(notification.update, "toolCall")

    tool_call = notification.update.toolCall
    assert tool_call.toolCallId == tool_call_id
    assert tool_call.title == "Test Tool Call"
    assert tool_call.kind == "execute"
    assert tool_call.status == "pending"


@pytest.mark.asyncio
async def test_update_tool_call():
    """Test updating a tool call sends update notifications."""
    connection = MockConnection()
    tracker = ToolCallTracker(session_id="test-session", connection=connection)

    # Create a tool call
    tool_call_id = await tracker.create_tool_call(
        title="Test Tool",
        kind="read",
        tool_name="test",
    )

    # Clear initial notification
    connection.notifications.clear()

    # Update to in_progress
    await tracker.update_tool_call(tool_call_id, status="in_progress")

    # Verify notification was sent
    assert len(connection.notifications) == 1
    notification = connection.notifications[0]
    assert notification.update.toolCall.status == "in_progress"


@pytest.mark.asyncio
async def test_complete_tool_call():
    """Test completing a tool call marks it as completed."""
    connection = MockConnection()
    tracker = ToolCallTracker(session_id="test-session", connection=connection)

    # Create and complete a tool call
    tool_call_id = await tracker.create_tool_call(
        title="Test Tool",
        kind="read",
        tool_name="test",
    )

    connection.notifications.clear()

    await tracker.complete_tool_call(
        tool_call_id=tool_call_id,
        content=[{"type": "text", "text": "result"}],
        is_error=False,
    )

    # Verify completion notification
    assert len(connection.notifications) == 1
    notification = connection.notifications[0]
    assert notification.update.toolCall.status == "completed"


@pytest.mark.asyncio
async def test_complete_tool_call_with_error():
    """Test completing a tool call with error marks it as failed."""
    connection = MockConnection()
    tracker = ToolCallTracker(session_id="test-session", connection=connection)

    # Create and fail a tool call
    tool_call_id = await tracker.create_tool_call(
        title="Test Tool",
        kind="execute",
        tool_name="test",
    )

    connection.notifications.clear()

    await tracker.complete_tool_call(
        tool_call_id=tool_call_id,
        content=[{"type": "text", "text": "error occurred"}],
        is_error=True,
    )

    # Verify failure notification
    assert len(connection.notifications) == 1
    notification = connection.notifications[0]
    assert notification.update.toolCall.status == "failed"


@pytest.mark.asyncio
async def test_tool_call_lifecycle():
    """Test complete tool call lifecycle: pending → in_progress → completed."""
    connection = MockConnection()
    tracker = ToolCallTracker(session_id="test-session", connection=connection)

    # Create tool call (pending)
    tool_call_id = await tracker.create_tool_call(
        title="Test Lifecycle",
        kind="other",
        tool_name="lifecycle_test",
    )

    # Update to in_progress
    await tracker.update_tool_call(tool_call_id, status="in_progress")

    # Complete
    await tracker.complete_tool_call(tool_call_id)

    # Verify we got 3 notifications
    assert len(connection.notifications) == 3

    # Verify status progression
    assert connection.notifications[0].update.toolCall.status == "pending"
    assert connection.notifications[1].update.toolCall.status == "in_progress"
    assert connection.notifications[2].update.toolCall.status == "completed"

    # Verify all notifications have the same tool call ID
    for notification in connection.notifications:
        assert notification.update.toolCall.toolCallId == tool_call_id


@pytest.mark.asyncio
async def test_update_unknown_tool_call():
    """Test that updating an unknown tool call logs a warning but doesn't crash."""
    connection = MockConnection()
    tracker = ToolCallTracker(session_id="test-session", connection=connection)

    # Try to update non-existent tool call
    await tracker.update_tool_call("unknown-id", status="completed")

    # Should not send any notifications
    assert len(connection.notifications) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
