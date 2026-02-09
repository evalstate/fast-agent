"""
Unit tests for ACPToolProgressManager chunking behavior.

Tests for:
- Basic stream event handling (start/delta)
- Chunk streaming without race conditions
"""

import asyncio
from typing import Any

import pytest
import pytest_asyncio

from fast_agent.acp.tool_call_context import acp_tool_call_context
from fast_agent.acp.tool_progress import ACPToolProgressManager


@pytest_asyncio.fixture(autouse=True)
async def cleanup_logging():
    yield
    from fast_agent.core.logging.logger import LoggingConfig
    from fast_agent.core.logging.transport import AsyncEventBus

    await LoggingConfig.shutdown()
    bus = AsyncEventBus._instance
    if bus is not None:
        bus_task = getattr(bus, "_task", None)
        await bus.stop()
        # bus.stop() is best-effort (it may swallow cancellation/timeouts). Ensure
        # the underlying processing task is fully awaited so pytest doesn't warn.
        if bus_task is not None and hasattr(bus_task, "done") and not bus_task.done():
            bus_task.cancel()
            await asyncio.gather(bus_task, return_exceptions=True)
    AsyncEventBus.reset()
    pending = []
    for task in asyncio.all_tasks():
        if task is asyncio.current_task():
            continue
        qn = getattr(task.get_coro(), "__qualname__", "")
        if "AsyncEventBus._process_events" in qn and not task.done():
            pending.append(task)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.sleep(0)
        await asyncio.gather(*pending, return_exceptions=True)

# =============================================================================
# Test Doubles
# =============================================================================


class FakeAgentSideConnection:
    """
    Test double for AgentSideConnection that captures session_update notifications.

    No mocking - this is a real class designed for testing.
    Uses SDK 0.7.0 kwargs-style signature.
    """

    def __init__(self):
        self.notifications: list[Any] = []

    async def session_update(
        self,
        session_id: str = "",
        update: Any = None,
        **kwargs: Any,
    ) -> None:
        """Capture notifications for assertions (SDK 0.7.0 kwargs style)."""
        # Store the update directly for easier test assertions
        self.notifications.append(update)


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
        manager.handle_tool_stream_event(
            "start",
            {
                "tool_name": "server__read_file",
                "tool_use_id": "use-123",
            },
        )

        # Wait for async task to complete
        await asyncio.sleep(0.1)

        # Should have sent one notification
        assert len(connection.notifications) == 1
        notification = connection.notifications[0]

        # Verify it's a tool_call with pending status
        assert notification.sessionUpdate == "tool_call"
        assert notification.status == "pending"
        assert notification.content == []

    @pytest.mark.asyncio
    async def test_on_tool_start_includes_all_args_in_title(self) -> None:
        """Tool start title should include trimmed argument list."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        arguments = {
            "prompt": "lion",
            "quality": "low",
            "tool_result": "image",
        }

        await manager.on_tool_start(
            tool_name="openai-images-generate",
            server_name="media-gen",
            arguments=arguments,
        )

        assert len(connection.notifications) == 1
        notification = connection.notifications[0]
        assert notification.sessionUpdate == "tool_call"
        assert "media-gen/openai-images-generate" in notification.title
        assert "prompt=lion" in notification.title
        assert "tool_result=image" in notification.title
        assert notification.content == []

    @pytest.mark.asyncio
    async def test_on_tool_start_omits_builtin_server_name(self) -> None:
        """Built-in ACP tools should not display the server name in titles."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        await manager.on_tool_start(
            tool_name="execute",
            server_name="acp_terminal",
            arguments={"command": "ls"},
        )

        assert len(connection.notifications) == 1
        notification = connection.notifications[0]
        assert notification.sessionUpdate == "tool_call"
        assert notification.title.startswith("execute")
        assert "acp_terminal" not in notification.title

    @pytest.mark.asyncio
    async def test_on_tool_start_does_not_emit_empty_meta(self) -> None:
        """When no contextual meta is set, `_meta` should not be serialized."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        await manager.on_tool_start(
            tool_name="list_chats",
            server_name="tg-ro",
            arguments={"chat_type": "group"},
        )

        assert len(connection.notifications) == 1
        notification = connection.notifications[0]
        assert getattr(notification, "field_meta", None) is None
        payload = notification.model_dump(by_alias=True, exclude_none=True)
        assert "_meta" not in payload

    @pytest.mark.asyncio
    async def test_on_tool_start_attaches_meta_from_context(self) -> None:
        """Tool calls should include contextual `_meta` for client-side grouping."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        with acp_tool_call_context(
            parent_tool_call_id="parent-abc",
        ):
            await manager.on_tool_start(
                tool_name="list_chats",
                server_name="tg-ro",
                arguments={"chat_type": "group"},
            )

        assert len(connection.notifications) == 1
        notification = connection.notifications[0]
        meta = getattr(notification, "field_meta", None)
        assert meta is not None
        assert meta.get("parentToolCallId") == "parent-abc"

    @pytest.mark.asyncio
    async def test_on_tool_progress_drops_self_parent_link(self) -> None:
        """When parentToolCallId equals toolCallId, omit it to avoid self edges."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        await manager.on_tool_start(
            tool_name="list_chats",
            server_name="tg-ro",
            arguments={"chat_type": "group"},
        )
        assert len(connection.notifications) == 1
        tool_call_id = connection.notifications[0].tool_call_id

        with acp_tool_call_context(
            parent_tool_call_id=tool_call_id,
        ):
            await manager.on_tool_progress(
                tool_call_id=tool_call_id,
                progress=1,
                total=2,
                message="tick",
            )

        assert len(connection.notifications) == 2
        progress_update = connection.notifications[1]
        meta = getattr(progress_update, "field_meta", None)
        # With only `parentToolCallId` in contextual meta, the manager drops self-parent links,
        # so the resulting update carries no `_meta`.
        assert meta is None or "parentToolCallId" not in meta
        payload = progress_update.model_dump(by_alias=True, exclude_none=True)
        assert "_meta" not in payload

    @pytest.mark.asyncio
    async def test_on_tool_complete_keeps_raw_input_for_clients(self) -> None:
        """Completion updates should include rawInput so clients can show INPUT reliably."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        arguments = {"chat_id": "-1001", "message": "hello"}
        tool_call_id = await manager.on_tool_start(
            tool_name="send_bot_message",
            server_name="tgbot",
            arguments=arguments,
        )

        await manager.on_tool_complete(
            tool_call_id=tool_call_id,
            success=True,
            content=None,
            error=None,
        )

        assert len(connection.notifications) == 2
        complete = connection.notifications[1]
        assert complete.sessionUpdate == "tool_call_update"
        assert complete.status == "completed"
        assert complete.rawInput == arguments

    @pytest.mark.asyncio
    async def test_delta_events_only_notify_after_threshold(self) -> None:
        """Delta notifications are only sent after 25 chunks to reduce UI noise."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Send start
        manager.handle_tool_stream_event(
            "start",
            {
                "tool_name": "server__read_file",
                "tool_use_id": "use-123",
            },
        )

        # Send 24 deltas - should NOT trigger notifications
        for i in range(24):
            manager.handle_tool_stream_event(
                "delta",
                {
                    "tool_use_id": "use-123",
                    "chunk": f"chunk{i}",
                },
            )

        await asyncio.sleep(0.1)

        # Should only have start notification (no delta notifications yet)
        assert len(connection.notifications) == 1

        # Send 25th chunk - should trigger notification
        manager.handle_tool_stream_event(
            "delta",
            {
                "tool_use_id": "use-123",
                "chunk": "chunk24",
            },
        )

        await asyncio.sleep(0.1)

        # Now should have start + 1 delta notification
        assert len(connection.notifications) == 2

        delta_notification = connection.notifications[1]
        assert delta_notification.sessionUpdate == "tool_call_update"
        assert "(streaming: 25)" in delta_notification.title

        # rawInput should NOT be set during streaming
        assert delta_notification.rawInput is None

    @pytest.mark.asyncio
    async def test_delta_chunks_accumulate_correctly(self) -> None:
        """Multiple delta events should accumulate into a single content block."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Send start then multiple deltas (need 25+ to trigger notifications)
        manager.handle_tool_stream_event(
            "start",
            {
                "tool_name": "server__read_file",
                "tool_use_id": "use-123",
            },
        )

        # Send 25 chunks to reach notification threshold
        for i in range(25):
            manager.handle_tool_stream_event(
                "delta",
                {
                    "tool_use_id": "use-123",
                    "chunk": f"chunk{i}_",
                },
            )

        # Wait for async tasks to complete
        await asyncio.sleep(0.1)

        # Should have start + 1 delta notification (at chunk 25)
        assert len(connection.notifications) == 2

        # Delta notification should have accumulated content from all chunks
        delta_notification = connection.notifications[1]
        expected_content = "".join(f"chunk{i}_" for i in range(25))
        assert delta_notification.content[0].content.text == expected_content

        # Title should show 25 chunks
        assert "(streaming: 25)" in delta_notification.title

    @pytest.mark.asyncio
    async def test_delta_before_start_is_dropped(self) -> None:
        """Delta event without prior start should be dropped (no notification)."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Send delta without start
        manager.handle_tool_stream_event(
            "delta",
            {
                "tool_use_id": "use-123",
                "chunk": '{"path": "/tmp',
            },
        )

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
        manager.handle_tool_stream_event(
            "start",
            {
                "tool_name": "server__read_file",
                "tool_use_id": "use-123",
            },
        )

        # external_id should be set IMMEDIATELY (synchronously)
        assert "use-123" in manager._stream_tool_use_ids

        # Send deltas immediately (no await between) - chunks are tracked even if not notified
        for i in range(5):
            manager.handle_tool_stream_event(
                "delta",
                {
                    "tool_use_id": "use-123",
                    "chunk": f"chunk{i}",
                },
            )

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
        manager.handle_tool_stream_event(
            "start",
            {
                "tool_name": "server__tool_a",
                "tool_use_id": "use-a",
            },
        )
        manager.handle_tool_stream_event(
            "start",
            {
                "tool_name": "server__tool_b",
                "tool_use_id": "use-b",
            },
        )

        # Send deltas to both (below threshold, so no notifications)
        manager.handle_tool_stream_event(
            "delta",
            {
                "tool_use_id": "use-a",
                "chunk": "chunk-a",
            },
        )
        manager.handle_tool_stream_event(
            "delta",
            {
                "tool_use_id": "use-b",
                "chunk": "chunk-b",
            },
        )

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

    @pytest.mark.asyncio
    async def test_parallel_tools_full_lifecycle(self) -> None:
        """
        Full lifecycle test for parallel tool calls:
        stream start → on_tool_start → on_tool_complete

        This verifies that BOTH tools receive completion notifications,
        not just one (the bug that was fixed).
        """
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # 1. Stream start for both tools (simulating parallel tool calls from LLM)
        manager.handle_tool_stream_event(
            "start",
            {
                "tool_name": "server__tool_a",
                "tool_use_id": "use-a",
            },
        )
        manager.handle_tool_stream_event(
            "start",
            {
                "tool_name": "server__tool_b",
                "tool_use_id": "use-b",
            },
        )

        # Wait for stream start notifications
        await asyncio.sleep(0.1)

        # Should have 2 start notifications
        assert len(connection.notifications) == 2

        # 2. on_tool_start for both tools (when execution begins)
        tool_call_id_a = await manager.on_tool_start(
            tool_name="tool_a",
            server_name="server",
            arguments={"path": "/file_a.txt"},
            tool_use_id="use-a",
        )
        tool_call_id_b = await manager.on_tool_start(
            tool_name="tool_b",
            server_name="server",
            arguments={"path": "/file_b.txt"},
            tool_use_id="use-b",
        )

        # Both should have different tool_call_ids
        assert tool_call_id_a != tool_call_id_b

        # Should have 2 more notifications (in_progress updates)
        assert len(connection.notifications) == 4

        # 3. on_tool_complete for both tools
        await manager.on_tool_complete(
            tool_call_id=tool_call_id_a,
            success=True,
            content=None,
        )
        await manager.on_tool_complete(
            tool_call_id=tool_call_id_b,
            success=True,
            content=None,
        )

        # Should have 2 more notifications (completed updates)
        # Total: 2 starts + 2 in_progress + 2 completed = 6
        assert len(connection.notifications) == 6

        # Verify both completion notifications were sent
        completion_notifications = [
            n for n in connection.notifications if hasattr(n, "status") and n.status == "completed"
        ]
        assert len(completion_notifications) == 2

        # Verify cleanup - streaming state should be cleared
        assert "use-a" not in manager._stream_tool_use_ids
        assert "use-b" not in manager._stream_tool_use_ids
        assert "use-a" not in manager._stream_chunk_counts
        assert "use-b" not in manager._stream_chunk_counts

    @pytest.mark.asyncio
    async def test_stream_start_then_on_tool_start_updates_same_tool_call_id(self) -> None:
        """
        If the LLM emits tool stream metadata first, on_tool_start must update
        the same ACP toolCallId (no duplicate tool cards in the client).
        """
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        manager.handle_tool_stream_event(
            "start",
            {
                "tool_name": "server__tool_a",
                "tool_use_id": "use-a",
            },
        )

        await asyncio.sleep(0.1)
        assert len(connection.notifications) == 1
        stream_start = connection.notifications[0]
        assert stream_start.sessionUpdate == "tool_call"

        tool_call_id = await manager.on_tool_start(
            tool_name="tool_a",
            server_name="server",
            arguments={"path": "/tmp/a.txt"},
            tool_use_id="use-a",
        )

        # A single call should be updated (tool_call_update), not duplicated.
        assert tool_call_id == stream_start.tool_call_id
        assert len(connection.notifications) == 2
        assert connection.notifications[1].sessionUpdate == "tool_call_update"

    @pytest.mark.asyncio
    async def test_on_tool_start_before_stream_start_dedups_late_stream_start(self) -> None:
        """
        If tool execution starts before (or without) tool stream metadata,
        a late stream start event must not create a second ToolCallStart.
        """
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        _ = await manager.on_tool_start(
            tool_name="tool_a",
            server_name="server",
            arguments={"path": "/tmp/a.txt"},
            tool_use_id="use-a",
        )

        assert len(connection.notifications) == 1
        assert connection.notifications[0].sessionUpdate == "tool_call"

        manager.handle_tool_stream_event(
            "start",
            {
                "tool_name": "server__tool_a",
                "tool_use_id": "use-a",
            },
        )

        await asyncio.sleep(0.1)
        # Still only one notification: no duplicate start card.
        assert len(connection.notifications) == 1

    @pytest.mark.asyncio
    async def test_duplicate_stream_start_events_are_deduped(self) -> None:
        """Duplicate stream start events for the same tool_use_id should be ignored."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        manager.handle_tool_stream_event(
            "start",
            {
                "tool_name": "server__tool_a",
                "tool_use_id": "use-a",
            },
        )
        manager.handle_tool_stream_event(
            "start",
            {
                "tool_name": "server__tool_a",
                "tool_use_id": "use-a",
            },
        )

        await asyncio.sleep(0.1)
        assert len(connection.notifications) == 1

    @pytest.mark.asyncio
    async def test_progress_updates_title_with_progress_and_total(self) -> None:
        """Progress updates should include progress/total in title when total is provided."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Start a tool without streaming
        tool_call_id = await manager.on_tool_start(
            tool_name="download_file",
            server_name="server",
            arguments={"url": "http://example.com/file.zip"},
        )

        # Send progress update with progress and total
        await manager.on_tool_progress(
            tool_call_id=tool_call_id,
            progress=50,
            total=100,
            message="Downloading...",
        )

        # Should have 2 notifications: start + progress
        assert len(connection.notifications) == 2

        progress_notification = connection.notifications[1]
        assert "[50/100]" in progress_notification.title
        assert "Downloading..." in progress_notification.title
        # Progress updates use simple title (no args) for cleaner display
        assert progress_notification.title == "server/download_file [50/100] - Downloading..."

    @pytest.mark.asyncio
    async def test_progress_updates_title_with_progress_only(self) -> None:
        """Progress updates should show progress value when no total is provided."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Start a tool
        tool_call_id = await manager.on_tool_start(
            tool_name="process_data",
            server_name="server",
            arguments={"input": "data.csv"},
        )

        # Send progress update with message but no total
        await manager.on_tool_progress(
            tool_call_id=tool_call_id,
            progress=10,
            total=None,
            message="Processing rows...",
        )

        # Should have 2 notifications
        assert len(connection.notifications) == 2

        progress_notification = connection.notifications[1]
        # Should have progress value and message, using simple title (no args)
        assert progress_notification.title == "server/process_data [10] - Processing rows..."

    @pytest.mark.asyncio
    async def test_progress_title_uses_simple_format(self) -> None:
        """Progress updates should use simple title (no args) for cleaner display."""
        connection = FakeAgentSideConnection()
        manager = ACPToolProgressManager(connection, "test-session")

        # Start a tool with arguments
        tool_call_id = await manager.on_tool_start(
            tool_name="read_file",
            server_name="filesystem",
            arguments={"path": "/tmp/large_file.txt"},
        )

        # Verify start notification has full title with args
        start_notification = connection.notifications[0]
        assert "path=" in start_notification.title

        # Send multiple progress updates
        await manager.on_tool_progress(
            tool_call_id=tool_call_id,
            progress=25,
            total=100,
            message="Reading...",
        )
        await manager.on_tool_progress(
            tool_call_id=tool_call_id,
            progress=75,
            total=100,
            message="Almost done...",
        )

        # Check the last progress notification - should use simple title (no args)
        last_progress = connection.notifications[-1]
        # Simple title without args for cleaner progress display
        assert last_progress.title == "filesystem/read_file [75/100] - Almost done..."
