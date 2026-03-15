from fast_agent.llm.tool_tracking import ToolCallTracker


def test_tool_call_tracker_register_resolve_and_close_happy_path() -> None:
    tracker = ToolCallTracker()

    state = tracker.register(tool_use_id="tool_1", name="weather", index=3)

    assert state.tool_use_id == "tool_1"
    assert tracker.resolve_open(tool_use_id="tool_1") is state
    assert tracker.resolve_open(index=3) is state

    closed = tracker.close(tool_use_id="tool_1")

    assert closed is state
    assert tracker.resolve_open(tool_use_id="tool_1") is None
    assert tracker.is_completed(tool_use_id="tool_1")
    assert tracker.completed() == [state]


def test_tool_call_tracker_reregister_is_idempotent() -> None:
    tracker = ToolCallTracker()

    original = tracker.register(tool_use_id="tool_1", name="tool")
    original.start_notified = True
    updated = tracker.register(tool_use_id="tool_1", name="weather")

    assert updated is original
    assert updated.name == "weather"
    assert updated.start_notified is True


def test_tool_call_tracker_late_index_attachment_updates_lookup() -> None:
    tracker = ToolCallTracker()

    state = tracker.register(tool_use_id="tool_1", name="weather")
    assert state.index is None

    tracker.register(tool_use_id="tool_1", name="weather", index=7)

    assert state.index == 7
    assert tracker.resolve_open(index=7) is state


def test_tool_call_tracker_close_supports_index_lookup() -> None:
    tracker = ToolCallTracker()
    state = tracker.register(tool_use_id="tool_1", name="weather", index=2)

    closed = tracker.close(index=2)

    assert closed is state
    assert tracker.is_completed(index=2)


def test_tool_call_tracker_incomplete_and_completed_states_are_separate() -> None:
    tracker = ToolCallTracker()
    open_state = tracker.register(tool_use_id="tool_open", name="weather", index=1)
    closed_state = tracker.register(tool_use_id="tool_closed", name="calculator", index=4)

    tracker.close(tool_use_id="tool_closed")

    assert tracker.incomplete() == [open_state]
    assert tracker.completed() == [closed_state]
    assert tracker.is_completed(tool_use_id="tool_closed")
    assert not tracker.is_completed(tool_use_id="tool_open")


def test_tool_call_tracker_reregister_by_index_rekeys_placeholder_identity() -> None:
    tracker = ToolCallTracker()
    placeholder = tracker.register(tool_use_id="item_123", name="web_search", index=5)

    updated = tracker.register(tool_use_id="call_456", name="web_search", index=5)

    assert updated is placeholder
    assert updated.tool_use_id == "call_456"
    assert tracker.resolve_open(tool_use_id="call_456") is updated
    assert tracker.resolve_open(tool_use_id="item_123") is None
