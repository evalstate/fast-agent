from fast_agent.ui import notification_tracker


def test_warning_notifications_are_tracked_in_counts_and_summary() -> None:
    notification_tracker.clear()

    notification_tracker.add_warning(" skills placeholder missing ")
    assert notification_tracker.get_latest() == {
        "type": "warning",
        "message": "skills placeholder missing",
    }

    notification_tracker.add_warning("another warning")

    counts = notification_tracker.get_counts_by_type()
    assert counts.get("warning") == 2
    assert notification_tracker.get_count() == 2

    summary = notification_tracker.get_summary(compact=True)
    assert "warn:2" in summary

    notification_tracker.clear()


def test_format_event_label_pluralizes_known_and_unknown_events() -> None:
    assert notification_tracker.format_event_label("warning", 1) == "1 warning"
    assert notification_tracker.format_event_label("warning", 2) == "2 warnings"
    assert notification_tracker.format_event_label("custom_event", 2) == "2 custom events"


def test_get_counts_by_type_preserves_known_order_before_unknown_events() -> None:
    notification_tracker.clear()

    notification_tracker.add_warning("warning")
    notification_tracker.notifications.append({"type": "custom_event"})
    notification_tracker.notifications.append({"type": "another_event"})
    notification_tracker.add_tool_update("server")

    assert notification_tracker.get_counts_by_type() == {
        "tool_update": 1,
        "warning": 1,
        "custom_event": 1,
        "another_event": 1,
    }

    notification_tracker.clear()


def test_sampling_and_elicitation_state_transitions_are_tracked() -> None:
    notification_tracker.clear()

    notification_tracker.start_sampling("demo")
    assert notification_tracker.get_active_status() == {"type": "sampling", "server": "demo"}

    notification_tracker.end_sampling("demo")
    notification_tracker.start_elicitation("forms")
    assert notification_tracker.get_active_status() == {"type": "elicitation", "server": "forms"}

    notification_tracker.end_elicitation("forms")
    assert notification_tracker.get_active_status() is None
    assert notification_tracker.get_counts_by_type() == {"sampling": 1, "elicitation": 1}

    notification_tracker.clear()


def test_ending_untracked_active_event_does_not_record_completion() -> None:
    notification_tracker.clear()

    notification_tracker.end_sampling("missing")

    assert notification_tracker.get_count() == 0
    assert notification_tracker.get_active_status() is None

    notification_tracker.clear()


def test_ending_different_server_keeps_matching_active_event() -> None:
    notification_tracker.clear()

    notification_tracker.start_sampling("alpha")
    notification_tracker.end_sampling("beta")

    assert notification_tracker.get_count() == 0
    assert notification_tracker.get_active_status() == {"type": "sampling", "server": "alpha"}

    notification_tracker.clear()


def test_overlapping_active_events_are_tracked_per_server() -> None:
    notification_tracker.clear()

    notification_tracker.start_sampling("alpha")
    notification_tracker.start_sampling("beta")
    assert notification_tracker.get_active_status() == {"type": "sampling", "server": "alpha"}

    notification_tracker.end_sampling("alpha")
    assert notification_tracker.get_active_status() == {"type": "sampling", "server": "beta"}

    notification_tracker.start_elicitation("forms")
    assert notification_tracker.get_active_status() == {"type": "sampling", "server": "beta"}

    notification_tracker.end_sampling("beta")
    assert notification_tracker.get_active_status() == {"type": "elicitation", "server": "forms"}

    notification_tracker.end_elicitation("forms")
    assert notification_tracker.get_active_status() is None

    notification_tracker.clear()


def test_startup_warnings_are_queued_separately_from_toolbar_counts() -> None:
    notification_tracker.clear()

    notification_tracker.add_warning("skills placeholder missing", surface="startup_once")
    notification_tracker.add_warning("skills placeholder missing", surface="startup_once")

    assert notification_tracker.get_count() == 0
    assert notification_tracker.get_counts_by_type() == {}

    queued = notification_tracker.pop_startup_warnings()
    assert queued == ["skills placeholder missing"]
    assert notification_tracker.pop_startup_warnings() == []

    notification_tracker.clear()


def test_remove_startup_warnings_containing_fragment() -> None:
    notification_tracker.clear()

    notification_tracker.add_warning("Agent A shell cwd missing", surface="startup_once")
    notification_tracker.add_warning("Agent B shell cwd missing", surface="startup_once")
    notification_tracker.add_warning("other startup warning", surface="startup_once")

    removed = notification_tracker.remove_startup_warnings_containing(" SHELL CWD ")
    assert removed == 2

    queued = notification_tracker.pop_startup_warnings()
    assert queued == ["other startup warning"]

    notification_tracker.clear()
