from fast_agent.ui import notification_tracker


def test_warning_notifications_are_tracked_in_counts_and_summary() -> None:
    notification_tracker.clear()

    notification_tracker.add_warning("skills placeholder missing")
    notification_tracker.add_warning("another warning")

    counts = notification_tracker.get_counts_by_type()
    assert counts.get("warning") == 2
    assert notification_tracker.get_count() == 2

    summary = notification_tracker.get_summary(compact=True)
    assert "warn:2" in summary

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

    removed = notification_tracker.remove_startup_warnings_containing("shell cwd")
    assert removed == 2

    queued = notification_tracker.pop_startup_warnings()
    assert queued == ["other startup warning"]

    notification_tracker.clear()
