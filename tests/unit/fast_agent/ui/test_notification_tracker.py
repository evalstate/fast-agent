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

