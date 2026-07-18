from fast_agent.ui.process_poll_display import format_process_output_activity


def test_process_output_activity_distinguishes_recent_quiet_and_missing_output() -> None:
    assert (
        format_process_output_activity(
            has_observed_output=False,
            seconds_since_last_output=90,
        )
        is None
    )
    assert (
        format_process_output_activity(
            has_observed_output=True,
            seconds_since_last_output=4,
        )
        == "output now"
    )
    assert (
        format_process_output_activity(
            has_observed_output=True,
            seconds_since_last_output=12,
        )
        == "output 12s ago"
    )
    assert (
        format_process_output_activity(
            has_observed_output=True,
            seconds_since_last_output=90,
        )
        == "quiet 1m30s"
    )
