from fast_agent.ui.process_poll_display import (
    format_process_output_activity,
    format_process_output_size,
)


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
        == "last output 12s ago"
    )
    assert (
        format_process_output_activity(
            has_observed_output=True,
            seconds_since_last_output=90,
        )
        == "quiet 1m30s"
    )


def test_process_output_size_uses_compact_decimal_units() -> None:
    assert format_process_output_size(None) is None
    assert format_process_output_size(0) is None
    assert format_process_output_size(999) == "999B"
    assert format_process_output_size(1_250) == "1.2KB"
    assert format_process_output_size(12_500) == "12.5KB"
    assert format_process_output_size(1_250_000) == "1.2MB"
