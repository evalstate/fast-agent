"""Compact managed-process polling display helpers."""

from fast_agent.utils.time import format_compact_duration


def format_process_output_activity(
    *,
    has_observed_output: bool | None,
    seconds_since_last_output: float | None,
) -> str | None:
    if has_observed_output is None or seconds_since_last_output is None:
        return None
    if not has_observed_output:
        return "no output"

    duration = format_compact_duration(max(seconds_since_last_output, 0.0)) or "<1s"
    if seconds_since_last_output <= 5:
        return "output now"
    if seconds_since_last_output < 30:
        return f"output {duration} ago"
    return f"quiet {duration}"
