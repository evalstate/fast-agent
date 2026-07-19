"""Compact managed-process polling display helpers."""

from dataclasses import dataclass

from pydantic import ByteSize


@dataclass(frozen=True, slots=True)
class ProcessOutputActivity:
    """Compact output-activity chip for process monitoring displays."""

    text: str
    style: str | None = None


def format_process_output_activity(
    *,
    has_observed_output: bool | None,
    seconds_since_last_output: float | None,
) -> ProcessOutputActivity | None:
    """Return a short activity chip that highlights recent output, then goes quiet.

    Recent output is emphasized for a short window, then fades, then collapses to
    an untimed ``quiet`` marker. Missing/never-seen output stays silent.
    """
    if has_observed_output is None or seconds_since_last_output is None:
        return None
    if not has_observed_output:
        return None

    age = max(seconds_since_last_output, 0.0)
    if age <= 5:
        return ProcessOutputActivity("output", "bold bright_green")
    if age < 30:
        return ProcessOutputActivity("output", "green")
    return ProcessOutputActivity("quiet")


def format_process_output_size(total_bytes: int | None) -> str | None:
    """Return a compact cumulative output size for active-process displays."""
    if type(total_bytes) is not int or total_bytes <= 0:
        return None

    return ByteSize(total_bytes).human_readable(decimal=True)
