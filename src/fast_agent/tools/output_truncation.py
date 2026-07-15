"""Shared head/tail truncation for tool output."""

from __future__ import annotations

from dataclasses import dataclass

from fast_agent.constants import TERMINAL_BYTES_PER_TOKEN


@dataclass(frozen=True, slots=True)
class TruncatedOutput:
    text: str
    total_bytes: int
    retained_bytes: int
    omitted_bytes: int


def format_output_truncation_notice(
    *,
    label: str,
    total_bytes: int,
    head_bytes: int,
    tail_bytes: int,
    guidance: str,
) -> str:
    retained_bytes = head_bytes + tail_bytes
    retained_tokens = max(int(retained_bytes / TERMINAL_BYTES_PER_TOKEN), 1)
    total_tokens = max(int(total_bytes / TERMINAL_BYTES_PER_TOKEN), 1)
    omitted_bytes = max(total_bytes - retained_bytes, 0)
    return (
        f"[{label} truncated: showing first "
        f"{head_bytes} bytes and last {tail_bytes} bytes of "
        f"{total_bytes} bytes "
        f"(~{retained_tokens} of ~{total_tokens} tokens); "
        f"omitted {omitted_bytes} middle bytes. {guidance}]"
    )


def truncate_text_output(
    text: str,
    *,
    byte_limit: int,
    label: str,
    guidance: str,
) -> TruncatedOutput | None:
    """Retain bounded UTF-8 head and tail text with an explanatory marker."""

    blob = text.encode("utf-8", errors="replace")
    if len(blob) <= byte_limit:
        return None

    tail_limit = max(byte_limit // 2, 1)
    head_limit = max(byte_limit - tail_limit, 1)
    head_blob = blob[:head_limit]
    tail_blob = blob[-tail_limit:]
    notice = format_output_truncation_notice(
        label=label,
        total_bytes=len(blob),
        head_bytes=len(head_blob),
        tail_bytes=len(tail_blob),
        guidance=guidance,
    )

    head = head_blob.decode("utf-8", errors="replace")
    tail = tail_blob.decode("utf-8", errors="replace")
    head_part = head if head.endswith("\n") else f"{head}\n"
    tail_part = tail if tail.startswith("\n") else f"\n{tail}"
    bounded = f"{head_part}{notice}{tail_part}"
    retained_bytes = len(head_blob) + len(tail_blob)
    return TruncatedOutput(
        text=bounded,
        total_bytes=len(blob),
        retained_bytes=retained_bytes,
        omitted_bytes=len(blob) - retained_bytes,
    )
