from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from fast_agent.tools.output_truncation import (
    format_output_truncation_notice,
    split_output_byte_limit,
)
from fast_agent.tools.transient_artifacts import format_retained_artifact_notice

_OUTPUT_LIMIT_GUIDANCE = "Increase shell_execution.output_byte_limit to retain more."


def process_output_preview(
    blob: bytes,
    *,
    limit: int,
    total_bytes: int,
    guidance: str = "Use action='read_output' for retained output.",
) -> str:
    """Cap output content at a UTF-8 boundary; the notice is outside the budget."""
    text = blob[:limit].decode("utf-8", errors="ignore")
    shown = len(text.encode("utf-8"))
    if shown < total_bytes:
        text += f"\n[Output truncated: showing {shown} of {total_bytes} bytes. {guidance}]\n"
    return text


@dataclass(slots=True)
class ShellOutputBuffer:
    output_byte_limit: int
    output_byte_limit_requested: bool = False
    output_segments: list[str] = field(default_factory=list)
    output_tail_bytes: bytearray = field(default_factory=bytearray)
    output_bytes: int = 0
    total_output_bytes: int = 0
    output_truncated: bool = False
    truncation_notice_printed: bool = False
    had_stream_output: bool = False
    unread_output_activity: bool = False
    output_line_count: int = 0
    unread_output_line_count: int = 0
    lifetime_output_bytes: int = 0
    lifetime_stdout_bytes: int = 0
    lifetime_stderr_bytes: int = 0
    retained_output_path: Path | None = None
    retained_output_max_bytes: int = 0
    retained_output_bytes: int = 0
    retained_output_complete: bool = True
    retained_output_via_process: bool = False
    extended_guidance: bool = False

    def append(self, text: str) -> None:
        output_blob = text.encode("utf-8", errors="replace")
        if self.retained_output_path is not None and self.retained_output_path.exists():
            self._append_retained(output_blob)
        elif self.total_output_bytes + len(output_blob) > self.output_byte_limit:
            self._start_retained(output_blob)
        self.total_output_bytes += len(output_blob)
        self.lifetime_output_bytes += len(output_blob)
        self._append_tail(output_blob)
        if self.output_truncated:
            return

        remaining = self.output_byte_limit - self.output_bytes
        if remaining <= 0:
            self.output_truncated = True
            return
        if len(output_blob) <= remaining:
            self.output_segments.append(text)
            self.output_bytes += len(output_blob)
            return

        truncated_text = output_blob[:remaining].decode("utf-8", errors="replace")
        if truncated_text:
            self.output_segments.append(truncated_text)
        self.output_bytes += remaining
        self.output_truncated = True

    def append_stream(
        self,
        text: str,
        *,
        is_stderr: bool,
        count_bytes: bool = True,
    ) -> None:
        """Append process output while tracking raw stdout/stderr byte totals."""
        if count_bytes:
            self.record_stream_bytes(
                len(text.encode("utf-8", errors="replace")),
                is_stderr=is_stderr,
            )
        self.append(text if not is_stderr else f"[stderr] {text}")

    def record_stream_bytes(self, byte_count: int, *, is_stderr: bool) -> None:
        """Record raw stream bytes at the environment read boundary."""
        if is_stderr:
            self.lifetime_stderr_bytes += byte_count
        else:
            self.lifetime_stdout_bytes += byte_count

    def combined(self) -> str:
        if not self.output_truncated:
            return "".join(self.output_segments)

        window = split_output_byte_limit(self.output_byte_limit)
        head_blob = "".join(self.output_segments).encode("utf-8", errors="replace")[
            : window.head_bytes
        ]
        tail_blob = bytes(self.output_tail_bytes)[-window.tail_bytes :]

        parts: list[str] = []
        if head_blob:
            head_text = head_blob.decode("utf-8", errors="replace")
            parts.append(head_text if head_text.endswith("\n") else f"{head_text}\n")

        parts.append(
            format_output_truncation_notice(
                label="Output",
                total_bytes=self.total_output_bytes,
                head_bytes=len(head_blob),
                tail_bytes=len(tail_blob),
                guidance=self._truncation_guidance(),
            )
            + "\n"
        )

        if tail_blob:
            tail_text = tail_blob.decode("utf-8", errors="replace")
            parts.append(tail_text if tail_text.endswith("\n") else f"{tail_text}\n")

        return "".join(parts)

    def consume(self, limit: int | None = None) -> str:
        if limit is None:
            combined_output = self.combined()
        else:
            blob = "".join(self.output_segments).encode("utf-8")
            if self.total_output_bytes > limit and (
                self.retained_output_path is not None and not self.retained_output_path.exists()
            ):
                self._start_retained(b"")
            combined_output = process_output_preview(
                blob,
                limit=min(limit, self.output_bytes),
                total_bytes=self.total_output_bytes,
                guidance=(
                    "Use action='read_output' for retained output."
                    if self.retained_output_path is not None and self.retained_output_path.exists()
                    else self._truncation_guidance()
                ),
            )
        self.output_segments.clear()
        self.output_tail_bytes.clear()
        self.output_bytes = 0
        self.total_output_bytes = 0
        self.output_truncated = False
        self.truncation_notice_printed = False
        self.unread_output_activity = False
        self.unread_output_line_count = 0
        return combined_output

    def _truncation_guidance(self) -> str:
        if self.retained_output_path is None or not self.retained_output_path.exists():
            return _OUTPUT_LIMIT_GUIDANCE
        if self.retained_output_via_process:
            completeness = (
                "The complete output"
                if self.retained_output_complete
                else (
                    f"The first {self.retained_output_bytes} bytes retained before "
                    "the temporary-output quota was reached"
                )
            )
            return (
                f"{completeness} can be read through the managed process handle. "
                "Use the process tool with action='read_output' and the process_id "
                "from this result."
            )
        notice = format_retained_artifact_notice(
            path=str(self.retained_output_path),
            retained_bytes=self.retained_output_bytes,
            complete=self.retained_output_complete,
            description="output",
        )
        if self.extended_guidance:
            return (
                f"{notice} Also, before drawing conclusions from truncated output, "
                "inspect the relevant retained content."
            )
        return notice

    def _start_retained(self, triggering_blob: bytes) -> None:
        if self.retained_output_path is None or self.retained_output_max_bytes <= 0:
            return
        prefix = "".join(self.output_segments).encode("utf-8", errors="replace")
        self.retained_output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.retained_output_path.write_bytes(b"")
        self.retained_output_path.chmod(0o600)
        self._append_retained(prefix)
        self._append_retained(triggering_blob)

    def _append_retained(self, output_blob: bytes) -> None:
        if (
            self.retained_output_path is None
            or not self.retained_output_path.exists()
            or not self.retained_output_complete
        ):
            return
        remaining = self.retained_output_max_bytes - self.retained_output_bytes
        if remaining <= 0:
            self.retained_output_complete = False
            return
        retained_blob = output_blob[:remaining]
        if retained_blob:
            with self.retained_output_path.open("ab") as retained_file:
                retained_file.write(retained_blob)
            self.retained_output_bytes += len(retained_blob)
        if len(output_blob) > remaining:
            self.retained_output_complete = False

    def _append_tail(self, output_blob: bytes) -> None:
        tail_limit = split_output_byte_limit(self.output_byte_limit).tail_bytes
        if len(output_blob) >= tail_limit:
            self.output_tail_bytes = bytearray(output_blob[-tail_limit:])
            return

        self.output_tail_bytes.extend(output_blob)
        overflow = len(self.output_tail_bytes) - tail_limit
        if overflow > 0:
            del self.output_tail_bytes[:overflow]
