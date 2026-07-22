from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from fast_agent.tools.output_truncation import (
    format_output_truncation_notice,
    split_output_byte_limit,
)

_OUTPUT_LIMIT_GUIDANCE = "Increase shell_execution.output_byte_limit to retain more."


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
    output_line_count: int = 0
    unread_output_line_count: int = 0
    lifetime_output_bytes: int = 0
    retained_output_path: Path | None = None
    retained_output_max_bytes: int = 0
    retained_output_bytes: int = 0
    retained_output_complete: bool = True

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

    def consume(self) -> str:
        combined_output = self.combined()
        self.output_segments.clear()
        self.output_tail_bytes.clear()
        self.output_bytes = 0
        self.total_output_bytes = 0
        self.output_truncated = False
        self.truncation_notice_printed = False
        self.unread_output_line_count = 0
        return combined_output

    def _truncation_guidance(self) -> str:
        if self.retained_output_path is None or not self.retained_output_path.exists():
            return _OUTPUT_LIMIT_GUIDANCE
        completeness = (
            "The complete output"
            if self.retained_output_complete
            else (
                f"The first {self.retained_output_bytes} bytes retained before the "
                "temporary-file quota was reached"
            )
        )
        return (
            f"{completeness} is available during this session at "
            f"{self.retained_output_path}. Use read_text_file for selected line ranges "
            "or run a targeted search against that file; avoid reading the entire file "
            "unless necessary."
        )

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
