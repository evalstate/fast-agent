"""Run summary aggregation for batch runs."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


@dataclass
class BatchSummary:
    input_rows: int
    selected_rows: int
    started_at: str
    metadata: dict[str, Any]
    processed_rows: int = 0
    skipped_rows: int = 0
    failed_rows: int = 0
    timing_duration_ms: list[float] = field(default_factory=list)
    timing_ttft_ms: list[float] = field(default_factory=list)
    timing_time_to_response_ms: list[float] = field(default_factory=list)
    started_monotonic: float = field(default_factory=time.monotonic)

    def add_timing(self, timing: dict[str, Any] | None) -> None:
        if not timing:
            return
        duration = timing.get("duration_ms")
        if isinstance(duration, int | float):
            self.timing_duration_ms.append(float(duration))
        ttft = timing.get("ttft_ms")
        if isinstance(ttft, int | float):
            self.timing_ttft_ms.append(float(ttft))
        time_to_response = timing.get("time_to_response_ms")
        if isinstance(time_to_response, int | float):
            self.timing_time_to_response_ms.append(float(time_to_response))

    def to_dict(self, completed_at: str) -> dict[str, Any]:
        return {
            **self.metadata,
            "started_at": self.started_at,
            "completed_at": completed_at,
            "input_rows": self.input_rows,
            "selected_rows": self.selected_rows,
            "processed_rows": self.processed_rows,
            "skipped_rows": self.skipped_rows,
            "failed_rows": self.failed_rows,
            "duration_ms": round((time.monotonic() - self.started_monotonic) * 1000, 2),
            "timing_ms": {
                "duration": _stats(self.timing_duration_ms),
                "ttft": _stats(self.timing_ttft_ms),
                "time_to_response": _stats(self.timing_time_to_response_ms),
            },
        }

