"""Run summary aggregation for batch runs."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from fast_agent.batch.monitoring import BatchUsageTotals
from fast_agent.utils.numeric import finite_number_or_none


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
    usage_totals: BatchUsageTotals = field(default_factory=BatchUsageTotals)
    started_monotonic: float = field(default_factory=time.monotonic)

    def add_timing(self, timing: dict[str, Any] | None) -> None:
        if not timing:
            return
        duration = finite_number_or_none(timing.get("duration_ms"))
        if duration is not None:
            self.timing_duration_ms.append(float(duration))
        ttft = finite_number_or_none(timing.get("ttft_ms"))
        if ttft is not None:
            self.timing_ttft_ms.append(float(ttft))
        time_to_response = finite_number_or_none(timing.get("time_to_response_ms"))
        if time_to_response is not None:
            self.timing_time_to_response_ms.append(float(time_to_response))

    def add_usage(self, usage: dict[str, Any] | None) -> None:
        self.usage_totals.add_row_usage(usage)

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
            "usage": self.usage_totals.usage_block(processed_rows=self.processed_rows),
            "cache": self.usage_totals.cache_block(),
        }
