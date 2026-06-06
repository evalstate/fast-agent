from __future__ import annotations

import math

from fast_agent.batch.summary import BatchSummary


def _summary() -> BatchSummary:
    return BatchSummary(
        input_rows=1,
        selected_rows=1,
        started_at="2026-06-02T00:00:00Z",
        metadata={},
    )


def test_batch_summary_add_timing_ignores_bool_and_non_finite_values() -> None:
    summary = _summary()

    summary.add_timing(
        {
            "duration_ms": True,
            "ttft_ms": math.nan,
            "time_to_response_ms": math.inf,
        }
    )

    assert summary.timing_duration_ms == []
    assert summary.timing_ttft_ms == []
    assert summary.timing_time_to_response_ms == []


def test_batch_summary_add_timing_keeps_finite_numeric_values() -> None:
    summary = _summary()

    summary.add_timing(
        {
            "duration_ms": 12,
            "ttft_ms": 3.5,
            "time_to_response_ms": 0,
        }
    )

    assert summary.timing_duration_ms == [12.0]
    assert summary.timing_ttft_ms == [3.5]
    assert summary.timing_time_to_response_ms == [0.0]
