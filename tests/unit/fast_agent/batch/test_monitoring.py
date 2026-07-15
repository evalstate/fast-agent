from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fast_agent.batch.monitoring import BatchUsageTotals, payload_metrics
from fast_agent.batch.structured import _add_usage_totals_delta, _UsageTotalsSnapshot


def _usage(
    prompt: int,
    completion: int,
    *,
    cache_read: int | None = None,
    cache_write: int | None = None,
    reasoning: int | None = None,
    tool_use: int | None = None,
    tool_calls: int = 0,
) -> dict[str, object]:
    return {
        "schema": "fast-agent.usage/v2",
        "provider_attempts": [
            {
                "provider": "openai",
                "usage_schema": "openai-chat",
                "model": "test",
                "prompt": {
                    "total": prompt,
                    "cache_read": cache_read,
                    "cache_write": cache_write,
                    "tool_use": tool_use,
                },
                "completion": {"total": completion, "reasoning": reasoning},
                "tool_calls": tool_calls,
                "raw_usage": {},
            },
        ],
    }


def test_usage_totals_aggregate_canonical_cache_read_write() -> None:
    totals = BatchUsageTotals()

    totals.add_row_usage(
        _usage(
            1000,
            100,
            cache_read=600,
            cache_write=100,
            reasoning=20,
            tool_use=5,
            tool_calls=2,
        )
    )

    assert totals.usage_block(processed_rows=1) == {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100,
        "reasoning_tokens": 20,
        "tool_use_prompt_tokens": 5,
        "tool_calls": 2,
        "rows_with_usage": 1,
        "usage_coverage_percent": 100.0,
    }
    assert totals.cache_block() == {
        "read_tokens": 600,
        "write_tokens": 100,
        "rows_with_cache_activity": 1,
        "row_cache_activity_percent": 100.0,
    }


def test_usage_totals_aggregate_openai_turn_cached_prompt_tokens() -> None:
    totals = BatchUsageTotals()

    totals.add_row_usage(_usage(1000, 100, cache_read=400))

    assert totals.prompt_tokens == 1000
    assert totals.completion_tokens == 100
    assert totals.cache_read_tokens == 400


def test_usage_totals_prefer_turn_usage_over_cumulative_summary() -> None:
    totals = BatchUsageTotals()

    totals.add_row_usage(_usage(10, 2))
    totals.add_row_usage(_usage(20, 3))

    assert totals.usage_block(processed_rows=2)["prompt_tokens"] == 30
    assert totals.usage_block(processed_rows=2)["completion_tokens"] == 5
    assert totals.usage_block(processed_rows=2)["total_tokens"] == 35


def test_usage_totals_include_every_provider_attempt_for_one_outward_turn() -> None:
    totals = BatchUsageTotals()
    usage = _usage(25, 5, cache_read=4)
    attempts = usage["provider_attempts"]
    assert isinstance(attempts, list)
    usage["provider_attempts"] = [
        {
            "provider": "openai",
            "usage_schema": "openai-chat",
            "model": "test",
            "prompt": {"total": 20, "cache_read": 3},
            "completion": {"total": 0},
            "tool_calls": 0,
            "raw_usage": {},
        },
        *attempts,
    ]

    totals.add_row_usage(usage)

    assert totals.prompt_tokens == 45
    assert totals.completion_tokens == 5
    assert totals.total_tokens == 50
    assert totals.cache_read_tokens == 7


def test_usage_totals_aggregate_google_turn_cached_content_tokens() -> None:
    totals = BatchUsageTotals()

    totals.add_row_usage(_usage(800, 120, tool_use=30, reasoning=50, cache_read=300))

    usage = totals.usage_block(processed_rows=1)
    cache = totals.cache_block()

    assert usage["tool_use_prompt_tokens"] == 30
    assert usage["reasoning_tokens"] == 50
    assert cache["read_tokens"] == 300


def test_parallel_usage_totals_add_second_chunks_first_snapshot() -> None:
    aggregate = BatchUsageTotals(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        reasoning_tokens=5,
        tool_use_prompt_tokens=3,
        tool_calls=1,
        rows_with_usage=1,
        cache_read_tokens=40,
        cache_write_tokens=10,
        rows_with_cache_activity=1,
    )
    second_chunk = BatchUsageTotals(
        prompt_tokens=200,
        completion_tokens=30,
        total_tokens=230,
        reasoning_tokens=7,
        tool_use_prompt_tokens=4,
        tool_calls=2,
        rows_with_usage=1,
        cache_read_tokens=50,
        cache_write_tokens=20,
        rows_with_cache_activity=1,
    )

    _add_usage_totals_delta(
        aggregate,
        current=second_chunk,
        previous=_UsageTotalsSnapshot(),
    )

    assert aggregate == BatchUsageTotals(
        prompt_tokens=300,
        completion_tokens=50,
        total_tokens=350,
        reasoning_tokens=12,
        tool_use_prompt_tokens=7,
        tool_calls=3,
        rows_with_usage=2,
        cache_read_tokens=90,
        cache_write_tokens=30,
        rows_with_cache_activity=2,
    )


def test_usage_totals_ignore_missing_usage_and_zero_denominator_percentages() -> None:
    totals = BatchUsageTotals()

    totals.add_row_usage(None)

    assert "usage_coverage_percent" not in totals.usage_block(processed_rows=0)
    assert "row_cache_activity_percent" not in totals.cache_block()


def test_payload_metrics_include_cumulative_rates_and_per_row_values() -> None:
    metrics = payload_metrics(
        {
            "processed_rows": 2,
            "failed_rows": 1,
            "skipped_rows": 0,
            "selected_rows": 4,
            "duration_ms": 1000,
            "timing_ms": {
                "duration": {"mean": 10, "median": 9, "min": 8, "max": 12},
            },
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "rows_with_usage": 2,
                "usage_coverage_percent": 100.0,
            },
            "cache": {"read_tokens": 40, "write_tokens": 10},
        }
    )

    assert metrics["batch/processed_rows"] == 2
    assert metrics["batch/progress_fraction"] == 0.5
    assert metrics["batch/error_rate"] == 0.5
    assert metrics["batch/timing/duration_ms_mean"] == 10
    assert metrics["batch/usage/total_tokens_per_row"] == 60


def test_usage_totals_do_not_expose_row_content_in_metrics() -> None:
    totals = BatchUsageTotals()
    payload = _usage(10, 2)
    payload["raw_usage"] = {
        "raw_prompt": "secret prompt",
        "row": {"secret": "value"},
    }
    totals.add_row_usage(payload)

    summary_payload = {
        "processed_rows": 1,
        "failed_rows": 0,
        "skipped_rows": 0,
        "selected_rows": 1,
        "usage": totals.usage_block(processed_rows=1),
        "cache": totals.cache_block(),
    }
    rendered = repr(payload_metrics(summary_payload))

    assert "secret" not in rendered
    assert "secret prompt" not in rendered


def test_trackio_monitor_logs_with_real_fake_module(tmp_path) -> None:
    module_dir = tmp_path / "modules"
    calls_path = tmp_path / "trackio-calls.jsonl"
    module_dir.mkdir()
    (module_dir / "trackio.py").write_text(
        """
import json
import os

CALLS = os.environ["TRACKIO_CALLS"]

def _write(payload):
    with open(CALLS, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\\n")

def init(**kwargs):
    _write({"kind": "init", "kwargs": kwargs})

def log(metrics, step=None):
    _write({"kind": "log", "metrics": metrics, "step": step})

def finish():
    _write({"kind": "finish"})
""",
        encoding="utf-8",
    )

    code = """
from pathlib import Path
from fast_agent.batch.monitoring import BatchTrackioOptions, create_batch_monitor
from fast_agent.batch.structured import StructuredBatchOptions
from fast_agent.batch.summary import BatchSummary

options = StructuredBatchOptions(
    input_path=Path("rows.jsonl"),
    output_path=Path("out.jsonl"),
    model="passthrough",
    trackio=BatchTrackioOptions(
        project="demo",
        name="run-1",
        group="group-1",
        log_every=1,
        config={"dataset": "pilot"},
    ),
)
summary = BatchSummary(
    input_rows=1,
    selected_rows=1,
    started_at="2026-06-11T00:00:00Z",
    metadata={},
)
monitor = create_batch_monitor(options)
monitor.start(options, 1)
summary.processed_rows = 1
summary.add_usage({
    "schema": "fast-agent.usage/v2",
    "provider_attempts": [{
        "provider": "openai",
        "usage_schema": "openai-chat",
        "model": "test",
        "prompt": {"total": 10},
        "completion": {"total": 3},
        "raw_usage": {},
    }],
})
monitor.row(summary)
monitor.complete(summary.to_dict("2026-06-11T00:00:01Z"))
monitor.close()
"""
    env = _subprocess_env_with_module_dir(module_dir)
    env["TRACKIO_CALLS"] = str(calls_path)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[4],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
    assert [call["kind"] for call in calls] == ["init", "log", "finish"]
    assert calls[0]["kwargs"]["project"] == "demo"
    assert calls[0]["kwargs"]["config"]["dataset"] == "pilot"
    assert calls[1]["step"] == 1
    assert calls[1]["metrics"]["batch/usage/prompt_tokens"] == 10
    assert calls[1]["metrics"]["batch/usage/prompt_tokens_per_row"] == 10


def test_trackio_monitor_missing_dependency_uses_clear_error(tmp_path) -> None:
    module_dir = tmp_path / "modules"
    module_dir.mkdir()
    (module_dir / "trackio.py").write_text(
        'raise ImportError("trackio intentionally unavailable")\n',
        encoding="utf-8",
    )

    code = """
from pathlib import Path
from fast_agent.batch.monitoring import (
    BatchTrackioOptions,
    TRACKIO_MISSING_MESSAGE,
    create_batch_monitor,
)
from fast_agent.batch.structured import StructuredBatchOptions

options = StructuredBatchOptions(
    input_path=Path("rows.jsonl"),
    output_path=Path("out.jsonl"),
    trackio=BatchTrackioOptions(project="demo"),
)
try:
    create_batch_monitor(options).start(options, 0)
except RuntimeError as exc:
    assert str(exc) == TRACKIO_MISSING_MESSAGE
else:
    raise SystemExit("expected RuntimeError")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[4],
        env=_subprocess_env_with_module_dir(module_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def _subprocess_env_with_module_dir(module_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = Path(__file__).resolve().parents[4] / "src"
    existing = env.get("PYTHONPATH")
    paths = [str(module_dir), str(src_path)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = ":".join(paths)
    return env
