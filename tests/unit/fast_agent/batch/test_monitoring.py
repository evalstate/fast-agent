from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fast_agent.batch.monitoring import BatchUsageTotals, payload_metrics


def test_usage_totals_aggregate_anthropic_summary_cache_read_write() -> None:
    totals = BatchUsageTotals()

    totals.add_row_usage(
        {
            "summary": {
                "cumulative_input_tokens": 1000,
                "cumulative_output_tokens": 100,
                "cumulative_billing_tokens": 1100,
                "cumulative_reasoning_tokens": 20,
                "cumulative_tool_use_tokens": 5,
                "cumulative_tool_calls": 2,
                "cumulative_cache_read_tokens": 600,
                "cumulative_cache_write_tokens": 100,
                "cumulative_cache_hit_tokens": 0,
                "cumulative_effective_input_tokens": 300,
            }
        }
    )

    assert totals.usage_block(processed_rows=1) == {
        "input_tokens": 1000,
        "output_tokens": 100,
        "total_tokens": 1100,
        "billing_tokens": 1100,
        "reasoning_tokens": 20,
        "tool_use_tokens": 5,
        "tool_calls": 2,
        "rows_with_usage": 1,
        "usage_coverage_percent": 100.0,
    }
    assert totals.cache_block() == {
        "read_tokens": 600,
        "write_tokens": 100,
        "hit_tokens": 0,
        "served_tokens": 600,
        "activity_tokens": 700,
        "effective_input_tokens": 300,
        "hit_rate_percent": 60.0,
        "write_rate_percent": 10.0,
        "activity_rate_percent": 70.0,
        "rows_with_cache_activity": 1,
        "row_cache_activity_percent": 100.0,
        "non_cached_input_tokens": 400,
        "served_to_effective_input_ratio": 2.0,
    }


def test_usage_totals_aggregate_openai_turn_cached_prompt_tokens() -> None:
    totals = BatchUsageTotals()

    totals.add_row_usage(
        {
            "turn": {
                "input_tokens": 1000,
                "display_input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 1100,
                "effective_input_tokens": 600,
                "cache_usage": {"cache_hit_tokens": 400},
            }
        }
    )

    assert totals.input_tokens == 1000
    assert totals.output_tokens == 100
    assert totals.cache_hit_tokens == 400
    assert totals.cache_block()["hit_rate_percent"] == 40.0


def test_usage_totals_prefer_turn_usage_over_cumulative_summary() -> None:
    totals = BatchUsageTotals()

    totals.add_row_usage(
        {
            "turn": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            "summary": {
                "cumulative_input_tokens": 10,
                "cumulative_output_tokens": 2,
                "cumulative_total_tokens": 12,
            },
        }
    )
    totals.add_row_usage(
        {
            "turn": {"input_tokens": 20, "output_tokens": 3, "total_tokens": 23},
            "summary": {
                "cumulative_input_tokens": 30,
                "cumulative_output_tokens": 5,
                "cumulative_total_tokens": 35,
            },
        }
    )

    assert totals.usage_block(processed_rows=2)["input_tokens"] == 30
    assert totals.usage_block(processed_rows=2)["output_tokens"] == 5
    assert totals.usage_block(processed_rows=2)["total_tokens"] == 35


def test_usage_totals_aggregate_google_turn_cached_content_tokens() -> None:
    totals = BatchUsageTotals()

    totals.add_row_usage(
        {
            "turn": {
                "input_tokens": 800,
                "output_tokens": 120,
                "total_tokens": 920,
                "tool_use_tokens": 30,
                "reasoning_tokens": 50,
                "effective_input_tokens": 500,
                "cache_usage": {"cache_hit_tokens": 300},
            }
        }
    )

    usage = totals.usage_block(processed_rows=1)
    cache = totals.cache_block()

    assert usage["tool_use_tokens"] == 30
    assert usage["reasoning_tokens"] == 50
    assert cache["served_tokens"] == 300
    assert cache["hit_rate_percent"] == 37.5


def test_usage_totals_ignore_missing_usage_and_zero_denominator_percentages() -> None:
    totals = BatchUsageTotals()

    totals.add_row_usage(None)

    assert totals.usage_block(processed_rows=0)["usage_coverage_percent"] is None
    cache = totals.cache_block()
    assert cache["hit_rate_percent"] is None
    assert cache["row_cache_activity_percent"] is None
    assert cache["served_to_effective_input_ratio"] is None


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
                "input_tokens": 100,
                "output_tokens": 20,
                "billing_tokens": 120,
                "rows_with_usage": 2,
                "usage_coverage_percent": 100.0,
            },
            "cache": {
                "served_tokens": 40,
                "write_tokens": 10,
                "hit_rate_percent": 40.0,
            },
        }
    )

    assert metrics["batch/processed_rows"] == 2
    assert metrics["batch/progress_fraction"] == 0.5
    assert metrics["batch/error_rate"] == 0.5
    assert metrics["batch/timing/duration_ms_mean"] == 10
    assert metrics["batch/usage/billing_tokens_per_row"] == 60
    assert metrics["batch/cache/served_tokens_per_row"] == 20


def test_usage_totals_do_not_expose_row_content_in_metrics() -> None:
    totals = BatchUsageTotals()
    totals.add_row_usage(
        {
            "turn": {
                "input_tokens": 10,
                "output_tokens": 2,
                "raw_prompt": "secret prompt",
                "row": {"secret": "value"},
            }
        }
    )

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
    assert "prompt" not in rendered


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
summary.add_usage({"turn": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}})
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
    assert calls[1]["metrics"]["batch/usage/input_tokens"] == 10
    assert calls[1]["metrics"]["batch/usage/input_tokens_per_row"] == 10


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
