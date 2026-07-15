"""Batch usage aggregation and optional Trackio monitoring."""

from __future__ import annotations

import importlib
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from fast_agent.llm.usage_tracking import UsageReport
from fast_agent.utils.numeric import finite_number_or_none, nonnegative_int_or_none

if TYPE_CHECKING:
    from fast_agent.batch.structured import StructuredBatchOptions
    from fast_agent.batch.summary import BatchSummary


TRACKIO_MISSING_MESSAGE = (
    "Trackio batch monitoring requires trackio. Install fast-agent-mcp[trackio], "
    "fast-agent-mcp[gepa], or install trackio in this environment."
)


@dataclass(frozen=True)
class BatchTrackioOptions:
    """Explicit Trackio configuration for one batch run."""

    project: str | None = None
    name: str | None = None
    group: str | None = None
    space_id: str | None = None
    server_url: str | None = None
    log_every: int | None = None
    config: Mapping[str, Any] | None = None


@dataclass
class BatchUsageTotals:
    """Cumulative usage/cache totals across processed batch rows."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    tool_use_prompt_tokens: int | None = None
    tool_calls: int = 0
    rows_with_usage: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    rows_with_cache_activity: int = 0

    def add_row_usage(self, usage: Mapping[str, Any] | None) -> None:
        """Add one canonical fast-agent usage observation."""

        if usage is None:
            return
        try:
            report = UsageReport.model_validate(usage)
        except ValueError:
            return
        usage_summary = report.consumed

        prior_rows = self.rows_with_usage
        self.rows_with_usage += 1
        self.prompt_tokens = _add_complete(
            self.prompt_tokens, usage_summary.prompt.total, prior_rows=prior_rows
        )
        self.completion_tokens = _add_complete(
            self.completion_tokens, usage_summary.completion.total, prior_rows=prior_rows
        )
        self.total_tokens = _add_complete(
            self.total_tokens, usage_summary.total, prior_rows=prior_rows
        )
        self.reasoning_tokens = _add_complete(
            self.reasoning_tokens, usage_summary.completion.reasoning, prior_rows=prior_rows
        )
        self.tool_use_prompt_tokens = _add_complete(
            self.tool_use_prompt_tokens, usage_summary.prompt.tool_use, prior_rows=prior_rows
        )
        self.cache_read_tokens = _add_complete(
            self.cache_read_tokens, usage_summary.prompt.cache_read, prior_rows=prior_rows
        )
        self.cache_write_tokens = _add_complete(
            self.cache_write_tokens, usage_summary.prompt.cache_write, prior_rows=prior_rows
        )
        self.tool_calls += usage_summary.tool_calls
        if (usage_summary.prompt.cache_read or 0) > 0 or (
            usage_summary.prompt.cache_write or 0
        ) > 0:
            self.rows_with_cache_activity += 1

    def usage_block(self, *, processed_rows: int) -> dict[str, Any]:
        return _without_none({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "tool_use_prompt_tokens": self.tool_use_prompt_tokens,
            "tool_calls": self.tool_calls,
            "rows_with_usage": self.rows_with_usage,
            "usage_coverage_percent": _percent(self.rows_with_usage, processed_rows),
        })

    def cache_block(self) -> dict[str, Any]:
        return _without_none({
            "read_tokens": self.cache_read_tokens,
            "write_tokens": self.cache_write_tokens,
            "rows_with_cache_activity": self.rows_with_cache_activity,
            "row_cache_activity_percent": _percent(
                self.rows_with_cache_activity, self.rows_with_usage
            ),
        })


class BatchMonitor(Protocol):
    def start(self, options: StructuredBatchOptions, selected_rows: int) -> None: ...

    def row(self, summary: BatchSummary) -> None: ...

    def complete(self, payload: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class NoOpBatchMonitor:
    def start(self, options: StructuredBatchOptions, selected_rows: int) -> None:
        pass

    def row(self, summary: BatchSummary) -> None:
        pass

    def complete(self, payload: Mapping[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class TrackioBatchMonitor:
    """Trackio monitor for explicit batch telemetry opt-in."""

    def __init__(self, options: BatchTrackioOptions) -> None:
        self.options = options
        self._trackio: Any | None = None
        self._log_every = options.log_every or 10
        self._last_logged_processed: int | None = None
        self._warned_log_failure = False

    def start(self, options: StructuredBatchOptions, selected_rows: int) -> None:
        if self.options.project is None:
            return
        try:
            trackio = importlib.import_module("trackio")
        except ImportError as exc:
            raise RuntimeError(TRACKIO_MISSING_MESSAGE) from exc

        init = trackio.__dict__.get("init")
        if not callable(init):
            raise RuntimeError(
                "Trackio batch monitoring requires a trackio package with trackio.init()."
            )

        config = _trackio_config(options, selected_rows, self.options.config)
        init_kwargs: dict[str, Any] = {
            "project": self.options.project,
            "name": self.options.name or _default_trackio_name(options),
            "config": config,
            "embed": False,
            "auto_log_gpu": False,
        }
        _set_if_not_none(init_kwargs, "group", self.options.group)
        _set_if_not_none(init_kwargs, "space_id", self.options.space_id)
        _set_if_not_none(init_kwargs, "server_url", self.options.server_url)
        init(**init_kwargs)
        self._trackio = trackio

    def row(self, summary: BatchSummary) -> None:
        if self._trackio is None:
            return
        processed = summary.processed_rows
        if processed <= 0 or processed % self._log_every != 0:
            return
        self._log(summary_metrics(summary), step=processed)

    def complete(self, payload: Mapping[str, Any]) -> None:
        if self._trackio is None:
            return
        processed = nonnegative_int_or_none(payload.get("processed_rows")) or 0
        metrics = payload_metrics(payload)
        if metrics:
            self._log(metrics, step=processed)

    def close(self) -> None:
        if self._trackio is None:
            return
        finish = self._trackio.__dict__.get("finish")
        if callable(finish):
            try:
                finish()
            except Exception as exc:
                self._warn_once(f"Trackio finish failed: {exc}")

    def _log(self, metrics: Mapping[str, int | float], *, step: int) -> None:
        if self._last_logged_processed == step:
            return
        log = self._trackio.__dict__.get("log") if self._trackio is not None else None
        if not callable(log):
            self._warn_once("Trackio logging failed: trackio.log() is unavailable")
            return
        try:
            log(dict(metrics), step=step)
            self._last_logged_processed = step
        except Exception as exc:
            self._warn_once(f"Trackio logging failed: {exc}")

    def _warn_once(self, message: str) -> None:
        if self._warned_log_failure:
            return
        self._warned_log_failure = True
        print(f"batch: {message}", file=sys.stderr, flush=True)


def create_batch_monitor(options: StructuredBatchOptions) -> BatchMonitor:
    trackio_options = options.trackio
    if trackio_options is None or trackio_options.project is None:
        return NoOpBatchMonitor()
    return TrackioBatchMonitor(trackio_options)


def payload_metrics(payload: Mapping[str, Any]) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    processed = _payload_int(payload, "processed_rows")
    failed = _payload_int(payload, "failed_rows")
    skipped = _payload_int(payload, "skipped_rows")
    selected = _payload_int(payload, "selected_rows")
    duration_ms = _number(payload.get("duration_ms")) or 0.0
    _add_progress_metrics(
        metrics,
        processed=processed,
        failed=failed,
        skipped=skipped,
        selected=selected,
        elapsed_seconds=duration_ms / 1000,
    )
    _add_timing_payload_metrics(metrics, payload)
    _add_usage_payload_metrics(metrics, payload)
    return metrics


def summary_metrics(summary: BatchSummary) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    elapsed_seconds = time.monotonic() - summary.started_monotonic
    _add_progress_metrics(
        metrics,
        processed=summary.processed_rows,
        failed=summary.failed_rows,
        skipped=summary.skipped_rows,
        selected=summary.selected_rows,
        elapsed_seconds=elapsed_seconds,
    )
    _add_timing_list_metrics(metrics, "duration", summary.timing_duration_ms)
    _add_timing_list_metrics(metrics, "ttft", summary.timing_ttft_ms)
    _add_timing_list_metrics(metrics, "time_to_response", summary.timing_time_to_response_ms)
    usage = summary.usage_totals.usage_block(processed_rows=summary.processed_rows)
    cache = summary.usage_totals.cache_block()
    _add_usage_cache_metrics(
        metrics, usage=usage, cache=cache, processed_rows=summary.processed_rows
    )
    return metrics


def merge_usage_totals_from_summaries(summaries: Sequence[Mapping[str, Any]]) -> BatchUsageTotals:
    totals = BatchUsageTotals()
    for summary in summaries:
        usage = _mapping(summary.get("usage"))
        cache = _mapping(summary.get("cache"))
        prior_groups = 1 if totals.rows_with_usage > 0 else 0
        totals.prompt_tokens = _add_complete(
            totals.prompt_tokens, _optional_int(usage, "prompt_tokens"), prior_rows=prior_groups
        )
        totals.completion_tokens = _add_complete(
            totals.completion_tokens,
            _optional_int(usage, "completion_tokens"),
            prior_rows=prior_groups,
        )
        totals.total_tokens = _add_complete(
            totals.total_tokens, _optional_int(usage, "total_tokens"), prior_rows=prior_groups
        )
        totals.reasoning_tokens = _add_complete(
            totals.reasoning_tokens,
            _optional_int(usage, "reasoning_tokens"),
            prior_rows=prior_groups,
        )
        totals.tool_use_prompt_tokens = _add_complete(
            totals.tool_use_prompt_tokens,
            _optional_int(usage, "tool_use_prompt_tokens"),
            prior_rows=prior_groups,
        )
        totals.tool_calls += _int(usage, "tool_calls")
        totals.rows_with_usage += _int(usage, "rows_with_usage")
        totals.cache_read_tokens = _add_complete(
            totals.cache_read_tokens,
            _optional_int(cache, "read_tokens"),
            prior_rows=prior_groups,
        )
        totals.cache_write_tokens = _add_complete(
            totals.cache_write_tokens,
            _optional_int(cache, "write_tokens"),
            prior_rows=prior_groups,
        )
        totals.rows_with_cache_activity += _int(cache, "rows_with_cache_activity")
    return totals


def _mapping(value: object) -> Mapping[str, Any] | None:
    return cast("Mapping[str, Any]", value) if isinstance(value, Mapping) else None


def _add_complete(
    current: int | None,
    observation: int | None,
    *,
    prior_rows: int,
) -> int | None:
    if prior_rows == 0:
        return observation
    if current is None or observation is None:
        return None
    return current + observation


def _without_none(values: dict[str, Any | None]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _optional_int(source: Mapping[str, Any] | None, key: str) -> int | None:
    return nonnegative_int_or_none(source.get(key)) if source is not None else None


def _int(source: Mapping[str, Any] | None, *keys: str) -> int:
    if source is None:
        return 0
    for key in keys:
        value = nonnegative_int_or_none(source.get(key))
        if value is not None:
            return value
    return 0


def _payload_int(payload: Mapping[str, Any], key: str) -> int:
    return nonnegative_int_or_none(payload.get(key)) or 0


def _number(value: object) -> float | None:
    number = finite_number_or_none(value)
    return float(number) if number is not None else None


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator * 100


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _set_if_not_none(values: dict[str, Any], key: str, value: Any | None) -> None:
    if value is not None:
        values[key] = value


def _trackio_config(
    options: StructuredBatchOptions,
    selected_rows: int,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    config = {
        "input": str(options.input_path),
        "output": str(options.output_path),
        "summary_output": str(options.summary_output_path)
        if options.summary_output_path is not None
        else None,
        "telemetry_output": str(options.telemetry_output_path)
        if options.telemetry_output_path is not None
        else None,
        "error_output": str(options.error_output_path)
        if options.error_output_path is not None
        else None,
        "model": options.model,
        "agent_card": options.agent_card_source,
        "agent": options.agent_name,
        "template": str(options.template_source) if options.template_source is not None else None,
        "schema": str(options.schema_source)
        if options.schema_source is not None
        else options.schema_model,
        "parallel": options.parallel,
        "limit": options.limit,
        "offset": options.offset,
        "sample": options.sample,
        "seed": options.seed,
        "id_field": options.id_field,
        "include_input": options.include_input,
        "resume": options.resume,
        "overwrite": options.overwrite,
        "selected_rows": selected_rows,
    }
    if extra is not None:
        config.update(dict(extra))
    return config


def _default_trackio_name(options: StructuredBatchOptions) -> str:
    return f"batch-{Path(options.output_path).stem}"


def _add_progress_metrics(
    metrics: dict[str, int | float],
    *,
    processed: int,
    failed: int,
    skipped: int,
    selected: int,
    elapsed_seconds: float,
) -> None:
    metrics["batch/processed_rows"] = processed
    metrics["batch/failed_rows"] = failed
    metrics["batch/skipped_rows"] = skipped
    metrics["batch/selected_rows"] = selected
    metrics["batch/progress_fraction"] = 1.0 if selected == 0 else processed / selected
    metrics["batch/elapsed_seconds"] = elapsed_seconds
    if elapsed_seconds > 0:
        metrics["batch/rows_per_second"] = processed / elapsed_seconds
    if processed > 0:
        metrics["batch/success_rate"] = (processed - failed) / processed
        metrics["batch/error_rate"] = failed / processed


def _add_timing_list_metrics(
    metrics: dict[str, int | float],
    name: str,
    values: Sequence[float],
) -> None:
    if not values:
        return
    prefix = f"batch/timing/{name}_ms"
    metrics[f"{prefix}_mean"] = statistics.fmean(values)
    metrics[f"{prefix}_median"] = statistics.median(values)
    metrics[f"{prefix}_min"] = min(values)
    metrics[f"{prefix}_max"] = max(values)


def _add_timing_payload_metrics(
    metrics: dict[str, int | float], payload: Mapping[str, Any]
) -> None:
    timing = _mapping(payload.get("timing_ms"))
    if timing is None:
        return
    for name in ("duration", "ttft", "time_to_response"):
        part = _mapping(timing.get(name))
        if part is None:
            continue
        prefix = f"batch/timing/{name}_ms"
        for stat in ("mean", "median", "median_approx", "min", "max"):
            value = _number(part.get(stat))
            if value is not None:
                metric_stat = "median" if stat == "median_approx" else stat
                metrics[f"{prefix}_{metric_stat}"] = value


def _add_usage_payload_metrics(metrics: dict[str, int | float], payload: Mapping[str, Any]) -> None:
    usage = _mapping(payload.get("usage")) or {}
    cache = _mapping(payload.get("cache")) or {}
    processed = _payload_int(payload, "processed_rows")
    _add_usage_cache_metrics(metrics, usage=usage, cache=cache, processed_rows=processed)


def _add_usage_cache_metrics(
    metrics: dict[str, int | float],
    *,
    usage: Mapping[str, Any],
    cache: Mapping[str, Any],
    processed_rows: int,
) -> None:
    _copy_numeric_metrics(
        metrics,
        usage,
        {
            "prompt_tokens": "batch/usage/prompt_tokens",
            "completion_tokens": "batch/usage/completion_tokens",
            "total_tokens": "batch/usage/total_tokens",
            "reasoning_tokens": "batch/usage/reasoning_tokens",
            "tool_use_prompt_tokens": "batch/usage/tool_use_prompt_tokens",
            "tool_calls": "batch/usage/tool_calls",
            "rows_with_usage": "batch/usage/rows_with_usage",
            "usage_coverage_percent": "batch/usage/usage_coverage_percent",
        },
    )
    _copy_numeric_metrics(
        metrics,
        cache,
        {
            "read_tokens": "batch/cache/read_tokens",
            "write_tokens": "batch/cache/write_tokens",
            "rows_with_cache_activity": "batch/cache/rows_with_cache_activity",
            "row_cache_activity_percent": "batch/cache/row_cache_activity_percent",
        },
    )
    rows_with_usage = _int(usage, "rows_with_usage")
    denominator = rows_with_usage or processed_rows
    _add_per_row_metric(
        metrics, usage, "prompt_tokens", "batch/usage/prompt_tokens_per_row", denominator
    )
    _add_per_row_metric(
        metrics,
        usage,
        "completion_tokens",
        "batch/usage/completion_tokens_per_row",
        denominator,
    )
    _add_per_row_metric(
        metrics, usage, "total_tokens", "batch/usage/total_tokens_per_row", denominator
    )
    _add_per_row_metric(
        metrics, usage, "reasoning_tokens", "batch/usage/reasoning_tokens_per_row", denominator
    )
    _add_per_row_metric(metrics, usage, "tool_calls", "batch/usage/tool_calls_per_row", denominator)
    _add_per_row_metric(
        metrics, cache, "served_tokens", "batch/cache/served_tokens_per_row", denominator
    )
    _add_per_row_metric(
        metrics, cache, "write_tokens", "batch/cache/write_tokens_per_row", denominator
    )


def _copy_numeric_metrics(
    metrics: dict[str, int | float],
    source: Mapping[str, Any],
    names: Mapping[str, str],
) -> None:
    for source_key, metric_key in names.items():
        value = finite_number_or_none(source.get(source_key))
        if value is not None:
            metrics[metric_key] = value


def _add_per_row_metric(
    metrics: dict[str, int | float],
    source: Mapping[str, Any],
    source_key: str,
    metric_key: str,
    denominator: int,
) -> None:
    value = finite_number_or_none(source.get(source_key))
    if value is not None and denominator > 0:
        metrics[metric_key] = value / denominator
