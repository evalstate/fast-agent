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

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    billing_tokens: int = 0
    reasoning_tokens: int = 0
    tool_use_tokens: int = 0
    tool_calls: int = 0
    rows_with_usage: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_hit_tokens: int = 0
    effective_input_tokens: int = 0
    rows_with_cache_activity: int = 0

    def add_row_usage(self, usage: Mapping[str, Any] | None) -> None:
        """Add normalized fast-agent usage telemetry for one processed row."""

        source = _usage_source(usage)
        if source is None:
            return

        self.rows_with_usage += 1
        is_summary = _mapping(usage.get("summary")) is source if usage is not None else False
        cache_source = source if is_summary else _mapping(source.get("cache_usage"))

        if is_summary:
            input_tokens = _int(source, "cumulative_input_tokens", "input_tokens")
            output_tokens = _int(source, "cumulative_output_tokens", "output_tokens")
            total_tokens = _int(source, "cumulative_total_tokens", "total_tokens")
            billing_tokens = _int(source, "cumulative_billing_tokens", "billing_tokens")
            tool_use_tokens = _int(source, "cumulative_tool_use_tokens", "tool_use_tokens")
            reasoning_tokens = _int(source, "cumulative_reasoning_tokens", "reasoning_tokens")
            tool_calls = _int(source, "cumulative_tool_calls", "tool_calls")
            cache_read_tokens = _int(source, "cumulative_cache_read_tokens", "cache_read_tokens")
            cache_write_tokens = _int(source, "cumulative_cache_write_tokens", "cache_write_tokens")
            cache_hit_tokens = _int(source, "cumulative_cache_hit_tokens", "cache_hit_tokens")
            effective_input_tokens = _int(
                source, "cumulative_effective_input_tokens", "effective_input_tokens"
            )
        else:
            input_tokens = _int(source, "display_input_tokens", "input_tokens")
            output_tokens = _int(source, "output_tokens")
            total_tokens = _int(source, "total_tokens")
            billing_tokens = _int(source, "billing_tokens")
            tool_use_tokens = _int(source, "tool_use_tokens")
            reasoning_tokens = _int(source, "reasoning_tokens")
            tool_calls = _int(source, "tool_calls")
            cache_read_tokens = _int(cache_source, "cache_read_tokens")
            cache_write_tokens = _int(cache_source, "cache_write_tokens")
            cache_hit_tokens = _int(cache_source, "cache_hit_tokens")
            effective_input_tokens = _int(source, "effective_input_tokens")

        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens
        if billing_tokens == 0:
            billing_tokens = total_tokens

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens
        self.billing_tokens += billing_tokens
        self.reasoning_tokens += reasoning_tokens
        self.tool_use_tokens += tool_use_tokens
        self.tool_calls += tool_calls
        self.cache_read_tokens += cache_read_tokens
        self.cache_write_tokens += cache_write_tokens
        self.cache_hit_tokens += cache_hit_tokens
        self.effective_input_tokens += effective_input_tokens
        if cache_read_tokens + cache_write_tokens + cache_hit_tokens > 0:
            self.rows_with_cache_activity += 1

    @property
    def served_tokens(self) -> int:
        return self.cache_read_tokens + self.cache_hit_tokens

    @property
    def activity_tokens(self) -> int:
        return self.served_tokens + self.cache_write_tokens

    @property
    def input_context_tokens(self) -> int:
        return max(
            self.input_tokens,
            self.effective_input_tokens + self.served_tokens + self.cache_write_tokens,
        )

    def usage_block(self, *, processed_rows: int) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "billing_tokens": self.billing_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "tool_use_tokens": self.tool_use_tokens,
            "tool_calls": self.tool_calls,
            "rows_with_usage": self.rows_with_usage,
            "usage_coverage_percent": _percent(self.rows_with_usage, processed_rows),
        }

    def cache_block(self) -> dict[str, Any]:
        input_context_tokens = self.input_context_tokens
        non_cached_input_tokens = max(0, input_context_tokens - self.served_tokens)
        return {
            "read_tokens": self.cache_read_tokens,
            "write_tokens": self.cache_write_tokens,
            "hit_tokens": self.cache_hit_tokens,
            "served_tokens": self.served_tokens,
            "activity_tokens": self.activity_tokens,
            "effective_input_tokens": self.effective_input_tokens,
            "hit_rate_percent": _percent(self.served_tokens, input_context_tokens),
            "write_rate_percent": _percent(self.cache_write_tokens, input_context_tokens),
            "activity_rate_percent": _percent(self.activity_tokens, input_context_tokens),
            "rows_with_cache_activity": self.rows_with_cache_activity,
            "row_cache_activity_percent": _percent(
                self.rows_with_cache_activity, self.rows_with_usage
            ),
            "non_cached_input_tokens": non_cached_input_tokens,
            "served_to_effective_input_ratio": _ratio(
                self.served_tokens, self.effective_input_tokens
            ),
        }


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
        totals.input_tokens += _int(usage, "input_tokens")
        totals.output_tokens += _int(usage, "output_tokens")
        totals.total_tokens += _int(usage, "total_tokens")
        totals.billing_tokens += _int(usage, "billing_tokens")
        totals.reasoning_tokens += _int(usage, "reasoning_tokens")
        totals.tool_use_tokens += _int(usage, "tool_use_tokens")
        totals.tool_calls += _int(usage, "tool_calls")
        totals.rows_with_usage += _int(usage, "rows_with_usage")
        totals.cache_read_tokens += _int(cache, "read_tokens")
        totals.cache_write_tokens += _int(cache, "write_tokens")
        totals.cache_hit_tokens += _int(cache, "hit_tokens")
        totals.effective_input_tokens += _int(cache, "effective_input_tokens")
        totals.rows_with_cache_activity += _int(cache, "rows_with_cache_activity")
    return totals


def _usage_source(usage: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if usage is None:
        return None
    turn = _mapping(usage.get("turn"))
    if turn is not None:
        return turn
    summary = _mapping(usage.get("summary"))
    if summary is not None:
        return summary
    if any(
        key in usage
        for key in (
            "input_tokens",
            "display_input_tokens",
            "cumulative_input_tokens",
            "total_tokens",
            "cumulative_billing_tokens",
        )
    ):
        return usage
    return None


def _mapping(value: object) -> Mapping[str, Any] | None:
    return cast("Mapping[str, Any]", value) if isinstance(value, Mapping) else None


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
            "input_tokens": "batch/usage/input_tokens",
            "output_tokens": "batch/usage/output_tokens",
            "total_tokens": "batch/usage/total_tokens",
            "billing_tokens": "batch/usage/billing_tokens",
            "reasoning_tokens": "batch/usage/reasoning_tokens",
            "tool_use_tokens": "batch/usage/tool_use_tokens",
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
            "hit_tokens": "batch/cache/hit_tokens",
            "served_tokens": "batch/cache/served_tokens",
            "activity_tokens": "batch/cache/activity_tokens",
            "effective_input_tokens": "batch/cache/effective_input_tokens",
            "hit_rate_percent": "batch/cache/hit_rate_percent",
            "write_rate_percent": "batch/cache/write_rate_percent",
            "activity_rate_percent": "batch/cache/activity_rate_percent",
            "rows_with_cache_activity": "batch/cache/rows_with_cache_activity",
            "row_cache_activity_percent": "batch/cache/row_cache_activity_percent",
            "non_cached_input_tokens": "batch/cache/non_cached_input_tokens",
            "served_to_effective_input_ratio": "batch/cache/served_to_effective_input_ratio",
        },
    )
    rows_with_usage = _int(usage, "rows_with_usage")
    denominator = rows_with_usage or processed_rows
    _add_per_row_metric(
        metrics, usage, "input_tokens", "batch/usage/input_tokens_per_row", denominator
    )
    _add_per_row_metric(
        metrics, usage, "output_tokens", "batch/usage/output_tokens_per_row", denominator
    )
    _add_per_row_metric(
        metrics, usage, "billing_tokens", "batch/usage/billing_tokens_per_row", denominator
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
