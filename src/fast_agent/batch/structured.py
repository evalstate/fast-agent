"""Batch runner for row-oriented agent/model jobs."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, TypeAlias, cast

from pydantic import BaseModel

from fast_agent.batch.input import (
    RowCandidate,
    RowError,
    count_parquet_input_rows,
    is_parquet_input_source,
    iter_input_rows,
    select_rows,
)
from fast_agent.batch.output import (
    ensure_parent,
    error_envelope,
    success_envelope,
    write_jsonl_record,
)
from fast_agent.batch.resume import load_completed_ids
from fast_agent.batch.summary import BatchSummary
from fast_agent.batch.template import DEFAULT_ROW_TEMPLATE, render_row_template
from fast_agent.batch.traces import BatchTraceOptions, BatchTraceRecorder
from fast_agent.cli.runtime.request_builders import resolve_default_instruction
from fast_agent.constants import FAST_AGENT_TIMING, FAST_AGENT_USAGE
from fast_agent.core.instruction_source import resolve_instruction_source
from fast_agent.io.source_resolver import read_text_source
from fast_agent.llm.request_params import BatchRequestContext, RequestParams
from fast_agent.llm.structured_schema import (
    StructuredSchemaSource,
    load_json_schema_file,
    load_pydantic_model,
)
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.session.trace_export_errors import SessionExportUploadError

if TYPE_CHECKING:
    from fast_agent.core.fastagent import FastAgent
    from fast_agent.interfaces import AgentProtocol


@dataclass(frozen=True)
class StructuredBatchOptions:
    input_path: str | Path
    output_path: Path
    prompt_template: str | None = None
    schema_source: str | Path | None = None
    schema_model: str | None = None
    template_source: str | Path | None = None
    instruction_source: str | Path | None = None
    model: str | None = None
    include_input: bool = False
    limit: int | None = None
    offset: int | None = None
    sample: int | None = None
    sql: str | None = None
    seed: int | None = None
    resume: bool = False
    overwrite: bool = False
    id_field: str | None = None
    max_errors: int | None = None
    error_output_path: Path | None = None
    telemetry_output_path: Path | None = None
    summary_output_path: Path | None = None
    final_summary: bool = True
    environment_dir: Path | None = None
    shell_runtime: bool = False
    agent_card_source: str | None = None
    agent_name: str | None = None
    export_traces_path: Path | None = None
    hf_dataset: str | None = None
    hf_dataset_path: str | None = None
    parallel: int | None = None
    work_dir: Path | None = None
    keep_temp: bool = False
    progress_every: int | None = None
    progress: bool = True
    progress_label: str | None = None


@dataclass(frozen=True)
class BatchShard:
    index: int
    offset: int
    limit: int
    output_path: Path
    error_output_path: Path | None
    telemetry_output_path: Path | None
    summary_output_path: Path


@dataclass(frozen=True)
class ParallelManifest:
    input_rows: int
    selected_rows: int
    shards: list[BatchShard]


LoadedSchemaSource: TypeAlias = StructuredSchemaSource


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json_schema(path: str | Path) -> dict[str, Any]:
    return load_json_schema_file(path)


def load_text_template(path: str | Path) -> str:
    return read_text_source(path, label="batch template")


def load_schema_source(options: StructuredBatchOptions) -> LoadedSchemaSource | None:
    if options.agent_card_source is not None and options.instruction_source is not None:
        raise ValueError("--agent-card and --instruction cannot be used together")
    if options.agent_name is not None and options.agent_card_source is None:
        raise ValueError("--agent requires --agent-card")
    if options.schema_source is not None and options.schema_model is not None:
        raise ValueError("--json-schema and --schema-model cannot be used together")
    if options.hf_dataset_path is not None and options.hf_dataset is None:
        raise ValueError("--hf-dataset-path requires --hf-dataset")
    if options.hf_dataset is not None and options.export_traces_path is None:
        raise ValueError("--hf-dataset requires --export-traces")
    if options.sql is not None:
        if not is_parquet_input_source(options.input_path):
            raise ValueError("--sql is only supported for parquet input")
        if options.limit is not None or options.offset is not None or options.sample is not None:
            raise ValueError("--sql cannot be used with --limit, --offset, or --sample")
        if options.parallel is not None and options.parallel > 1:
            raise ValueError("--sql cannot be used with --parallel")
    if options.schema_model is not None:
        return load_pydantic_model(options.schema_model)
    if options.schema_source is not None:
        return load_json_schema(options.schema_source)
    return None


def _identity_for_candidate(candidate: RowCandidate, id_field: str | None) -> tuple[str | int, RowError | None]:
    if id_field is None:
        return candidate.row_number, None
    row = candidate.row
    if row is None:
        return candidate.row_number, None
    if id_field not in row:
        return candidate.row_number, RowError(
            "MissingIdField",
            f"Missing id field '{id_field}'",
        )
    return str(row[id_field]), None


def _extract_json_channel(response: Any, channel_name: str) -> dict[str, Any] | None:
    channels = response.channels
    if not isinstance(channels, Mapping):
        return None
    blocks = channels.get(channel_name)
    if not blocks:
        return None
    text = get_text(blocks[0])
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_timing(response: Any) -> dict[str, Any] | None:
    return _extract_json_channel(response, FAST_AGENT_TIMING)


def _extract_usage(response: Any) -> dict[str, Any] | None:
    usage = _extract_json_channel(response, FAST_AGENT_USAGE)
    if usage is None:
        return None
    if "turn" not in usage and "raw_usage" not in usage:
        return usage
    return {
        key: value
        for key in ("turn", "raw_usage")
        if (value := usage.get(key)) is not None
    }


def _write_optional_failure(
    error_handle: TextIO | None,
    record: dict[str, Any],
) -> None:
    if error_handle is not None:
        write_jsonl_record(error_handle, record)


def _write_optional_telemetry(
    telemetry_handle: TextIO | None,
    *,
    identity: str | int,
    row_number: int,
    ok: bool,
    timing: dict[str, Any] | None,
    usage: dict[str, Any] | None = None,
) -> None:
    if telemetry_handle is None:
        return
    write_jsonl_record(
        telemetry_handle,
        {
            "id": identity,
            "row_number": row_number,
            "ok": ok,
            "timing": timing or {},
            "usage": usage or {},
        },
    )


def _emit_progress(options: StructuredBatchOptions, message: str) -> None:
    if not options.progress:
        return
    label = f"{options.progress_label} " if options.progress_label else ""
    print(f"batch: {label}{message}", file=sys.stderr, flush=True)


def _emit_row_progress(options: StructuredBatchOptions, summary: BatchSummary) -> None:
    every = options.progress_every
    if every is None or every <= 0 or summary.processed_rows % every != 0:
        return
    _emit_progress(
        options,
        (
            f"processed={summary.processed_rows} "
            f"failed={summary.failed_rows} "
            f"skipped={summary.skipped_rows}"
        ),
    )


def _can_push_down_input_selection(options: StructuredBatchOptions) -> bool:
    return (
        options.sql is None
        and options.sample is None
        and not options.resume
        and is_parquet_input_source(options.input_path)
        and (options.offset is not None or options.limit is not None)
    )


def _load_input_candidates(options: StructuredBatchOptions) -> tuple[int, list[RowCandidate]]:
    if options.sql is not None:
        selected = list(iter_input_rows(options.input_path, sql=options.sql))
        return len(selected), selected
    if _can_push_down_input_selection(options):
        selected = list(
            iter_input_rows(
                options.input_path,
                offset=options.offset,
                limit=options.limit,
            )
        )
        return count_parquet_input_rows(options.input_path), selected
    all_candidates = list(iter_input_rows(options.input_path))
    selected = select_rows(
        all_candidates,
        offset=options.offset,
        sample=options.sample,
        seed=options.seed,
        limit=options.limit,
    )
    return len(all_candidates), selected


def _load_parallel_input_counts(options: StructuredBatchOptions) -> tuple[int, int]:
    if options.sample is None and is_parquet_input_source(options.input_path):
        input_rows = count_parquet_input_rows(options.input_path)
        offset = options.offset or 0
        available = max(0, input_rows - offset)
        selected_rows = available if options.limit is None else min(options.limit, available)
        return input_rows, selected_rows

    input_rows, selected = _load_input_candidates(options)
    return input_rows, len(selected)


def _prepare_output_files(options: StructuredBatchOptions) -> None:
    if options.resume and options.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    _reject_duplicate_output_paths(options)
    if options.output_path.exists() and not options.resume and not options.overwrite:
        raise ValueError(
            f"Output file {options.output_path} already exists; use --resume or --overwrite"
        )

    for path in (
        options.output_path,
        options.error_output_path,
        options.telemetry_output_path,
        options.summary_output_path,
    ):
        if path is not None:
            ensure_parent(path)


def _reject_duplicate_output_paths(options: StructuredBatchOptions) -> None:
    configured_paths = {
        "--output": options.output_path,
        "--error-output": options.error_output_path,
        "--telemetry-output": options.telemetry_output_path,
        "--summary-output": options.summary_output_path,
    }
    resolved_paths: dict[Path, str] = {}
    for label, path in configured_paths.items():
        if path is None:
            continue
        resolved = path.resolve(strict=False)
        existing_label = resolved_paths.get(resolved)
        if existing_label is not None:
            raise ValueError(
                f"{label} must not point to the same file as {existing_label}: {path}"
            )
        resolved_paths[resolved] = label


async def run_structured_batch(options: StructuredBatchOptions) -> dict[str, Any]:
    """Run a batch job and return the summary payload."""
    _prepare_output_files(options)

    schema_source = load_schema_source(options)
    template = (
        load_text_template(options.template_source)
        if options.template_source is not None
        else options.prompt_template if options.prompt_template is not None else DEFAULT_ROW_TEMPLATE
    )
    if options.agent_card_source is None:
        instruction: str | None = (
            resolve_instruction_source(options.instruction_source)
            if options.instruction_source is not None
            else resolve_default_instruction(options.model, "interactive")
        )
    else:
        instruction = None

    input_rows, selected = _load_input_candidates(options)
    completed_ids = load_completed_ids(options.output_path) if options.resume else set()

    started_at = utc_now_iso()
    summary = BatchSummary(
        input_rows=input_rows,
        selected_rows=len(selected),
        started_at=started_at,
        metadata={
            "model": options.model,
            "input": str(options.input_path),
            "sql": options.sql,
            "output": str(options.output_path),
            "schema": str(options.schema_source) if options.schema_source is not None else None,
            "schema_model": options.schema_model,
            "instruction": str(options.instruction_source) if options.instruction_source else None,
            "agent_card": options.agent_card_source,
            "agent": None,
            "template": str(options.template_source) if options.template_source else "<default>",
            "shell_runtime": options.shell_runtime,
            "output_mode": "structured" if schema_source is not None else "text",
            "export_traces": str(options.export_traces_path) if options.export_traces_path else None,
            "hf_dataset": options.hf_dataset,
            "hf_dataset_path": options.hf_dataset_path,
        },
    )
    _emit_progress(
        options,
        f"start selected_rows={len(selected)} output={options.output_path}",
    )

    from fast_agent import FastAgent

    fast = FastAgent(
        name="batch",
        parse_cli_args=False,
        ignore_unknown_args=True,
        quiet=True,
        environment_dir=options.environment_dir,
    )
    if options.model:
        fast.args.model = options.model

    target_agent_name = await _configure_batch_worker(fast, options, instruction)
    if options.agent_card_source is not None:
        summary.metadata["agent"] = target_agent_name

    if options.shell_runtime:
        await fast.app.initialize()
        setattr(fast.app.context, "shell_runtime", True)

    output_mode = "a" if options.resume else "w"
    if options.overwrite:
        output_mode = "w"

    async with fast.run() as agent_app:
        worker = agent_app._agent(target_agent_name)
        trace_recorder = _configure_trace_recorder(worker, options, summary.metadata)
        with options.output_path.open(output_mode, encoding="utf-8") as output_handle:
            with _optional_jsonl_handle(options.error_output_path, "a" if options.resume else "w") as error_handle:
                with _optional_jsonl_handle(
                    options.telemetry_output_path,
                    "a" if options.resume else "w",
                ) as telemetry_handle:
                    for candidate in selected:
                        if _max_errors_reached(summary.failed_rows, options.max_errors):
                            break
                        identity, id_error = _identity_for_candidate(candidate, options.id_field)
                        if str(identity) in completed_ids:
                            summary.skipped_rows += 1
                            continue

                        row_error = candidate.error or id_error
                        if row_error is None and candidate.row is not None:
                            rendered, template_error = render_row_template(template, candidate.row)
                            row_error = template_error
                        else:
                            rendered = None

                        if row_error is not None:
                            record = error_envelope(
                                identity=identity,
                                row_number=candidate.row_number,
                                error=row_error,
                                row=candidate.row,
                                include_input=options.include_input,
                            )
                            write_jsonl_record(output_handle, record)
                            _write_optional_failure(error_handle, record)
                            _write_optional_telemetry(
                                telemetry_handle,
                                identity=identity,
                                row_number=candidate.row_number,
                                ok=False,
                                timing=None,
                            )
                            summary.processed_rows += 1
                            summary.failed_rows += 1
                            _emit_row_progress(options, summary)
                            if trace_recorder is not None:
                                trace_recorder.record_row_without_trace(
                                    row_number=candidate.row_number,
                                    identity=identity,
                                    ok=False,
                                    error_type=row_error.type,
                                    error_message=row_error.message,
                                )
                            if _max_errors_reached(summary.failed_rows, options.max_errors):
                                break
                            continue

                        assert rendered is not None
                        assert candidate.row is not None
                        if trace_recorder is not None:
                            trace_recorder.start_row(
                                row_number=candidate.row_number,
                                identity=identity,
                                rendered=rendered,
                            )
                        try:
                            parsed, response = await _row_call(
                                worker,
                                rendered=rendered,
                                schema_source=schema_source,
                                batch_context=BatchRequestContext(
                                    row_number=candidate.row_number,
                                    identity=identity,
                                ),
                            )
                            timing = _extract_timing(response)
                            usage = _extract_usage(response)
                            summary.add_timing(timing)
                            if parsed is None:
                                record = error_envelope(
                                    identity=identity,
                                    row_number=candidate.row_number,
                                    error=RowError(
                                        "StructuredOutputError",
                                        "Model response did not satisfy the JSON schema",
                                    ),
                                    row=candidate.row,
                                    include_input=options.include_input,
                                )
                                write_jsonl_record(output_handle, record)
                                _write_optional_failure(error_handle, record)
                                _write_optional_telemetry(
                                    telemetry_handle,
                                    identity=identity,
                                    row_number=candidate.row_number,
                                    ok=False,
                                    timing=timing,
                                    usage=usage,
                                )
                                summary.processed_rows += 1
                                summary.failed_rows += 1
                                _emit_row_progress(options, summary)
                                if trace_recorder is not None:
                                    trace_recorder.finish_row(
                                        ok=False,
                                        response=response,
                                        error_type="StructuredOutputError",
                                        error_message="Model response did not satisfy the JSON schema",
                                    )
                                if _max_errors_reached(summary.failed_rows, options.max_errors):
                                    break
                                continue
                            record = success_envelope(
                                identity=identity,
                                row_number=candidate.row_number,
                                result=_json_result(parsed),
                                row=candidate.row,
                                include_input=options.include_input,
                            )
                            write_jsonl_record(output_handle, record)
                            _write_optional_telemetry(
                                telemetry_handle,
                                identity=identity,
                                row_number=candidate.row_number,
                                ok=True,
                                timing=timing,
                                usage=usage,
                            )
                            summary.processed_rows += 1
                            _emit_row_progress(options, summary)
                            if trace_recorder is not None:
                                trace_recorder.finish_row(ok=True, response=response)
                        except Exception as exc:
                            record = error_envelope(
                                identity=identity,
                                row_number=candidate.row_number,
                                error=RowError(type(exc).__name__, str(exc)),
                                row=candidate.row,
                                include_input=options.include_input,
                            )
                            write_jsonl_record(output_handle, record)
                            _write_optional_failure(error_handle, record)
                            _write_optional_telemetry(
                                telemetry_handle,
                                identity=identity,
                                row_number=candidate.row_number,
                                ok=False,
                                timing=None,
                            )
                            summary.processed_rows += 1
                            summary.failed_rows += 1
                            _emit_row_progress(options, summary)
                            if trace_recorder is not None:
                                trace_recorder.finish_row(
                                    ok=False,
                                    error_type=type(exc).__name__,
                                    error_message=str(exc),
                                )
                            if _max_errors_reached(summary.failed_rows, options.max_errors):
                                break

        if trace_recorder is not None:
            summary.metadata["trace_run_id"] = trace_recorder.run_id
            if options.hf_dataset is not None:
                try:
                    upload = trace_recorder.upload_to_hf_dataset(
                        dataset_repo=options.hf_dataset,
                        dataset_path=options.hf_dataset_path,
                    )
                except SessionExportUploadError as exc:
                    raise ValueError(str(exc)) from exc
                summary.metadata["hf_dataset_upload"] = {
                    "repo_id": upload.repo_id,
                    "path_in_repo": upload.path_in_repo,
                    "commit_url": upload.commit_url,
                    "file_url": upload.file_url,
                }

    completed_at = utc_now_iso()
    payload = summary.to_dict(completed_at)
    if options.summary_output_path is not None:
        options.summary_output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    _emit_progress(
        options,
        (
            "complete "
            f"processed={payload['processed_rows']} "
            f"failed={payload['failed_rows']} "
            f"skipped={payload['skipped_rows']}"
        ),
    )
    return payload


async def run_parallel_structured_batch(options: StructuredBatchOptions) -> dict[str, Any]:
    """Run a batch job across local shard workers and merge their JSONL outputs."""
    parallel = options.parallel or 1
    if parallel <= 1:
        return await run_structured_batch(options)

    _validate_parallel_options(options, parallel)
    work_dir = _resolve_parallel_work_dir(options)
    _validate_parallel_final_outputs_outside_work_dir(options, work_dir)
    _prepare_parallel_output_files(options)

    input_rows, selected_rows = _load_parallel_input_counts(options)
    started_at = utc_now_iso()
    started_monotonic = time.monotonic()
    _prepare_parallel_work_dir(work_dir, resume=options.resume, overwrite=options.overwrite)

    if options.resume:
        manifest = _load_parallel_manifest(options, work_dir, input_rows=input_rows)
        shards = manifest.shards
        selected_rows = manifest.selected_rows

    if not selected_rows:
        _write_empty_parallel_outputs(options)
        payload = _empty_parallel_summary(options, started_at, input_rows, work_dir)
        _write_parallel_summary(options, payload)
        _cleanup_parallel_work_dir(options, work_dir)
        return payload

    if not options.resume:
        shards = _plan_parallel_shards(options, work_dir, selected_rows, parallel)
        _write_parallel_manifest(options, work_dir, shards, input_rows, selected_rows)
    _emit_progress(
        options,
        f"planned {len(shards)} shards for {selected_rows} selected rows work_dir={work_dir}",
    )

    shard_tasks = []
    for shard in shards:
        _emit_progress(
            options,
            (
                f"shard {shard.index} start offset={shard.offset} "
                f"limit={shard.limit} output={shard.output_path}"
            ),
        )
        shard_tasks.append(run_structured_batch(_shard_options(options, shard)))

    try:
        shard_summaries = await asyncio.gather(*shard_tasks)
    except Exception:
        _emit_progress(options, f"failed; kept shard outputs in {work_dir}")
        raise

    _emit_progress(options, f"merging {len(shards)} shards into {options.output_path}")
    _merge_jsonl_shards([shard.output_path for shard in shards], options.output_path, work_dir)
    if options.error_output_path is not None:
        _merge_jsonl_shards(
            [shard.error_output_path for shard in shards if shard.error_output_path is not None],
            options.error_output_path,
            work_dir,
        )
    if options.telemetry_output_path is not None:
        _merge_jsonl_shards(
            [
                shard.telemetry_output_path
                for shard in shards
                if shard.telemetry_output_path is not None
            ],
            options.telemetry_output_path,
            work_dir,
        )

    payload = _merge_parallel_summaries(
        options=options,
        started_at=started_at,
        completed_at=utc_now_iso(),
        duration_ms=round((time.monotonic() - started_monotonic) * 1000, 2),
        input_rows=input_rows,
        selected_rows=selected_rows,
        work_dir=work_dir,
        shards=shards,
        shard_summaries=shard_summaries,
    )
    _write_parallel_summary(options, payload)
    _emit_progress(
        options,
        (
            "complete "
            f"processed={payload['processed_rows']} "
            f"failed={payload['failed_rows']} "
            f"skipped={payload['skipped_rows']}"
        ),
    )
    _cleanup_parallel_work_dir(options, work_dir)
    return payload


def _validate_parallel_options(options: StructuredBatchOptions, parallel: int) -> None:
    if parallel < 1:
        raise ValueError("--parallel must be greater than zero")
    if options.resume and options.work_dir is None:
        raise ValueError("--parallel --resume requires --work-dir from the interrupted run")
    if options.sample is not None:
        raise ValueError("--parallel cannot be used with --sample yet")
    if options.max_errors is not None:
        raise ValueError("--parallel cannot be used with --max-errors yet")
    if options.export_traces_path is not None:
        raise ValueError("--parallel cannot be used with --export-traces yet")


def _prepare_parallel_output_files(options: StructuredBatchOptions) -> None:
    if options.resume and options.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    _reject_duplicate_output_paths(options)
    if options.output_path.exists():
        if options.resume and not options.overwrite:
            raise ValueError(
                "--parallel --resume resumes shard work directories, not an existing final output; "
                "move the output file or use --overwrite"
            )
        if not options.overwrite:
            raise ValueError(
                f"Output file {options.output_path} already exists; use --resume or --overwrite"
            )
    for path in (
        options.output_path,
        options.error_output_path,
        options.telemetry_output_path,
        options.summary_output_path,
    ):
        if path is not None:
            ensure_parent(path)


def _resolve_parallel_work_dir(options: StructuredBatchOptions) -> Path:
    if options.work_dir is not None:
        return options.work_dir
    run_id = f"{utc_now_iso().replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:8]}"
    return options.output_path.parent / f".{options.output_path.name}.batch" / run_id


def _validate_parallel_final_outputs_outside_work_dir(
    options: StructuredBatchOptions,
    work_dir: Path,
) -> None:
    work_dir_path = work_dir.resolve()
    final_paths = (
        ("--output", options.output_path),
        ("--error-output", options.error_output_path),
        ("--telemetry-output", options.telemetry_output_path),
        ("--summary-output", options.summary_output_path),
    )
    for option_name, path in final_paths:
        if path is not None and path.resolve().is_relative_to(work_dir_path):
            raise ValueError(f"{option_name} must be outside --work-dir for parallel batches")


def _prepare_parallel_work_dir(work_dir: Path, *, resume: bool, overwrite: bool) -> None:
    if resume:
        if not work_dir.exists():
            raise ValueError(f"Work directory {work_dir} does not exist")
        if not (work_dir / "manifest.json").exists():
            raise ValueError(f"Work directory {work_dir} does not contain manifest.json")
        return
    if work_dir.exists() and any(work_dir.iterdir()):
        if not overwrite:
            raise ValueError(f"Work directory {work_dir} already exists; use --resume or --overwrite")
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)


def _plan_parallel_shards(
    options: StructuredBatchOptions,
    work_dir: Path,
    selected_rows: int,
    parallel: int,
) -> list[BatchShard]:
    shard_count = min(parallel, selected_rows)
    base = selected_rows // shard_count
    remainder = selected_rows % shard_count
    global_offset = options.offset or 0
    relative_offset = 0
    width = max(3, len(str(shard_count - 1)))
    shards: list[BatchShard] = []
    for index in range(shard_count):
        limit = base + (1 if index < remainder else 0)
        suffix = f"part-{index:0{width}d}"
        shards.append(
            BatchShard(
                index=index,
                offset=global_offset + relative_offset,
                limit=limit,
                output_path=work_dir / f"{suffix}.jsonl",
                error_output_path=(
                    work_dir / f"errors.{suffix}.jsonl"
                    if options.error_output_path is not None
                    else None
                ),
                telemetry_output_path=(
                    work_dir / f"telemetry.{suffix}.jsonl"
                    if options.telemetry_output_path is not None
                    else None
                ),
                summary_output_path=work_dir / f"summary.{suffix}.json",
            )
        )
        relative_offset += limit
    return shards


def _shard_options(options: StructuredBatchOptions, shard: BatchShard) -> StructuredBatchOptions:
    return replace(
        options,
        output_path=shard.output_path,
        offset=shard.offset,
        limit=shard.limit,
        sample=None,
        seed=None,
        resume=options.resume,
        overwrite=not options.resume,
        error_output_path=shard.error_output_path,
        telemetry_output_path=shard.telemetry_output_path,
        summary_output_path=shard.summary_output_path,
        final_summary=False,
        parallel=None,
        work_dir=None,
        keep_temp=True,
        progress_label=f"shard {shard.index}",
    )


def _write_parallel_manifest(
    options: StructuredBatchOptions,
    work_dir: Path,
    shards: list[BatchShard],
    input_rows: int,
    selected_rows: int,
) -> None:
    manifest = {
        "input": _input_source_identity(options.input_path),
        "output": str(options.output_path),
        "parallel": options.parallel,
        "input_rows": input_rows,
        "selected_rows": selected_rows,
        "created_at": utc_now_iso(),
        "shards": [
            {
                "index": shard.index,
                "offset": shard.offset,
                "limit": shard.limit,
                "output": str(shard.output_path),
                "error_output": str(shard.error_output_path)
                if shard.error_output_path is not None
                else None,
                "telemetry_output": str(shard.telemetry_output_path)
                if shard.telemetry_output_path is not None
                else None,
                "summary_output": str(shard.summary_output_path),
            }
            for shard in shards
        ],
    }
    (work_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_parallel_manifest(
    options: StructuredBatchOptions,
    work_dir: Path,
    *,
    input_rows: int,
) -> ParallelManifest:
    manifest_path = work_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Work directory {work_dir} contains invalid manifest.json") from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"Work directory {work_dir} contains invalid manifest.json")
    manifest_mapping = cast("Mapping[str, object]", manifest)

    manifest_input = manifest_mapping.get("input")
    if not isinstance(manifest_input, str) or manifest_input != _input_source_identity(options.input_path):
        raise ValueError("--parallel --resume input does not match the saved manifest")

    manifest_input_rows = manifest_mapping.get("input_rows")
    if not isinstance(manifest_input_rows, int) or manifest_input_rows != input_rows:
        raise ValueError("--parallel --resume input row count does not match the saved manifest")

    manifest_selected_rows = manifest_mapping.get("selected_rows")
    if not isinstance(manifest_selected_rows, int) or manifest_selected_rows < 0:
        raise ValueError("Saved parallel manifest has invalid selected_rows")

    manifest_shards = manifest_mapping.get("shards")
    if not isinstance(manifest_shards, list):
        raise ValueError("Saved parallel manifest has invalid shards")

    shards = [_load_parallel_manifest_shard(item) for item in manifest_shards]
    if sum(shard.limit for shard in shards) != manifest_selected_rows:
        raise ValueError("Saved parallel manifest shard limits do not match selected_rows")

    return ParallelManifest(
        input_rows=manifest_input_rows,
        selected_rows=manifest_selected_rows,
        shards=shards,
    )


def _load_parallel_manifest_shard(item: object) -> BatchShard:
    if not isinstance(item, dict):
        raise ValueError("Saved parallel manifest has invalid shard entries")
    shard = cast("Mapping[str, object]", item)

    index = shard.get("index")
    offset = shard.get("offset")
    limit = shard.get("limit")
    output = shard.get("output")
    summary_output = shard.get("summary_output")
    error_output = shard.get("error_output")
    telemetry_output = shard.get("telemetry_output")

    if not isinstance(index, int) or index < 0:
        raise ValueError("Saved parallel manifest has invalid shard index")
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("Saved parallel manifest has invalid shard offset")
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Saved parallel manifest has invalid shard limit")
    if not isinstance(output, str):
        raise ValueError("Saved parallel manifest has invalid shard output")
    if not isinstance(summary_output, str):
        raise ValueError("Saved parallel manifest has invalid shard summary_output")
    if error_output is not None and not isinstance(error_output, str):
        raise ValueError("Saved parallel manifest has invalid shard error_output")
    if telemetry_output is not None and not isinstance(telemetry_output, str):
        raise ValueError("Saved parallel manifest has invalid shard telemetry_output")

    return BatchShard(
        index=index,
        offset=offset,
        limit=limit,
        output_path=Path(output),
        error_output_path=Path(error_output) if error_output is not None else None,
        telemetry_output_path=Path(telemetry_output) if telemetry_output is not None else None,
        summary_output_path=Path(summary_output),
    )


def _input_source_identity(source: str | Path) -> str:
    source_text = str(source)
    if source_text.startswith("hf://"):
        return source_text
    return str(Path(source_text).expanduser().resolve())


def _merge_jsonl_shards(source_paths: list[Path], output_path: Path, work_dir: Path) -> None:
    ensure_parent(output_path)
    tmp_path = work_dir / f"{output_path.name}.tmp"
    with tmp_path.open("w", encoding="utf-8") as output_handle:
        for source_path in source_paths:
            if not source_path.exists():
                raise ValueError(f"Shard output missing: {source_path}")
            with source_path.open("r", encoding="utf-8") as input_handle:
                shutil.copyfileobj(input_handle, output_handle)
    tmp_path.replace(output_path)


def _merge_parallel_summaries(
    *,
    options: StructuredBatchOptions,
    started_at: str,
    completed_at: str,
    duration_ms: float,
    input_rows: int,
    selected_rows: int,
    work_dir: Path,
    shards: list[BatchShard],
    shard_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    schema_source = load_schema_source(options)
    payload: dict[str, Any] = {
        "model": options.model,
        "input": str(options.input_path),
        "output": str(options.output_path),
        "schema": str(options.schema_source) if options.schema_source is not None else None,
        "schema_model": options.schema_model,
        "instruction": str(options.instruction_source) if options.instruction_source else None,
        "agent_card": options.agent_card_source,
        "agent": _first_summary_value(shard_summaries, "agent"),
        "template": str(options.template_source) if options.template_source else "<default>",
        "shell_runtime": options.shell_runtime,
        "output_mode": "structured" if schema_source is not None else "text",
        "export_traces": None,
        "hf_dataset": None,
        "hf_dataset_path": None,
        "parallel": len(shards),
        "work_dir": str(work_dir),
        "started_at": started_at,
        "completed_at": completed_at,
        "input_rows": input_rows,
        "selected_rows": selected_rows,
        "processed_rows": sum(_summary_int(summary, "processed_rows") for summary in shard_summaries),
        "skipped_rows": sum(_summary_int(summary, "skipped_rows") for summary in shard_summaries),
        "failed_rows": sum(_summary_int(summary, "failed_rows") for summary in shard_summaries),
        "duration_ms": duration_ms,
        "timing_ms": _merge_timing_summaries(shard_summaries),
        "shards": [
            {
                "index": shard.index,
                "offset": shard.offset,
                "limit": shard.limit,
                "output": str(shard.output_path),
            }
            for shard in shards
        ],
    }
    return payload


def _empty_parallel_summary(
    options: StructuredBatchOptions,
    started_at: str,
    input_rows: int,
    work_dir: Path,
) -> dict[str, Any]:
    completed_at = utc_now_iso()
    return {
        "model": options.model,
        "input": str(options.input_path),
        "output": str(options.output_path),
        "schema": str(options.schema_source) if options.schema_source is not None else None,
        "schema_model": options.schema_model,
        "instruction": str(options.instruction_source) if options.instruction_source else None,
        "agent_card": options.agent_card_source,
        "agent": None,
        "template": str(options.template_source) if options.template_source else "<default>",
        "shell_runtime": options.shell_runtime,
        "output_mode": "structured"
        if options.schema_source is not None or options.schema_model is not None
        else "text",
        "export_traces": None,
        "hf_dataset": None,
        "hf_dataset_path": None,
        "parallel": options.parallel,
        "work_dir": str(work_dir),
        "started_at": started_at,
        "completed_at": completed_at,
        "input_rows": input_rows,
        "selected_rows": 0,
        "processed_rows": 0,
        "skipped_rows": 0,
        "failed_rows": 0,
        "duration_ms": 0,
        "timing_ms": {
            "duration": {"count": 0},
            "ttft": {"count": 0},
            "time_to_response": {"count": 0},
        },
        "shards": [],
    }


def _write_empty_parallel_outputs(options: StructuredBatchOptions) -> None:
    for path in (options.output_path, options.error_output_path, options.telemetry_output_path):
        if path is not None:
            ensure_parent(path)
            path.write_text("", encoding="utf-8")


def _write_parallel_summary(options: StructuredBatchOptions, payload: dict[str, Any]) -> None:
    if options.summary_output_path is not None:
        options.summary_output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _cleanup_parallel_work_dir(options: StructuredBatchOptions, work_dir: Path) -> None:
    if options.keep_temp:
        return
    shutil.rmtree(work_dir, ignore_errors=True)


def _first_summary_value(summaries: list[dict[str, Any]], key: str) -> Any:
    for summary in summaries:
        value = summary.get(key)
        if value is not None:
            return value
    return None


def _summary_int(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key)
    return value if isinstance(value, int) else 0


def _merge_timing_summaries(summaries: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    return {
        key: _merge_timing_key(summaries, key)
        for key in ("duration", "ttft", "time_to_response")
    }


def _merge_timing_key(summaries: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    parts: list[dict[str, Any]] = []
    for summary in summaries:
        timing = summary.get("timing_ms")
        if not isinstance(timing, dict):
            continue
        part = timing.get(key)
        if isinstance(part, dict) and isinstance(part.get("count"), int) and part["count"] > 0:
            parts.append(part)
    if not parts:
        return {"count": 0}
    count = sum(cast("int", part["count"]) for part in parts)
    weighted_mean = sum(cast("float", part.get("mean", 0.0)) * part["count"] for part in parts) / count
    weighted_median = (
        sum(cast("float", part.get("median", 0.0)) * part["count"] for part in parts) / count
    )
    return {
        "count": count,
        "min": min(cast("float", part.get("min", 0.0)) for part in parts),
        "mean": weighted_mean,
        "median": weighted_median,
        "max": max(cast("float", part.get("max", 0.0)) for part in parts),
    }


def _configure_trace_recorder(
    worker: AgentProtocol,
    options: StructuredBatchOptions,
    metadata: dict[str, Any],
) -> BatchTraceRecorder | None:
    trace_options = BatchTraceOptions(
        export_traces_path=options.export_traces_path,
        hf_dataset=options.hf_dataset,
        hf_dataset_path=options.hf_dataset_path,
    )
    if trace_options.export_traces_path is None:
        return None
    recorder = BatchTraceRecorder(
        trace_dir=trace_options.export_traces_path,
        agent=worker,
        run_metadata=metadata,
    )
    recorder.initialize()
    recorder.install_hook()
    return recorder


async def _configure_batch_worker(
    fast: FastAgent,
    options: StructuredBatchOptions,
    instruction: str | None,
) -> str:
    if options.agent_card_source is None:
        assert instruction is not None

        @fast.agent(name="batch_worker", instruction=instruction, model=options.model, default=True)
        async def batch_worker() -> None:
            pass

        return "batch_worker"

    from fast_agent.batch.agent_card import (
        force_loaded_card_history_off,
        load_batch_agent_card,
        override_selected_agent_model,
    )

    selection = load_batch_agent_card(
        fast,
        source=options.agent_card_source,
        requested_agent=options.agent_name,
    )
    force_loaded_card_history_off(fast, selection.loaded_names)
    if options.model is not None:
        override_selected_agent_model(fast, selection.target_name, options.model)
    return selection.target_name


async def _row_call(
    worker: Any,
    *,
    rendered: str,
    schema_source: LoadedSchemaSource | None,
    batch_context: BatchRequestContext,
) -> tuple[Any | None, Any]:
    request_params = RequestParams(use_history=False, batch_context=batch_context)
    if schema_source is None:
        response = await worker.generate(rendered, request_params)
        return response.last_text() or "", response
    if isinstance(schema_source, type) and issubclass(schema_source, BaseModel):
        return await worker.structured(rendered, schema_source, request_params)
    return await worker.structured_schema(rendered, schema_source, request_params)


def _json_result(parsed: Any) -> Any:
    if isinstance(parsed, BaseModel):
        return parsed.model_dump(mode="json")
    return parsed


def _max_errors_reached(failed_rows: int, max_errors: int | None) -> bool:
    return max_errors is not None and failed_rows >= max_errors


class _optional_jsonl_handle:
    def __init__(self, path: Path | None, mode: str) -> None:
        self._path = path
        self._mode = mode
        self._handle: TextIO | None = None

    def __enter__(self) -> TextIO | None:
        if self._path is None:
            return None
        self._handle = cast("TextIO", self._path.open(self._mode, encoding="utf-8"))
        return self._handle

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._handle is not None:
            self._handle.close()
