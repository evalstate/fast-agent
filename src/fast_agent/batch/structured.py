"""Batch runner for row-oriented agent/model jobs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TextIO, TypeAlias, cast

from pydantic import BaseModel

from fast_agent.batch.input import RowCandidate, RowError, iter_input_rows, select_rows
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
from fast_agent.llm.request_params import BatchRequestContext, RequestParams
from fast_agent.llm.structured_schema import (
    StructuredSchemaSource,
    load_json_schema_file,
    load_pydantic_model,
)
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.session.trace_export_errors import SessionExportUploadError

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.core.fastagent import FastAgent
    from fast_agent.interfaces import AgentProtocol


@dataclass(frozen=True)
class StructuredBatchOptions:
    input_path: Path
    output_path: Path
    schema_path: Path | None = None
    schema_model: str | None = None
    template_path: Path | None = None
    instruction_path: Path | None = None
    model: str | None = None
    include_input: bool = False
    limit: int | None = None
    offset: int | None = None
    sample: int | None = None
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


SchemaSource: TypeAlias = StructuredSchemaSource


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json_schema(path: Path) -> dict[str, Any]:
    return load_json_schema_file(path)


def load_schema_source(options: StructuredBatchOptions) -> SchemaSource | None:
    if options.agent_card_source is not None and options.instruction_path is not None:
        raise ValueError("--agent-card and --instruction cannot be used together")
    if options.agent_name is not None and options.agent_card_source is None:
        raise ValueError("--agent requires --agent-card")
    if options.schema_path is not None and options.schema_model is not None:
        raise ValueError("--schema and --schema-model cannot be used together")
    if options.hf_dataset_path is not None and options.hf_dataset is None:
        raise ValueError("--hf-dataset-path requires --hf-dataset")
    if options.hf_dataset is not None and options.export_traces_path is None:
        raise ValueError("--hf-dataset requires --export-traces")
    if options.schema_model is not None:
        return load_pydantic_model(options.schema_model)
    if options.schema_path is not None:
        return load_json_schema(options.schema_path)
    return None


def load_text_file(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read {label} file {path}: {exc}") from exc


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
        load_text_file(options.template_path, "template")
        if options.template_path is not None
        else DEFAULT_ROW_TEMPLATE
    )
    if options.agent_card_source is None:
        instruction: str | None = (
            load_text_file(options.instruction_path, "instruction")
            if options.instruction_path is not None
            else resolve_default_instruction(options.model, "interactive")
        )
    else:
        instruction = None

    all_candidates = list(iter_input_rows(options.input_path))
    selected = select_rows(
        all_candidates,
        offset=options.offset,
        sample=options.sample,
        seed=options.seed,
        limit=options.limit,
    )
    completed_ids = load_completed_ids(options.output_path) if options.resume else set()

    started_at = utc_now_iso()
    summary = BatchSummary(
        input_rows=len(all_candidates),
        selected_rows=len(selected),
        started_at=started_at,
        metadata={
            "model": options.model,
            "input": str(options.input_path),
            "output": str(options.output_path),
            "schema": str(options.schema_path) if options.schema_path is not None else None,
            "schema_model": options.schema_model,
            "instruction": str(options.instruction_path) if options.instruction_path else None,
            "agent_card": options.agent_card_source,
            "agent": None,
            "template": str(options.template_path) if options.template_path else "<default>",
            "shell_runtime": options.shell_runtime,
            "output_mode": "structured" if schema_source is not None else "text",
            "export_traces": str(options.export_traces_path) if options.export_traces_path else None,
            "hf_dataset": options.hf_dataset,
            "hf_dataset_path": options.hf_dataset_path,
        },
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
    return payload


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
    schema_source: SchemaSource | None,
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
