"""Direct-mode structured batch runner."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from typing import TYPE_CHECKING, Any, TextIO, TypeAlias, cast

from jsonschema.exceptions import SchemaError
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
from fast_agent.cli.runtime.request_builders import resolve_default_instruction
from fast_agent.constants import FAST_AGENT_TIMING
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.structured_schema import validate_json_schema_definition
from fast_agent.mcp.helpers.content_helpers import get_text

if TYPE_CHECKING:
    from pathlib import Path


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


PydanticModel: TypeAlias = type[BaseModel]
SchemaSource: TypeAlias = dict[str, Any] | PydanticModel


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json_schema(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read JSON schema file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON schema file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON schema file {path} must contain a JSON object")
    try:
        return validate_json_schema_definition(payload)
    except SchemaError as exc:
        raise ValueError(f"Invalid JSON schema in {path}: {exc.message}") from exc


def load_pydantic_model(spec: str) -> PydanticModel:
    module_name, separator, class_path = spec.partition(":")
    if not module_name or separator != ":" or not class_path:
        raise ValueError("Expected --schema-model in the form module.path:ClassName")

    try:
        target: object = import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"Could not import schema model module {module_name}: {exc}") from exc

    try:
        for part in class_path.split("."):
            target = getattr(target, part)
    except AttributeError as exc:
        raise ValueError(f"Could not resolve schema model {spec}: missing {part}") from exc

    if not isinstance(target, type) or not issubclass(target, BaseModel):
        raise ValueError("--schema-model must point to a pydantic BaseModel subclass")

    return target


def load_schema_source(options: StructuredBatchOptions) -> SchemaSource:
    if options.schema_path is not None and options.schema_model is not None:
        raise ValueError("--schema and --schema-model cannot be used together")
    if options.schema_path is None and options.schema_model is None:
        raise ValueError("One of --schema or --schema-model is required")
    if options.schema_model is not None:
        return load_pydantic_model(options.schema_model)
    assert options.schema_path is not None
    return load_json_schema(options.schema_path)


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


def _extract_timing(response: Any) -> dict[str, Any] | None:
    channels = response.channels
    if not isinstance(channels, Mapping):
        return None
    timing_blocks = channels.get(FAST_AGENT_TIMING)
    if not timing_blocks:
        return None
    timing_text = get_text(timing_blocks[0])
    if not timing_text:
        return None
    try:
        timing = json.loads(timing_text)
    except json.JSONDecodeError:
        return None
    return timing if isinstance(timing, dict) else None


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
            "usage": {},
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
    """Run a direct-mode structured batch job and return the summary payload."""
    _prepare_output_files(options)

    schema_source = load_schema_source(options)
    template = (
        load_text_file(options.template_path, "template")
        if options.template_path is not None
        else DEFAULT_ROW_TEMPLATE
    )
    instruction = (
        load_text_file(options.instruction_path, "instruction")
        if options.instruction_path is not None
        else resolve_default_instruction(options.model, "interactive")
    )

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
            "template": str(options.template_path) if options.template_path else "<default>",
        },
    )

    from fast_agent import FastAgent

    fast = FastAgent(
        name="structured batch",
        parse_cli_args=False,
        ignore_unknown_args=True,
        quiet=True,
        environment_dir=options.environment_dir,
    )
    if options.model:
        fast.args.model = options.model

    @fast.agent(name="batch_worker", instruction=instruction, model=options.model, default=True)
    async def batch_worker() -> None:
        pass

    output_mode = "a" if options.resume else "w"
    if options.overwrite:
        output_mode = "w"

    async with fast.run() as agent_app:
        worker = agent_app._agent("batch_worker")
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
                            if _max_errors_reached(summary.failed_rows, options.max_errors):
                                break
                            continue

                        assert rendered is not None
                        assert candidate.row is not None
                        try:
                            parsed, response = await _structured_row_call(
                                worker,
                                rendered=rendered,
                                schema_source=schema_source,
                            )
                            timing = _extract_timing(response)
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
                                )
                                summary.processed_rows += 1
                                summary.failed_rows += 1
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
                            )
                            summary.processed_rows += 1
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
                            if _max_errors_reached(summary.failed_rows, options.max_errors):
                                break

    completed_at = utc_now_iso()
    payload = summary.to_dict(completed_at)
    if options.summary_output_path is not None:
        options.summary_output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


async def _structured_row_call(
    worker: Any,
    *,
    rendered: str,
    schema_source: SchemaSource,
) -> tuple[Any | None, Any]:
    request_params = RequestParams(use_history=False)
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
