"""Batch processing commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer

from fast_agent.batch.monitoring import BatchTrackioOptions
from fast_agent.batch.structured import (
    StructuredBatchOptions,
    run_parallel_structured_batch,
    run_structured_batch,
)
from fast_agent.cli.command_support import ensure_context_object
from fast_agent.cli.shared_options import CommonAgentOptions
from fast_agent.utils.async_utils import run_coroutine
from fast_agent.utils.text import strip_to_none

app = typer.Typer(help="Run batch processing jobs.", add_completion=False)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Run batch processing jobs."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _validate_non_negative(value: int | None, name: str) -> None:
    if value is not None and value < 0:
        raise typer.BadParameter(f"{name} must be non-negative")


def _validate_positive(value: int | None, name: str) -> None:
    if value is not None and value <= 0:
        raise typer.BadParameter(f"{name} must be greater than zero")


def _fail_validation(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(2)


def _fail_runtime(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(1)


def _parse_var_entries(entries: list[str] | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for entry in entries or []:
        name, separator, value = entry.partition("=")
        if not separator or not name:
            raise typer.BadParameter("--var entries must use name=value")
        values[name] = value
    return values


def _parse_var_file_entries(entries: list[str] | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for entry in entries or []:
        name, separator, path_text = entry.partition("=")
        if not separator or not name:
            raise typer.BadParameter("--var-file entries must use name=path")
        values[name] = Path(path_text).expanduser().read_text(encoding="utf-8")
    return values


def _load_vars_json(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter("--vars-json must contain a JSON object")
    values: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise typer.BadParameter("--vars-json keys and values must be strings")
        values[key] = value
    return values


def _load_trackio_config_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter("--trackio-config-json must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("--trackio-config-json must contain a JSON object")
    return dict(payload)


def _merge_batch_variables(
    *,
    var_entries: list[str] | None,
    var_file_entries: list[str] | None,
    vars_json_path: Path | None,
) -> dict[str, str] | None:
    values = _load_vars_json(vars_json_path)
    values.update(_parse_var_file_entries(var_file_entries))
    values.update(_parse_var_entries(var_entries))
    return values or None


def _validate_local_input_exists(input_path: str) -> None:
    parsed = urlparse(input_path)
    if parsed.scheme:
        return
    path = Path(input_path).expanduser()
    if not path.exists():
        _fail_runtime(f"Input file not found: {input_path}")
    if not path.is_file():
        _fail_runtime(f"Input path is not a file: {input_path}")


def _run_async(coro):
    return run_coroutine(coro)


def _validate_batch_run_options(
    *,
    input_path: str,
    prompt: str | None,
    template_source: str | None,
    resume: bool,
    overwrite: bool,
    instruction_source: str | None,
    agent_card_source: str | None,
    agent_name: str | None,
    schema_source: str | None,
    schema_model: str | None,
    hf_dataset: str | None,
    hf_dataset_path: str | None,
    export_traces_path: Path | None,
    sql: str | None,
    limit: int | None,
    offset: int | None,
    sample: int | None,
    seed: int | None,
    max_errors: int | None,
    parallel: int | None,
    progress_every: int | None,
    trackio_project: str | None,
    trackio_name: str | None,
    trackio_group: str | None,
    trackio_space_id: str | None,
    trackio_server_url: str | None,
    trackio_every: int | None,
    trackio_config_json: Path | None,
    no_trackio: bool,
) -> None:
    hf_dataset = strip_to_none(hf_dataset)
    hf_dataset_path = strip_to_none(hf_dataset_path)
    _validate_batch_numeric_options(
        limit=limit,
        offset=offset,
        sample=sample,
        seed=seed,
        max_errors=max_errors,
        parallel=parallel,
        progress_every=progress_every,
        trackio_every=trackio_every,
    )
    _validate_batch_conflicting_options(
        prompt=prompt,
        template_source=template_source,
        resume=resume,
        overwrite=overwrite,
        instruction_source=instruction_source,
        agent_card_source=agent_card_source,
        agent_name=agent_name,
        schema_source=schema_source,
        schema_model=schema_model,
        hf_dataset=hf_dataset,
        hf_dataset_path=hf_dataset_path,
        export_traces_path=export_traces_path,
        sql=sql,
        limit=limit,
        offset=offset,
        sample=sample,
        parallel=parallel,
        trackio_project=trackio_project,
        trackio_name=trackio_name,
        trackio_group=trackio_group,
        trackio_space_id=trackio_space_id,
        trackio_server_url=trackio_server_url,
        trackio_every=trackio_every,
        trackio_config_json=trackio_config_json,
        no_trackio=no_trackio,
    )
    _validate_local_input_exists(input_path)


def _validate_batch_numeric_options(
    *,
    limit: int | None,
    offset: int | None,
    sample: int | None,
    seed: int | None,
    max_errors: int | None,
    parallel: int | None,
    progress_every: int | None,
    trackio_every: int | None,
) -> None:
    for value, name in (
        (limit, "--limit"),
        (offset, "--offset"),
        (sample, "--sample"),
        (seed, "--seed"),
        (max_errors, "--max-errors"),
    ):
        _validate_non_negative(value, name)
    for value, name in (
        (parallel, "--parallel"),
        (progress_every, "--progress-every"),
        (trackio_every, "--trackio-every"),
    ):
        _validate_positive(value, name)


def _validate_batch_conflicting_options(
    *,
    prompt: str | None,
    template_source: str | None,
    resume: bool,
    overwrite: bool,
    instruction_source: str | None,
    agent_card_source: str | None,
    agent_name: str | None,
    schema_source: str | None,
    schema_model: str | None,
    hf_dataset: str | None,
    hf_dataset_path: str | None,
    export_traces_path: Path | None,
    sql: str | None,
    limit: int | None,
    offset: int | None,
    sample: int | None,
    parallel: int | None,
    trackio_project: str | None,
    trackio_name: str | None,
    trackio_group: str | None,
    trackio_space_id: str | None,
    trackio_server_url: str | None,
    trackio_every: int | None,
    trackio_config_json: Path | None,
    no_trackio: bool,
) -> None:
    if prompt is not None and template_source is not None:
        _fail_validation("--prompt and --template cannot be used together")
    if resume and overwrite:
        _fail_validation("--resume and --overwrite cannot be used together")
    if instruction_source is not None and agent_card_source is not None:
        _fail_validation("--agent-card and --instruction cannot be used together")
    if agent_name is not None and agent_card_source is None:
        _fail_validation("--agent requires --agent-card")
    if schema_source is not None and schema_model is not None:
        _fail_validation("--json-schema and --schema-model cannot be used together")
    if hf_dataset_path is not None and hf_dataset is None:
        _fail_validation("--hf-dataset-path requires --hf-dataset")
    if hf_dataset is not None and export_traces_path is None:
        _fail_validation("--hf-dataset requires --export-traces")
    if sql is not None and (limit is not None or offset is not None or sample is not None):
        _fail_validation("--sql cannot be used with --limit, --offset, or --sample")
    if sql is not None and parallel is not None and parallel > 1:
        _fail_validation("--sql cannot be used with --parallel")
    trackio_project = strip_to_none(trackio_project)
    trackio_detail_values = (
        trackio_name,
        trackio_group,
        trackio_space_id,
        trackio_server_url,
        trackio_every,
        trackio_config_json,
    )
    if no_trackio and (
        trackio_project is not None or any(value is not None for value in trackio_detail_values)
    ):
        _fail_validation("--no-trackio cannot be used with Trackio options")
    if (
        trackio_project is None
        and not no_trackio
        and any(value is not None for value in trackio_detail_values)
    ):
        _fail_validation("Trackio options require --project or --trackio-project")


def _home_from_context(ctx: typer.Context) -> Path | None:
    home = ensure_context_object(ctx).get("home")
    return home if isinstance(home, Path) else None


def _batch_progress_enabled(
    *,
    progress: bool,
    parallel: int | None,
    progress_every: int | None,
) -> bool:
    return progress and ((parallel is not None and parallel > 1) or progress_every is not None)


def _build_structured_batch_options(
    *,
    ctx: typer.Context,
    input_path: str,
    prompt: str | None,
    output_path: Path,
    schema_source: str | None,
    schema_model: str | None,
    template_source: str | None,
    instruction_source: str | None,
    agent_card_source: str | None,
    agent_name: str | None,
    model: str | None,
    include_input: bool,
    limit: int | None,
    offset: int | None,
    sample: int | None,
    sql: str | None,
    seed: int | None,
    resume: bool,
    overwrite: bool,
    id_field: str | None,
    max_errors: int | None,
    error_output_path: Path | None,
    telemetry_output_path: Path | None,
    summary_output_path: Path | None,
    export_traces_path: Path | None,
    hf_dataset: str | None,
    hf_dataset_path: str | None,
    parallel: int | None,
    work_dir: Path | None,
    keep_temp: bool,
    progress_every: int | None,
    progress: bool,
    final_summary: bool,
    shell_runtime: bool,
    variables: dict[str, str] | None,
    trackio_project: str | None,
    trackio_name: str | None,
    trackio_group: str | None,
    trackio_space_id: str | None,
    trackio_server_url: str | None,
    trackio_every: int | None,
    trackio_config: dict[str, Any] | None,
    no_trackio: bool,
) -> StructuredBatchOptions:
    hf_dataset = strip_to_none(hf_dataset)
    hf_dataset_path = strip_to_none(hf_dataset_path)
    trackio = _build_trackio_options(
        project=trackio_project,
        name=trackio_name,
        group=trackio_group,
        space_id=trackio_space_id,
        server_url=trackio_server_url,
        trackio_every=trackio_every,
        progress_every=progress_every,
        config=trackio_config,
        disabled=no_trackio,
    )
    return StructuredBatchOptions(
        input_path=input_path,
        output_path=output_path,
        prompt_template=prompt,
        schema_source=schema_source,
        schema_model=schema_model,
        template_source=template_source,
        instruction_source=instruction_source,
        model=model,
        include_input=include_input,
        limit=limit,
        offset=offset,
        sample=sample,
        sql=sql,
        seed=seed,
        resume=resume,
        overwrite=overwrite,
        id_field=id_field,
        max_errors=max_errors,
        error_output_path=error_output_path,
        telemetry_output_path=telemetry_output_path,
        summary_output_path=summary_output_path,
        export_traces_path=export_traces_path,
        hf_dataset=hf_dataset,
        hf_dataset_path=hf_dataset_path,
        parallel=parallel,
        work_dir=work_dir,
        keep_temp=keep_temp,
        progress_every=progress_every,
        progress=_batch_progress_enabled(
            progress=progress,
            parallel=parallel,
            progress_every=progress_every,
        ),
        final_summary=final_summary,
        home=_home_from_context(ctx),
        shell_runtime=shell_runtime,
        agent_card_source=agent_card_source,
        agent_name=agent_name,
        variables=variables,
        trackio=trackio,
    )


def _build_trackio_options(
    *,
    project: str | None,
    name: str | None,
    group: str | None,
    space_id: str | None,
    server_url: str | None,
    trackio_every: int | None,
    progress_every: int | None,
    config: dict[str, Any] | None,
    disabled: bool,
) -> BatchTrackioOptions | None:
    project = strip_to_none(project)
    if disabled or project is None:
        return None
    return BatchTrackioOptions(
        project=project,
        name=strip_to_none(name),
        group=strip_to_none(group),
        space_id=strip_to_none(space_id),
        server_url=strip_to_none(server_url),
        log_every=trackio_every or progress_every or 10,
        config=config,
    )


def _run_structured_batch_options(options: StructuredBatchOptions) -> dict:
    try:
        if options.parallel is not None and options.parallel > 1:
            return _run_async(run_parallel_structured_batch(options))
        return _run_async(run_structured_batch(options))
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except FileNotFoundError as exc:
        typer.echo(
            f"Error: File not found: {exc.filename or options.input_path}",
            err=True,
        )
        raise typer.Exit(1) from exc
    except OSError as exc:
        typer.echo(f"Error: File error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("run")
def run(
    ctx: typer.Context,
    input_path: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input .jsonl/.csv/.parquet path or hf:// URI to a Hugging Face dataset",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Inline row prompt template; mutually exclusive with --template",
    ),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output JSONL file"),
    schema_source: str | None = typer.Option(
        None,
        "--json-schema",
        metavar="<path-or-uri>",
        help="Optional JSON Schema file or URI for structured results",
    ),
    schema_model: str | None = typer.Option(
        None,
        "--schema-model",
        help="Optional Pydantic BaseModel import path for structured results",
    ),
    template_source: str | None = typer.Option(
        None,
        "--template",
        metavar="<path-or-uri>",
        help="Row prompt template file or URI; defaults to dumping the full row JSON",
    ),
    instruction_source: str | None = typer.Option(
        None,
        "--instruction",
        metavar="<path-or-uri>",
        help=(
            "System instruction file or URI for direct mode; defaults to fast-agent's "
            "standard instruction. Mutually exclusive with --agent-card"
        ),
    ),
    agent_card_source: str | None = typer.Option(
        None,
        "--agent-card",
        metavar="<path-or-uri>",
        help=(
            "AgentCard file, directory, or URI defining the batch worker. "
            "Mutually exclusive with --instruction"
        ),
    ),
    agent_name: str | None = typer.Option(
        None,
        "--agent",
        help="Agent name to run when --agent-card loads multiple runnable agents",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model override for direct mode or the selected AgentCard worker",
    ),
    include_input: bool = typer.Option(
        False,
        "--include-input/--no-include-input",
        help="Include the source row in each output envelope",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Maximum selected rows to process"),
    offset: int | None = typer.Option(None, "--offset", help="Rows to skip before sampling"),
    sample: int | None = typer.Option(None, "--sample", help="Deterministic sample size"),
    sql: str | None = typer.Option(
        None,
        "--sql",
        help="DuckDB SELECT query over parquet input view named input",
    ),
    seed: int | None = typer.Option(None, "--seed", help="Deterministic sampling seed"),
    resume: bool = typer.Option(False, "--resume", help="Append missing/retried rows"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace existing output"),
    id_field: str | None = typer.Option(None, "--id-field", help="Input field used as row ID"),
    max_errors: int | None = typer.Option(
        None,
        "--max-errors",
        help="Stop after this many row-level failures",
    ),
    error_output_path: Path | None = typer.Option(
        None,
        "--error-output",
        help="Additional JSONL file containing failed envelopes",
    ),
    telemetry_output_path: Path | None = typer.Option(
        None,
        "--telemetry-output",
        help="JSONL file containing per-attempt normalized telemetry",
    ),
    summary_output_path: Path | None = typer.Option(
        None,
        "--summary-output",
        help="Write final summary JSON to this path",
    ),
    export_traces_path: Path | None = typer.Option(
        None,
        "--export-traces",
        help="Directory for per-row Codex trace JSONL files and manifest.jsonl",
    ),
    hf_dataset: str | None = typer.Option(
        None,
        "--hf-dataset",
        help="Upload exported traces to this Hugging Face dataset repository",
    ),
    hf_dataset_path: str | None = typer.Option(
        None,
        "--hf-dataset-path",
        help="Path or prefix inside the Hugging Face dataset for exported traces",
    ),
    parallel: int | None = typer.Option(
        None,
        "--parallel",
        help="Run this many local workers and merge their chunk outputs",
    ),
    work_dir: Path | None = typer.Option(
        None,
        "--work-dir",
        help="Directory for parallel chunk outputs and resume manifests",
    ),
    keep_temp: bool = typer.Option(
        False,
        "--keep-temp/--no-keep-temp",
        help="Keep parallel chunk outputs after a successful merge",
    ),
    progress_every: int | None = typer.Option(
        None,
        "--progress-every",
        help="Print progress every N processed rows per worker",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Print batch progress messages to stderr",
    ),
    var_entries: list[str] | None = typer.Option(
        None,
        "--var",
        help="AgentCard template variable as name=value. May be repeated.",
    ),
    var_file_entries: list[str] | None = typer.Option(
        None,
        "--var-file",
        help="AgentCard template variable loaded from a file as name=path. May be repeated.",
    ),
    vars_json_path: Path | None = typer.Option(
        None,
        "--vars-json",
        help="JSON object containing AgentCard template variables.",
    ),
    trackio_project: str | None = typer.Option(
        None,
        "--project",
        "--trackio-project",
        help="Enable Trackio monitoring and set the Trackio project",
    ),
    trackio_name: str | None = typer.Option(
        None,
        "--run-name",
        "--trackio-name",
        help="Trackio run name",
    ),
    trackio_group: str | None = typer.Option(
        None,
        "--group",
        "--trackio-group",
        help="Trackio group for repeated runs or data-build phases",
    ),
    trackio_space_id: str | None = typer.Option(
        None,
        "--trackio-space-id",
        help="Optional Trackio/Hugging Face Space id",
    ),
    trackio_server_url: str | None = typer.Option(
        None,
        "--trackio-server-url",
        help="Optional Trackio server URL",
    ),
    trackio_every: int | None = typer.Option(
        None,
        "--trackio-every",
        help="Log Trackio aggregate progress every N processed rows",
    ),
    trackio_config_json: Path | None = typer.Option(
        None,
        "--trackio-config-json",
        help="JSON object merged into Trackio init config",
    ),
    no_trackio: bool = typer.Option(
        False,
        "--no-trackio",
        help="Disable Trackio monitoring",
    ),
    final_summary: bool = typer.Option(
        True,
        "--final-summary/--no-final-summary",
        help="Print final summary to stdout",
    ),
    shell_runtime: bool = CommonAgentOptions.shell(),
) -> None:
    """Run one selected input row -> one agent/model request -> one output record."""
    _validate_batch_run_options(
        input_path=input_path,
        template_source=template_source,
        prompt=prompt,
        instruction_source=instruction_source,
        agent_card_source=agent_card_source,
        agent_name=agent_name,
        schema_source=schema_source,
        schema_model=schema_model,
        hf_dataset=hf_dataset,
        hf_dataset_path=hf_dataset_path,
        export_traces_path=export_traces_path,
        sql=sql,
        limit=limit,
        offset=offset,
        sample=sample,
        seed=seed,
        resume=resume,
        overwrite=overwrite,
        max_errors=max_errors,
        parallel=parallel,
        progress_every=progress_every,
        trackio_project=trackio_project,
        trackio_name=trackio_name,
        trackio_group=trackio_group,
        trackio_space_id=trackio_space_id,
        trackio_server_url=trackio_server_url,
        trackio_every=trackio_every,
        trackio_config_json=trackio_config_json,
        no_trackio=no_trackio,
    )
    trackio_config = _load_trackio_config_json(trackio_config_json)
    options = _build_structured_batch_options(
        ctx=ctx,
        input_path=input_path,
        prompt=prompt,
        output_path=output_path,
        schema_source=schema_source,
        schema_model=schema_model,
        template_source=template_source,
        instruction_source=instruction_source,
        agent_card_source=agent_card_source,
        agent_name=agent_name,
        model=model,
        include_input=include_input,
        limit=limit,
        offset=offset,
        sample=sample,
        sql=sql,
        seed=seed,
        resume=resume,
        overwrite=overwrite,
        id_field=id_field,
        max_errors=max_errors,
        error_output_path=error_output_path,
        telemetry_output_path=telemetry_output_path,
        summary_output_path=summary_output_path,
        export_traces_path=export_traces_path,
        hf_dataset=hf_dataset,
        hf_dataset_path=hf_dataset_path,
        parallel=parallel,
        work_dir=work_dir,
        keep_temp=keep_temp,
        progress_every=progress_every,
        progress=progress,
        final_summary=final_summary,
        shell_runtime=shell_runtime,
        variables=_merge_batch_variables(
            var_entries=var_entries,
            var_file_entries=var_file_entries,
            vars_json_path=vars_json_path,
        ),
        trackio_project=trackio_project,
        trackio_name=trackio_name,
        trackio_group=trackio_group,
        trackio_space_id=trackio_space_id,
        trackio_server_url=trackio_server_url,
        trackio_every=trackio_every,
        trackio_config=trackio_config,
        no_trackio=no_trackio,
    )
    summary = _run_structured_batch_options(options)

    if final_summary:
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
