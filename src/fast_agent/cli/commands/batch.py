"""Batch processing commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from fast_agent.batch.structured import StructuredBatchOptions, run_structured_batch
from fast_agent.cli.command_support import ensure_context_object
from fast_agent.cli.shared_options import CommonAgentOptions
from fast_agent.utils.async_utils import configure_uvloop

app = typer.Typer(help="Run batch processing jobs.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Run batch processing jobs."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _validate_non_negative(value: int | None, name: str) -> None:
    if value is not None and value < 0:
        raise typer.BadParameter(f"{name} must be non-negative")


def _fail_validation(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(2)


def _run_async(coro):
    configure_uvloop()
    return asyncio.run(coro)


@app.command("run")
def run(
    ctx: typer.Context,
    input_path: Path = typer.Option(..., "--input", "-i", help="Input .jsonl or .csv file"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output JSONL file"),
    schema_path: Path | None = typer.Option(
        None,
        "--schema",
        help="Optional JSON Schema file for structured results",
    ),
    schema_model: str | None = typer.Option(
        None,
        "--schema-model",
        help="Optional Pydantic BaseModel import path for structured results",
    ),
    template_path: Path | None = typer.Option(
        None,
        "--template",
        help="Row prompt template file; defaults to dumping the full row JSON",
    ),
    instruction_path: Path | None = typer.Option(
        None,
        "--instruction",
        help=(
            "System instruction file for direct mode; defaults to fast-agent's "
            "standard instruction. Mutually exclusive with --agent-card"
        ),
    ),
    agent_card_source: str | None = typer.Option(
        None,
        "--agent-card",
        help=(
            "AgentCard file, directory, or URL defining the batch worker. "
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
    final_summary: bool = typer.Option(
        True,
        "--final-summary/--no-final-summary",
        help="Print final summary to stdout",
    ),
    shell_runtime: bool = CommonAgentOptions.shell(),
) -> None:
    """Run one selected input row -> one agent/model request -> one output record."""
    for value, name in (
        (limit, "--limit"),
        (offset, "--offset"),
        (sample, "--sample"),
        (seed, "--seed"),
        (max_errors, "--max-errors"),
    ):
        _validate_non_negative(value, name)

    if resume and overwrite:
        _fail_validation("--resume and --overwrite cannot be used together")
    if instruction_path is not None and agent_card_source is not None:
        _fail_validation("--agent-card and --instruction cannot be used together")
    if agent_name is not None and agent_card_source is None:
        _fail_validation("--agent requires --agent-card")
    if schema_path is not None and schema_model is not None:
        _fail_validation("--schema and --schema-model cannot be used together")
    if hf_dataset_path is not None and hf_dataset is None:
        _fail_validation("--hf-dataset-path requires --hf-dataset")
    if hf_dataset is not None and export_traces_path is None:
        _fail_validation("--hf-dataset requires --export-traces")

    context = ensure_context_object(ctx)
    env_dir = context.get("env_dir")
    environment_dir = env_dir if isinstance(env_dir, Path) else None

    options = StructuredBatchOptions(
        input_path=input_path,
        output_path=output_path,
        schema_path=schema_path,
        schema_model=schema_model,
        template_path=template_path,
        instruction_path=instruction_path,
        model=model,
        include_input=include_input,
        limit=limit,
        offset=offset,
        sample=sample,
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
        final_summary=final_summary,
        environment_dir=environment_dir,
        shell_runtime=shell_runtime,
        agent_card_source=agent_card_source,
        agent_name=agent_name,
    )

    try:
        summary = _run_async(run_structured_batch(options))
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if final_summary:
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
