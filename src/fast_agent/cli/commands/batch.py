"""Batch processing commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from fast_agent.batch.structured import StructuredBatchOptions, run_structured_batch
from fast_agent.cli.command_support import ensure_context_object
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


def _run_async(coro):
    configure_uvloop()
    return asyncio.run(coro)


@app.command("structured")
def structured(
    ctx: typer.Context,
    input_path: Path = typer.Option(..., "--input", "-i", help="Input .jsonl or .csv file"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output JSONL file"),
    schema_path: Path | None = typer.Option(None, "--schema", help="JSON Schema file"),
    schema_model: str | None = typer.Option(
        None,
        "--schema-model",
        help="Pydantic BaseModel import path, for example myapp.schemas:Result",
    ),
    template_path: Path | None = typer.Option(
        None,
        "--template",
        help="Row prompt template file; defaults to dumping the full row JSON",
    ),
    instruction_path: Path | None = typer.Option(
        None,
        "--instruction",
        help="System instruction file; defaults to fast-agent's standard instruction",
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model override"),
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
    final_summary: bool = typer.Option(
        True,
        "--final-summary/--no-final-summary",
        help="Print final summary to stdout",
    ),
) -> None:
    """Run one selected input row -> one structured model request -> one output record."""
    for value, name in (
        (limit, "--limit"),
        (offset, "--offset"),
        (sample, "--sample"),
        (seed, "--seed"),
        (max_errors, "--max-errors"),
    ):
        _validate_non_negative(value, name)

    if resume and overwrite:
        raise typer.BadParameter("--resume and --overwrite cannot be used together")
    if schema_path is not None and schema_model is not None:
        raise typer.BadParameter("--schema and --schema-model cannot be used together")
    if schema_path is None and schema_model is None:
        raise typer.BadParameter("One of --schema or --schema-model is required")

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
        final_summary=final_summary,
        environment_dir=environment_dir,
    )

    try:
        summary = _run_async(run_structured_batch(options))
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if final_summary:
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
