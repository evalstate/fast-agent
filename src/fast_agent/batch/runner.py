"""Public batch runner API."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fast_agent.batch.structured import StructuredBatchOptions, run_parallel_structured_batch

BatchBackend = Literal["harness", "process"]


@dataclass(frozen=True)
class BatchRunResult:
    """Inspectable result from a batch run."""

    rows: list[dict[str, Any]]
    output_path: Path
    summary: dict[str, Any]
    telemetry_path: Path | None
    error_output_path: Path | None
    summary_path: Path | None

    @property
    def error_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.rows if row.get("ok") is False]

    @property
    def artifact_paths(self) -> dict[str, Path]:
        paths = {"output": self.output_path}
        if self.telemetry_path is not None:
            paths["telemetry"] = self.telemetry_path
        if self.error_output_path is not None:
            paths["errors"] = self.error_output_path
        if self.summary_path is not None:
            paths["summary"] = self.summary_path
        return paths


class BatchRunner:
    """Small public wrapper around fast-agent's structured batch engine."""

    def __init__(self, env_dir: str | Path | None = None, *, backend: BatchBackend = "harness"):
        self.env_dir = Path(env_dir) if env_dir is not None else None
        self.backend = backend

    async def run(
        self,
        *,
        input: str | Path,
        output_path: str | Path,
        agent_card: str | Path | None = None,
        agent: str | None = None,
        template: str | None = None,
        template_source: str | Path | None = None,
        json_schema: str | Path | None = None,
        schema_model: str | None = None,
        model: str | None = None,
        parallel: int | None = None,
        include_input: bool = False,
        variables: dict[str, str] | None = None,
        summary_path: str | Path | None = None,
        telemetry_path: str | Path | None = None,
        error_output_path: str | Path | None = None,
        overwrite: bool = False,
        resume: bool = False,
        id_field: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        sample: int | None = None,
        seed: int | None = None,
        max_errors: int | None = None,
        progress: bool = False,
    ) -> BatchRunResult:
        output = Path(output_path)
        summary_output = Path(summary_path) if summary_path is not None else None
        telemetry_output = Path(telemetry_path) if telemetry_path is not None else None
        error_output = Path(error_output_path) if error_output_path is not None else None
        if self.backend == "process":
            return self._run_process(
                input=input,
                output=output,
                agent_card=agent_card,
                agent=agent,
                template=template,
                template_source=template_source,
                json_schema=json_schema,
                schema_model=schema_model,
                model=model,
                parallel=parallel,
                include_input=include_input,
                variables=variables,
                summary_output=summary_output,
                telemetry_output=telemetry_output,
                error_output=error_output,
                overwrite=overwrite,
                resume=resume,
                id_field=id_field,
                limit=limit,
                offset=offset,
                sample=sample,
                seed=seed,
                max_errors=max_errors,
                progress=progress,
            )
        options = StructuredBatchOptions(
            input_path=input,
            output_path=output,
            prompt_template=template,
            schema_source=json_schema,
            schema_model=schema_model,
            template_source=template_source,
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
            error_output_path=error_output,
            telemetry_output_path=telemetry_output,
            summary_output_path=summary_output,
            environment_dir=self.env_dir,
            agent_card_source=str(agent_card) if agent_card is not None else None,
            agent_name=agent,
            parallel=parallel,
            progress=progress,
            variables=variables,
        )
        summary = await run_parallel_structured_batch(options)
        return BatchRunResult(
            rows=_read_jsonl(output),
            output_path=output,
            summary=summary,
            telemetry_path=telemetry_output,
            error_output_path=error_output,
            summary_path=summary_output,
        )

    def _run_process(
        self,
        *,
        input: str | Path,
        output: Path,
        agent_card: str | Path | None,
        agent: str | None,
        template: str | None,
        template_source: str | Path | None,
        json_schema: str | Path | None,
        schema_model: str | None,
        model: str | None,
        parallel: int | None,
        include_input: bool,
        variables: dict[str, str] | None,
        summary_output: Path | None,
        telemetry_output: Path | None,
        error_output: Path | None,
        overwrite: bool,
        resume: bool,
        id_field: str | None,
        limit: int | None,
        offset: int | None,
        sample: int | None,
        seed: int | None,
        max_errors: int | None,
        progress: bool,
    ) -> BatchRunResult:
        summary_path = summary_output or output.with_suffix(".summary.json")
        command = [sys.executable, "-m", "fast_agent.cli.__main__", "--no-update-check"]
        if self.env_dir is not None:
            command.extend(["--env", str(self.env_dir)])
        command.extend(["batch", "run", "--input", str(input), "--output", str(output)])
        _extend_optional(command, "--agent-card", agent_card)
        _extend_optional(command, "--agent", agent)
        _extend_optional(command, "--prompt", template)
        _extend_optional(command, "--template", template_source)
        _extend_optional(command, "--json-schema", json_schema)
        _extend_optional(command, "--schema-model", schema_model)
        _extend_optional(command, "--model", model)
        _extend_optional(command, "--parallel", parallel)
        _extend_optional(command, "--id-field", id_field)
        _extend_optional(command, "--limit", limit)
        _extend_optional(command, "--offset", offset)
        _extend_optional(command, "--sample", sample)
        _extend_optional(command, "--seed", seed)
        _extend_optional(command, "--max-errors", max_errors)
        _extend_optional(command, "--summary-output", summary_path)
        _extend_optional(command, "--telemetry-output", telemetry_output)
        _extend_optional(command, "--error-output", error_output)
        if include_input:
            command.append("--include-input")
        if overwrite:
            command.append("--overwrite")
        if resume:
            command.append("--resume")
        if not progress:
            command.append("--no-progress")
        command.append("--no-final-summary")
        vars_path = None
        if variables:
            output.parent.mkdir(parents=True, exist_ok=True)
            vars_path = output.parent / f".{output.name}.vars.json"
            vars_path.write_text(
                json.dumps(variables, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            command.extend(["--vars-json", str(vars_path)])
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"fast-agent batch process failed with exit {completed.returncode}\n"
                f"{completed.stderr[-4000:]}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return BatchRunResult(
            rows=_read_jsonl(output),
            output_path=output,
            summary=summary,
            telemetry_path=telemetry_output,
            error_output_path=error_output,
            summary_path=summary_path,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _extend_optional(command: list[str], option: str, value: str | int | Path | None) -> None:
    if value is not None:
        command.extend([option, str(value)])
