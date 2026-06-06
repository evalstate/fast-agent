"""GEPA-compatible adapters built on fast-agent primitives."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from fast_agent.batch import BatchRunner, BatchRunResult
from fast_agent.eval import ArtifactRun, CandidateRun
from fast_agent.utils.async_utils import run_coroutine

if TYPE_CHECKING:
    from fast_agent.batch.runner import BatchBackend


class BatchScorer(Protocol):
    def __call__(
        self,
        result: BatchRunResult,
        candidate: Mapping[str, str],
        candidate_run: CandidateRun,
    ) -> tuple[float, Any] | tuple[float, Any, Mapping[str, Any]]: ...


CommandRunner = Callable[[Sequence[str], Path | None, float | None], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ReflectionAudit:
    index: int
    path: Path
    prompt_path: Path
    response_path: Path
    request_path: Path
    timing_path: Path


class FastAgentReflectionLM:
    """Synchronous language-model callable matching GEPA's structural protocol."""

    def __init__(
        self,
        *,
        env_dir: str | Path | None = None,
        model: str | None = None,
        audit_dir: str | Path,
        agent: str | None = None,
        timeout_seconds: float | None = None,
        command_runner: CommandRunner | None = None,
    ):
        self.env_dir = Path(env_dir) if env_dir is not None else None
        self.model = model
        self.audit_dir = Path(audit_dir)
        self.agent = agent
        self.timeout_seconds = timeout_seconds
        self.command_runner = command_runner or _run_command
        self._calls = 0

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        self._calls += 1
        audit = self._prepare_audit(prompt)
        command = self._command(prompt, audit)
        started = time.monotonic()
        try:
            completed = self.command_runner(command, None, self.timeout_seconds)
        except Exception as exc:
            (audit.path / "error.json").write_text(
                json.dumps({"type": type(exc).__name__, "message": str(exc)}, indent=2) + "\n",
                encoding="utf-8",
            )
            raise
        duration = time.monotonic() - started
        (audit.path / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (audit.path / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        audit.timing_path.write_text(
            json.dumps({"duration_seconds": duration, "returncode": completed.returncode}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            (audit.path / "error.json").write_text(
                json.dumps(
                    {"type": "FastAgentReflectionError", "returncode": completed.returncode},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(f"fast-agent reflection command failed: {completed.returncode}")
        usage = self._usage_from_results(audit.path / "results.json")
        if usage:
            (audit.path / "usage.json").write_text(
                json.dumps(usage, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        response = completed.stdout.strip()
        audit.response_path.write_text(response + "\n", encoding="utf-8")
        return response

    def _prepare_audit(self, prompt: str | list[dict[str, Any]]) -> ReflectionAudit:
        call_dir = self.audit_dir / f"call-{self._calls:04d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(prompt, str):
            prompt_path = call_dir / "prompt.md"
            prompt_path.write_text(prompt, encoding="utf-8")
        else:
            prompt_path = call_dir / "prompt.json"
            prompt_path.write_text(json.dumps(prompt, ensure_ascii=False, indent=2) + "\n", "utf-8")
        request_path = call_dir / "request.json"
        request_path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "agent": self.agent,
                    "env_dir": str(self.env_dir) if self.env_dir is not None else None,
                    "backend": "process",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return ReflectionAudit(
            index=self._calls,
            path=call_dir,
            prompt_path=prompt_path,
            response_path=call_dir / "response.txt",
            request_path=request_path,
            timing_path=call_dir / "timing.json",
        )

    def _command(self, prompt: str | list[dict[str, Any]], audit: ReflectionAudit) -> list[str]:
        command = [sys.executable, "-m", "fast_agent.cli.__main__", "go", "--quiet"]
        if self.env_dir is not None:
            command.extend(["--env", str(self.env_dir)])
        if self.model is not None:
            command.extend(["--model", self.model])
        if self.agent is not None:
            command.extend(["--agent", self.agent])
        if isinstance(prompt, str):
            command.extend(["--message", prompt])
        else:
            command.extend(["--prompt-file", str(audit.prompt_path)])
        command.extend(["--results", str(audit.path / "results.json")])
        return command

    @staticmethod
    def _usage_from_results(results_path: Path) -> dict[str, Any]:
        if not results_path.exists():
            return {}
        try:
            doc = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        summaries: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        messages = doc.get("messages")
        if not isinstance(messages, list):
            return {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            channels = message.get("channels")
            if not isinstance(channels, dict):
                continue
            usage_blocks = channels.get("fast-agent-usage")
            if not isinstance(usage_blocks, list):
                continue
            for block in usage_blocks:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if not isinstance(text, str):
                    continue
                try:
                    usage = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(usage.get("turn"), dict):
                    turns.append(usage["turn"])
                if isinstance(usage.get("summary"), dict):
                    summaries.append(usage["summary"])
        return {"turns": turns, "summary": summaries[-1] if summaries else {}}


class FastAgentBatchEvaluator:
    """GEPA-style candidate -> (score, side_info) evaluator for row/batch tasks."""

    def __init__(
        self,
        *,
        env_dir: str | Path | None = None,
        agent_card: str | Path,
        agent: str | None = None,
        candidate_variables: Mapping[str, str],
        input: str | Path,
        template: str | None = None,
        template_source: str | Path | None = None,
        schema: str | Path | None = None,
        model: str | None = None,
        parallel: int | None = None,
        scorer: BatchScorer,
        run_dir: str | Path,
        backend: BatchBackend = "harness",
        include_input: bool = True,
    ):
        self.env_dir = Path(env_dir) if env_dir is not None else None
        self.agent_card = agent_card
        self.agent = agent
        self.candidate_variables = dict(candidate_variables)
        self.input = input
        self.template = template
        self.template_source = template_source
        self.schema = schema
        self.model = model
        self.parallel = parallel
        self.scorer = scorer
        self.run = ArtifactRun(run_dir)
        self.backend = backend
        self.include_input = include_input

    def __call__(self, candidate: Mapping[str, str]) -> tuple[float, Any]:
        candidate_run = self.run.candidate()
        variables = self._variables_for_candidate(candidate)
        candidate_run.materialize_candidate(candidate, variables=variables)
        result = run_coroutine(self._run_batch(candidate_run, variables))
        score_result = self.scorer(result, candidate, candidate_run)
        if len(score_result) == 2:
            score, side_info = score_result
            metadata: Mapping[str, Any] = {}
        else:
            score, side_info, metadata = score_result
        candidate_run.write_score(score, side_info, metadata=metadata)
        return score, side_info

    def _variables_for_candidate(self, candidate: Mapping[str, str]) -> dict[str, str]:
        return {
            variable_name: candidate[candidate_key]
            for candidate_key, variable_name in self.candidate_variables.items()
        }

    async def _run_batch(
        self,
        candidate_run: CandidateRun,
        variables: dict[str, str],
    ) -> BatchRunResult:
        runner = BatchRunner(env_dir=self.env_dir, backend=self.backend)
        return await runner.run(
            input=self.input,
            output_path=candidate_run.path / "results.jsonl",
            agent_card=self.agent_card,
            agent=self.agent,
            template=self.template,
            template_source=self.template_source,
            json_schema=self.schema,
            model=self.model,
            parallel=self.parallel,
            include_input=self.include_input,
            variables=variables,
            summary_path=candidate_run.path / "batch-summary.json",
            telemetry_path=candidate_run.path / "telemetry.jsonl",
            overwrite=True,
        )


def _run_command(
    command: Sequence[str],
    cwd: Path | None,
    timeout_seconds: float | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        timeout=timeout_seconds,
        capture_output=True,
        text=True,
        check=False,
    )
