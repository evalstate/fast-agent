"""GEPA-compatible adapters built on fast-agent primitives."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeGuard

from fast_agent.batch import BatchRunner, BatchRunResult
from fast_agent.eval import ArtifactRun, CandidateRun
from fast_agent.utils.async_utils import run_coroutine

if TYPE_CHECKING:
    from fast_agent.batch.runner import BatchBackend


JsonRow: TypeAlias = dict[str, Any]
NumericMetric: TypeAlias = int | float

GEPA_EXTRA_REQUIREMENTS = {"gepa": "gepa"}
GEPA_EXTRA_INSTALL_MESSAGE = (
    'Install fast-agent with the GEPA extra, "fast-agent-mcp[gepa]", '
    "in the environment where fast-agent runs. Trackio-specific GEPA helpers require "
    "a GEPA release with Trackio support; until that is released, install GEPA from "
    "the Trackio integration branch in your application environment."
)


class GEPAIntegrationError(RuntimeError):
    """Raised when an installed GEPA package does not provide the expected API."""


def missing_gepa_dependencies() -> list[str]:
    """Return package names missing for GEPA optimizer integration."""

    return [
        package for module, package in GEPA_EXTRA_REQUIREMENTS.items() if find_spec(module) is None
    ]


def format_missing_gepa_dependencies(packages: list[str]) -> str:
    """Render an actionable missing-dependency error for GEPA optimizer workflows."""

    package_lines = "\n".join(f"  - {package}" for package in packages)
    return (
        "GEPA optimization requires optional dependencies that are not installed:\n"
        f"{package_lines}\n\n"
        f"{GEPA_EXTRA_INSTALL_MESSAGE}"
    )


def gepa_trackio_init_kwargs(
    *,
    project: str = "fast-agent-gepa",
    name: str | None = None,
    group: str | None = None,
    config: Mapping[str, Any] | None = None,
    embed: bool = False,
    auto_log_gpu: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Return sensible ``trackio.init`` kwargs for fast-agent GEPA runs.

    The returned mapping is intentionally plain so callers can use it with either
    a direct ``trackio.init(**kwargs)`` call or GEPA's ``trackio_init_kwargs``.
    """

    init_kwargs: dict[str, Any] = {
        "project": project,
        "embed": embed,
        "auto_log_gpu": auto_log_gpu,
    }
    if name is not None:
        init_kwargs["name"] = name
    if group is not None:
        init_kwargs["group"] = group
    if config is not None:
        init_kwargs["config"] = dict(config)
    init_kwargs.update(overrides)
    return init_kwargs


def gepa_api_trackio_kwargs(
    *,
    project: str = "fast-agent-gepa",
    name: str | None = None,
    group: str | None = None,
    config: Mapping[str, Any] | None = None,
    step_metric: str = "gepa/iteration",
    key_prefix: str = "gepa/",
    attach_existing: bool = False,
    **trackio_init_overrides: Any,
) -> dict[str, Any]:
    """Return Trackio kwargs for ``gepa.api.optimize``.

    Use ``attach_existing=True`` when your script has already called
    ``trackio.init`` and GEPA should log into that active run. Otherwise GEPA
    will initialize Trackio with the returned ``trackio_init_kwargs``.
    """

    return {
        "use_trackio": True,
        "trackio_init_kwargs": gepa_trackio_init_kwargs(
            project=project,
            name=name,
            group=group,
            config=config,
            **trackio_init_overrides,
        ),
        "trackio_attach_existing": attach_existing,
        "trackio_step_metric": step_metric,
        "tracking_key_prefix": key_prefix,
    }


def make_gepa_tracking_config(
    *,
    project: str = "fast-agent-gepa",
    name: str | None = None,
    group: str | None = None,
    config: Mapping[str, Any] | None = None,
    step_metric: str = "gepa/iteration",
    key_prefix: str = "gepa/",
    attach_existing: bool = False,
    **trackio_init_overrides: Any,
) -> Any:
    """Build GEPA's ``TrackingConfig`` with fast-agent Trackio defaults.

    The GEPA import is dynamic so importing ``fast_agent.integrations.gepa`` does
    not require the optional ``[gepa]`` dependency.
    """

    try:
        optimize_anything_module = importlib.import_module("gepa.optimize_anything")
    except ImportError as exc:
        raise GEPAIntegrationError(
            f"GEPA TrackingConfig requires GEPA to be installed. {GEPA_EXTRA_INSTALL_MESSAGE}"
        ) from exc
    tracking_config = optimize_anything_module.__dict__.get("TrackingConfig")
    if not callable(tracking_config):
        raise GEPAIntegrationError(
            "GEPA is installed, but gepa.optimize_anything.TrackingConfig is unavailable."
        )
    return tracking_config(
        use_trackio=True,
        trackio_init_kwargs=gepa_trackio_init_kwargs(
            project=project,
            name=name,
            group=group,
            config=config,
            **trackio_init_overrides,
        ),
        trackio_attach_existing=attach_existing,
        trackio_step_metric=step_metric,
        key_prefix=key_prefix,
    )


def gepa_numeric_metrics(
    side_info: Mapping[str, Any],
    *,
    score_prefix: str = "candidate/",
    detail_prefix: str = "candidate/detail/",
    raw_prefix: str = "candidate/raw/",
) -> dict[str, NumericMetric]:
    """Flatten GEPA side-info scores/details into Trackio-friendly metrics.

    ``side_info["scores"]`` is treated as frontier-safe, higher-is-better
    metrics. ``score_details`` and ``raw_metrics`` are treated as diagnostics
    and logged under separate prefixes.
    """

    metrics: dict[str, NumericMetric] = {}
    _extend_numeric_metrics(metrics, side_info.get("scores"), prefix=score_prefix)
    _extend_numeric_metrics(metrics, side_info.get("score_details"), prefix=detail_prefix)
    _extend_numeric_metrics(metrics, side_info.get("raw_metrics"), prefix=raw_prefix)
    return metrics


def safe_trackio_log(payload: Mapping[str, Any], *, step: int | None = None) -> bool:
    """Best-effort Trackio logging for evaluator-specific metrics.

    Returns ``True`` when a Trackio log call was made successfully and ``False``
    when Trackio is not installed or rejects the payload.
    """

    try:
        trackio = importlib.import_module("trackio")
        log = trackio.__dict__.get("log")
        if not callable(log):
            return False
        log(dict(payload), step=step)
        return True
    except Exception:
        return False


class BatchScorer(Protocol):
    def __call__(
        self,
        result: BatchRunResult,
        candidate: Mapping[str, str],
        candidate_run: CandidateRun,
    ) -> tuple[float, Any] | tuple[float, Any, Mapping[str, Any]]: ...


@dataclass(frozen=True)
class RowWiseEvaluationRun:
    """Inspectable directory and batch result for one GEPA row-wise evaluation."""

    index: int
    path: Path
    input_path: Path
    result: BatchRunResult


@dataclass(frozen=True)
class RowWiseScore:
    """Per-row score payload returned by ``FastAgentRowWiseBatchAdapter`` scorers."""

    score: float
    trajectory: Any = None
    objective_scores: Mapping[str, float] | None = None


class RowWiseBatchScorer(Protocol):
    def __call__(
        self,
        output_row: JsonRow,
        input_row: JsonRow,
        candidate: Mapping[str, str],
        evaluation: RowWiseEvaluationRun,
    ) -> RowWiseScore | float | tuple[float, Any] | tuple[float, Any, Mapping[str, float]]: ...


class ReflectiveDatasetBuilder(Protocol):
    def __call__(
        self,
        candidate: Mapping[str, str],
        eval_batch: Any,
        components_to_update: Sequence[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]: ...


class BatchRunnerFactory(Protocol):
    def __call__(self, env_dir: Path | None, *, backend: "BatchBackend") -> BatchRunner: ...


CommandRunner = Callable[
    [Sequence[str], Path | None, float | None], subprocess.CompletedProcess[str]
]


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
        self._pending_gepa_reflection_metrics: deque[dict[str, NumericMetric]] = deque()

    def pop_pending_gepa_reflection_metrics(self) -> list[dict[str, NumericMetric]]:
        metrics = list(self._pending_gepa_reflection_metrics)
        self._pending_gepa_reflection_metrics.clear()
        return metrics

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
            self._pending_gepa_reflection_metrics.append(
                _reflection_usage_metrics(
                    usage,
                    call_index=self._calls,
                    duration_seconds=duration,
                )
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


class FastAgentRowWiseBatchAdapter:
    """GEPA adapter protocol implementation for row-wise fast-agent batch evaluation.

    ``FastAgentBatchEvaluator`` evaluates a full batch as one aggregate GEPA metric call.
    This adapter is for GEPA's lower-level ``gepa.api.optimize`` path: GEPA supplies a
    minibatch of input rows, fast-agent runs those rows through ``BatchRunner``, and the
    caller-provided scorer returns one score/trajectory per row.
    """

    propose_new_texts = None

    def __init__(
        self,
        *,
        env_dir: str | Path | None = None,
        agent_card: str | Path,
        agent: str | None = None,
        candidate_variables: Mapping[str, str],
        template: str | None = None,
        template_source: str | Path | None = None,
        schema: str | Path | None = None,
        model: str | None = None,
        parallel: int | None = None,
        row_scorer: RowWiseBatchScorer,
        run_dir: str | Path,
        backend: "BatchBackend" = "harness",
        include_input: bool = True,
        id_field: str | None = None,
        reflective_dataset_builder: ReflectiveDatasetBuilder | None = None,
        batch_runner_factory: BatchRunnerFactory | None = None,
    ):
        self.env_dir = Path(env_dir) if env_dir is not None else None
        self.agent_card = agent_card
        self.agent = agent
        self.candidate_variables = dict(candidate_variables)
        self.template = template
        self.template_source = template_source
        self.schema = schema
        self.model = model
        self.parallel = parallel
        self.row_scorer = row_scorer
        self.run_dir = Path(run_dir)
        self.backend = backend
        self.include_input = include_input
        self.id_field = id_field
        self.reflective_dataset_builder = reflective_dataset_builder
        self.batch_runner_factory = batch_runner_factory or BatchRunner
        self._evaluations = 0
        self._pending_gepa_eval_metrics: deque[dict[str, NumericMetric]] = deque()

    def pop_pending_gepa_eval_metrics(self) -> dict[str, NumericMetric] | None:
        if not self._pending_gepa_eval_metrics:
            return None
        return self._pending_gepa_eval_metrics.popleft()

    def evaluate(
        self,
        batch: list[JsonRow],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        self._evaluations += 1
        eval_dir = self.run_dir / "row-wise-evals" / f"eval-{self._evaluations:05d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        input_rows = [dict(row) for row in batch]
        input_path = eval_dir / "input.jsonl"
        _write_jsonl(input_path, input_rows)

        variables = self._variables_for_candidate(candidate)
        result = run_coroutine(self._run_batch(eval_dir, input_path, variables, len(input_rows)))
        evaluation = RowWiseEvaluationRun(
            index=self._evaluations,
            path=eval_dir,
            input_path=input_path,
            result=result,
        )
        output_rows = _align_output_rows(
            input_rows=input_rows,
            output_rows=result.rows,
            id_field=self.id_field,
        )
        scores: list[float] = []
        trajectories: list[Any] = []
        objective_scores: list[dict[str, float]] = []
        for input_row, output_row in zip(input_rows, output_rows, strict=True):
            parsed = _parse_row_wise_score(
                self.row_scorer(output_row, input_row, candidate, evaluation)
            )
            scores.append(parsed.score)
            trajectories.append(parsed.trajectory)
            objective_scores.append(dict(parsed.objective_scores or {}))

        summary = {
            "eval_index": self._evaluations,
            "batch_size": len(input_rows),
            "avg_score": sum(scores) / max(1, len(scores)),
            "num_metric_calls": len(input_rows),
            "objective_averages": _objective_averages(objective_scores),
        }
        (eval_dir / "row-wise-score.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._pending_gepa_eval_metrics.append(
            _row_wise_eval_metrics(
                result.summary,
                summary,
            )
        )
        return _evaluation_batch(
            outputs=output_rows,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            objective_scores=objective_scores,
            num_metric_calls=len(input_rows),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if self.reflective_dataset_builder is not None:
            return self.reflective_dataset_builder(candidate, eval_batch, components_to_update)
        return _default_reflective_dataset(eval_batch, components_to_update)

    def _variables_for_candidate(self, candidate: Mapping[str, str]) -> dict[str, str]:
        return {
            variable_name: candidate[candidate_key]
            for candidate_key, variable_name in self.candidate_variables.items()
        }

    async def _run_batch(
        self,
        eval_dir: Path,
        input_path: Path,
        variables: dict[str, str],
        batch_size: int,
    ) -> BatchRunResult:
        runner = self.batch_runner_factory(self.env_dir, backend=self.backend)
        return await runner.run(
            input=input_path,
            output_path=eval_dir / "results.jsonl",
            agent_card=self.agent_card,
            agent=self.agent,
            template=self.template,
            template_source=self.template_source,
            json_schema=self.schema,
            model=self.model,
            parallel=min(self.parallel, batch_size) if self.parallel is not None else None,
            include_input=self.include_input,
            variables=variables,
            summary_path=eval_dir / "batch-summary.json",
            telemetry_path=eval_dir / "telemetry.jsonl",
            id_field=self.id_field,
            overwrite=True,
        )


class FastAgentGEPATrackioCallback:
    """GEPA callback that logs compact fast-agent telemetry on GEPA's x-axis."""

    def __init__(
        self,
        *,
        row_wise_adapter: FastAgentRowWiseBatchAdapter | None = None,
        reflection_lm: FastAgentReflectionLM | None = None,
        include_gepa_context: bool = True,
        include_eval_score_summary: bool = True,
    ) -> None:
        self.row_wise_adapter = row_wise_adapter
        self.reflection_lm = reflection_lm
        self.include_gepa_context = include_gepa_context
        self.include_eval_score_summary = include_eval_score_summary

    def on_evaluation_end(self, event: Mapping[str, Any]) -> None:
        if self.row_wise_adapter is None:
            return
        metrics = self.row_wise_adapter.pop_pending_gepa_eval_metrics()
        if metrics is not None:
            safe_trackio_log(
                {
                    **_evaluation_event_metrics(event, include_context=self.include_gepa_context),
                    **_filter_eval_metrics(
                        metrics,
                        include_score_summary=self.include_eval_score_summary,
                    ),
                }
            )

    def on_proposal_end(self, event: Mapping[str, Any]) -> None:
        if self.reflection_lm is None:
            return
        context = _proposal_event_metrics(event, include_context=self.include_gepa_context)
        for metrics in self.reflection_lm.pop_pending_gepa_reflection_metrics():
            safe_trackio_log({**context, **metrics})


@dataclass
class _FallbackEvaluationBatch:
    outputs: list[Any]
    scores: list[float]
    trajectories: list[Any] | None = None
    objective_scores: list[dict[str, float]] | None = None
    num_metric_calls: int | None = None


def _evaluation_batch(
    *,
    outputs: list[Any],
    scores: list[float],
    trajectories: list[Any] | None,
    objective_scores: list[dict[str, float]] | None,
    num_metric_calls: int,
) -> Any:
    try:
        adapter_module = importlib.import_module("gepa.core.adapter")
    except ImportError as exc:
        if missing_gepa_dependencies():
            return _fallback_evaluation_batch(
                outputs=outputs,
                scores=scores,
                trajectories=trajectories,
                objective_scores=objective_scores,
                num_metric_calls=num_metric_calls,
            )
        raise GEPAIntegrationError(
            "GEPA is installed, but fast-agent could not import gepa.core.adapter. "
            f"{GEPA_EXTRA_INSTALL_MESSAGE}"
        ) from exc
    except Exception as exc:
        raise GEPAIntegrationError(
            f"GEPA is installed, but loading gepa.core.adapter failed. {GEPA_EXTRA_INSTALL_MESSAGE}"
        ) from exc
    evaluation_batch = adapter_module.__dict__.get("EvaluationBatch")
    if not callable(evaluation_batch):
        raise GEPAIntegrationError(
            "GEPA is installed, but gepa.core.adapter.EvaluationBatch is unavailable. "
            "Install a GEPA version that provides the row-wise adapter API."
        )
    try:
        return evaluation_batch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
            num_metric_calls=num_metric_calls,
        )
    except TypeError:
        return _fallback_evaluation_batch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
            num_metric_calls=num_metric_calls,
        )


def _fallback_evaluation_batch(
    *,
    outputs: list[Any],
    scores: list[float],
    trajectories: list[Any] | None,
    objective_scores: list[dict[str, float]] | None,
    num_metric_calls: int,
) -> _FallbackEvaluationBatch:
    return _FallbackEvaluationBatch(
        outputs=outputs,
        scores=scores,
        trajectories=trajectories,
        objective_scores=objective_scores,
        num_metric_calls=num_metric_calls,
    )


def _extend_numeric_metrics(
    metrics: dict[str, NumericMetric],
    values: Any,
    *,
    prefix: str,
) -> None:
    if not isinstance(values, Mapping):
        return
    for key, value in values.items():
        if isinstance(key, str) and _is_numeric_metric(value):
            metrics[f"{prefix}{key}"] = value


def _row_wise_eval_metrics(
    batch_summary: Mapping[str, Any],
    row_score_summary: Mapping[str, Any],
) -> dict[str, NumericMetric]:
    metrics: dict[str, NumericMetric] = {}
    for key in ("eval_index", "batch_size", "avg_score", "num_metric_calls"):
        value = row_score_summary.get(key)
        if _is_numeric_metric(value):
            metrics[f"fast_agent/eval/{key}"] = value
    for key in ("processed_rows", "failed_rows", "skipped_rows", "selected_rows", "duration_ms"):
        value = batch_summary.get(key)
        if _is_numeric_metric(value):
            metrics[f"fast_agent/eval/{key}"] = value
    usage = batch_summary.get("usage")
    if isinstance(usage, Mapping):
        _copy_selected_metrics(
            metrics,
            usage,
            prefix="fast_agent/eval/usage/",
            names=(
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "billing_tokens",
                "reasoning_tokens",
                "tool_use_tokens",
                "tool_calls",
                "rows_with_usage",
                "usage_coverage_percent",
            ),
        )
        rows_with_usage = usage.get("rows_with_usage")
        billing_tokens = usage.get("billing_tokens")
        if _is_numeric_metric(rows_with_usage) and rows_with_usage > 0 and _is_numeric_metric(billing_tokens):
            metrics["fast_agent/eval/usage/billing_tokens_per_row"] = billing_tokens / rows_with_usage
    cache = batch_summary.get("cache")
    if isinstance(cache, Mapping):
        _copy_selected_metrics(
            metrics,
            cache,
            prefix="fast_agent/eval/cache/",
            names=(
                "read_tokens",
                "write_tokens",
                "hit_tokens",
                "served_tokens",
                "activity_tokens",
                "effective_input_tokens",
                "hit_rate_percent",
                "write_rate_percent",
                "activity_rate_percent",
                "served_to_effective_input_ratio",
            ),
        )
    return metrics


def _filter_eval_metrics(
    metrics: Mapping[str, NumericMetric],
    *,
    include_score_summary: bool,
) -> dict[str, NumericMetric]:
    if include_score_summary:
        return dict(metrics)
    blocked = {
        "fast_agent/eval/batch_size",
        "fast_agent/eval/avg_score",
        "fast_agent/eval/num_metric_calls",
    }
    return {key: value for key, value in metrics.items() if key not in blocked}


def _reflection_usage_metrics(
    usage: Mapping[str, Any],
    *,
    call_index: int,
    duration_seconds: float,
) -> dict[str, NumericMetric]:
    metrics: dict[str, NumericMetric] = {
        "fast_agent/reflection/call_index": call_index,
        "fast_agent/reflection/duration_seconds": duration_seconds,
    }
    summary = usage.get("summary")
    if isinstance(summary, Mapping):
        _copy_selected_metrics(
            metrics,
            summary,
            prefix="fast_agent/reflection/usage/",
            names=(
                "cumulative_input_tokens",
                "cumulative_output_tokens",
                "cumulative_total_tokens",
                "cumulative_billing_tokens",
                "cumulative_reasoning_tokens",
                "cumulative_cache_read_tokens",
                "cumulative_cache_write_tokens",
                "cumulative_cache_hit_tokens",
                "cache_hit_rate_percent",
            ),
        )
    turns = usage.get("turns")
    if isinstance(turns, Sequence) and not isinstance(turns, str | bytes):
        metrics["fast_agent/reflection/turns"] = len(turns)
    return metrics


def _evaluation_event_metrics(
    event: Mapping[str, Any],
    *,
    include_context: bool = True,
) -> dict[str, NumericMetric]:
    metrics: dict[str, NumericMetric] = {}
    metric_names = {
        "iteration": "gepa/iteration",
        "candidate_idx": "gepa/candidate_idx",
    }
    if include_context:
        metric_names.update(
            {
                "batch_size": "fast_agent/gepa_context/batch_size",
                "capture_traces": "fast_agent/gepa_context/capture_traces",
                "is_seed_candidate": "fast_agent/gepa_context/is_seed_candidate",
            }
        )
    for source_key, metric_key in metric_names.items():
        value = event.get(source_key)
        if isinstance(value, bool):
            metrics[metric_key] = int(value)
        elif _is_numeric_metric(value):
            metrics[metric_key] = value
    if not include_context:
        return metrics
    parent_ids = event.get("parent_ids")
    if isinstance(parent_ids, Sequence) and not isinstance(parent_ids, str | bytes):
        metrics["fast_agent/gepa_context/parent_count"] = len(parent_ids)
    scores = event.get("scores")
    if isinstance(scores, Sequence) and not isinstance(scores, str | bytes):
        numeric_scores = [score for score in scores if _is_numeric_metric(score)]
        if numeric_scores:
            metrics["fast_agent/gepa_context/score_mean"] = sum(numeric_scores) / len(numeric_scores)
    return metrics


def _proposal_event_metrics(
    event: Mapping[str, Any],
    *,
    include_context: bool = True,
) -> dict[str, NumericMetric]:
    metrics: dict[str, NumericMetric] = {}
    iteration = event.get("iteration")
    if _is_numeric_metric(iteration):
        metrics["gepa/iteration"] = iteration
    if not include_context:
        return metrics
    new_instructions = event.get("new_instructions")
    if isinstance(new_instructions, Mapping):
        metrics["fast_agent/gepa_context/proposed_components"] = len(new_instructions)
    prompts = event.get("prompts")
    if isinstance(prompts, Mapping):
        metrics["fast_agent/gepa_context/proposal_prompts"] = len(prompts)
    return metrics


def _copy_selected_metrics(
    metrics: dict[str, NumericMetric],
    values: Mapping[str, Any],
    *,
    prefix: str,
    names: Sequence[str],
) -> None:
    for name in names:
        value = values.get(name)
        if _is_numeric_metric(value):
            metrics[f"{prefix}{name}"] = value


def _is_numeric_metric(value: Any) -> TypeGuard[NumericMetric]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parse_row_wise_score(
    value: RowWiseScore | float | tuple[float, Any] | tuple[float, Any, Mapping[str, float]],
) -> RowWiseScore:
    if isinstance(value, RowWiseScore):
        return value
    if isinstance(value, int | float):
        return RowWiseScore(score=float(value))
    if len(value) == 2:
        score, trajectory = value
        return RowWiseScore(score=float(score), trajectory=trajectory)
    score, trajectory, objective_scores = value
    return RowWiseScore(
        score=float(score),
        trajectory=trajectory,
        objective_scores=objective_scores,
    )


def _objective_averages(objective_scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row_scores in objective_scores:
        for key, value in row_scores.items():
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in sorted(totals)}


def _align_output_rows(
    *,
    input_rows: Sequence[JsonRow],
    output_rows: Sequence[JsonRow],
    id_field: str | None,
) -> list[JsonRow]:
    if id_field is None:
        return [
            output_rows[index]
            if index < len(output_rows)
            else _missing_output_row(input_row, id_field)
            for index, input_row in enumerate(input_rows)
        ]

    by_id: dict[Any, JsonRow] = {}
    for output_row in output_rows:
        identity = _row_identity(output_row, id_field)
        if identity is not None:
            by_id[identity] = output_row

    aligned: list[JsonRow] = []
    for index, input_row in enumerate(input_rows):
        identity = _row_identity(input_row, id_field)
        if identity is not None and identity in by_id:
            aligned.append(by_id[identity])
        elif index < len(output_rows):
            aligned.append(output_rows[index])
        else:
            aligned.append(_missing_output_row(input_row, id_field))
    return aligned


def _row_identity(row: Mapping[str, Any], id_field: str) -> Any:
    nested_input = row.get("input")
    if isinstance(nested_input, dict) and nested_input.get(id_field) is not None:
        return nested_input[id_field]
    return row.get(id_field)


def _missing_output_row(input_row: JsonRow, id_field: str | None) -> JsonRow:
    row: JsonRow = {
        "ok": False,
        "input": dict(input_row),
        "error": "No output row was returned for this input.",
    }
    if id_field is not None and input_row.get(id_field) is not None:
        row[id_field] = input_row[id_field]
    return row


def _default_reflective_dataset(
    eval_batch: Any,
    components_to_update: Sequence[str],
) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    try:
        trajectories = eval_batch.trajectories
    except AttributeError:
        trajectories = []
    try:
        scores = eval_batch.scores
    except AttributeError:
        scores = []
    if not isinstance(trajectories, list):
        trajectories = []
    if not isinstance(scores, list):
        scores = []

    rows: list[dict[str, Any]] = []
    for index, trajectory in enumerate(trajectories):
        row = _trajectory_to_reflective_row(trajectory)
        if index < len(scores):
            row["selected_row_score"] = scores[index]
        rows.append(row)
    return {component: list(rows) for component in components_to_update}


def _trajectory_to_reflective_row(trajectory: Any) -> dict[str, Any]:
    if not isinstance(trajectory, Mapping):
        return {"trajectory": trajectory}
    row: dict[str, Any] = {}
    for key, value in trajectory.items():
        row["Scores (Higher is Better)" if key == "scores" else str(key)] = value
    return row


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
