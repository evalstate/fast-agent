import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fast_agent.batch import BatchRunResult
from fast_agent.integrations.gepa import (
    FastAgentBatchEvaluator,
    FastAgentGEPATrackioCallback,
    FastAgentReflectionLM,
    FastAgentRowWiseBatchAdapter,
    RowWiseEvaluationRun,
    RowWiseScore,
    _evaluation_batch,
    gepa_api_trackio_kwargs,
    gepa_numeric_metrics,
    gepa_trackio_init_kwargs,
)


def test_reflection_lm_writes_audit_files(tmp_path):
    commands = []

    def fake_runner(command, cwd, timeout_seconds):
        commands.append((command, cwd, timeout_seconds))
        results_path = command[command.index("--results") + 1]
        with open(results_path, "w", encoding="utf-8") as handle:
            handle.write(
                '{"messages":[{"channels":{"fast-agent-usage":[{"text":"'
                '{\\"turn\\":{\\"input_tokens\\":3},'
                '\\"summary\\":{\\"cumulative_input_tokens\\":3}}"}]}}]}'
            )
        return subprocess.CompletedProcess(command, 0, stdout="reflection\n", stderr="")

    lm = FastAgentReflectionLM(
        env_dir=tmp_path / "env",
        model="passthrough",
        audit_dir=tmp_path / "audit",
        command_runner=fake_runner,
    )

    assert lm("think about this") == "reflection"
    call_dir = tmp_path / "audit" / "call-0001"
    assert (call_dir / "prompt.md").read_text(encoding="utf-8") == "think about this"
    assert (call_dir / "response.txt").read_text(encoding="utf-8") == "reflection\n"
    assert (call_dir / "usage.json").exists()
    assert "--results" in commands[0][0]
    assert "--env" in commands[0][0]
    assert "--env-dir" not in commands[0][0]


def test_reflection_lm_logs_usage_metrics(monkeypatch, tmp_path):
    logged: list[dict[str, Any]] = []

    def fake_runner(command, cwd, timeout_seconds):
        results_path = command[command.index("--results") + 1]
        Path(results_path).write_text(
            json.dumps(
                {
                    "messages": [
                        {
                            "channels": {
                                "fast-agent-usage": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "turn": {
                                                    "input_tokens": 10,
                                                    "output_tokens": 2,
                                                    "total_tokens": 12,
                                                    "effective_input_tokens": 6,
                                                    "cache_usage": {"cache_hit_tokens": 4},
                                                },
                                                "summary": {
                                                    "cumulative_input_tokens": 10,
                                                    "cumulative_output_tokens": 2,
                                                    "cumulative_billing_tokens": 12,
                                                    "cumulative_cache_hit_tokens": 4,
                                                    "cache_hit_rate_percent": 40.0,
                                                },
                                            }
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="reflection\n", stderr="")

    monkeypatch.setattr(
        "fast_agent.integrations.gepa.safe_trackio_log",
        lambda payload, **kwargs: logged.append(dict(payload)) or True,
    )

    lm = FastAgentReflectionLM(
        env_dir=tmp_path / "env",
        model="passthrough",
        audit_dir=tmp_path / "audit",
        command_runner=fake_runner,
    )

    assert lm("think about this") == "reflection"
    assert not logged
    FastAgentGEPATrackioCallback(reflection_lm=lm).on_proposal_end(
        {
            "iteration": 3,
            "new_instructions": {"policy": "new"},
            "prompts": {"policy": "prompt"},
            "raw_lm_outputs": {"policy": "raw"},
        }
    )
    assert logged
    assert logged[0]["gepa/iteration"] == 3
    assert logged[0]["fast_agent/gepa_context/proposed_components"] == 1
    assert logged[0]["fast_agent/reflection/call_index"] == 1
    assert logged[0]["fast_agent/reflection/usage/cumulative_billing_tokens"] == 12
    assert logged[0]["fast_agent/reflection/usage/cache_hit_rate_percent"] == 40.0


def test_batch_evaluator_allocates_candidate_and_scores(monkeypatch, tmp_path):
    captured = {}

    async def fake_run(self, **kwargs):
        captured.update(kwargs)
        output_path = kwargs["output_path"]
        output_path.write_text('{"ok": true, "result": {"label": "A"}}\n', encoding="utf-8")
        return BatchRunResult(
            rows=[{"ok": True, "result": {"label": "A"}}],
            output_path=output_path,
            summary={"processed_rows": 1},
            telemetry_path=kwargs["telemetry_path"],
            error_output_path=None,
            summary_path=kwargs["summary_path"],
        )

    monkeypatch.setattr("fast_agent.integrations.gepa.BatchRunner.run", fake_run)

    def scorer(result, candidate, candidate_run):
        assert result.rows[0]["result"] == {"label": "A"}
        assert candidate["policy"] == "route A"
        assert candidate_run.path.name == "candidate-0001"
        return 1.0, "matched", {"rows": 1}

    evaluator = FastAgentBatchEvaluator(
        env_dir=tmp_path / "env",
        agent_card=tmp_path / "card.md",
        candidate_variables={"policy": "policy"},
        input=tmp_path / "input.jsonl",
        template="{{row_json}}",
        scorer=scorer,
        run_dir=tmp_path / "runs",
    )

    assert evaluator({"policy": "route A"}) == (1.0, "matched")
    assert captured["variables"] == {"policy": "route A"}
    assert (tmp_path / "runs" / "candidate-0001" / "score.json").exists()


class RecordingBatchRunner:
    def __init__(self, *, summary: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.summary = summary or {"processed_rows": 2}

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        rows = [
            {
                "id": "row-2",
                "ok": True,
                "input": {"id": "row-2", "expected": "B"},
                "result": {"label": "B"},
            },
            {
                "id": "row-1",
                "ok": True,
                "input": {"id": "row-1", "expected": "A"},
                "result": {"label": "C"},
            },
        ]
        output_path = Path(kwargs["output_path"])
        output_path.write_text(
            "".join(f"{row}\n" for row in rows),
            encoding="utf-8",
        )
        return BatchRunResult(
            rows=rows,
            output_path=output_path,
            summary=self.summary,
            telemetry_path=kwargs["telemetry_path"],
            error_output_path=None,
            summary_path=kwargs["summary_path"],
        )


def test_row_wise_batch_adapter_evaluates_minibatch_and_builds_reflection_rows(tmp_path):
    runner = RecordingBatchRunner()

    def runner_factory(env_dir, *, backend):
        assert env_dir == tmp_path / "env"
        assert backend == "harness"
        return runner

    def row_scorer(
        output_row: dict[str, Any],
        input_row: dict[str, Any],
        candidate: Mapping[str, str],
        evaluation: RowWiseEvaluationRun,
    ) -> RowWiseScore:
        assert candidate == {"policy": "route carefully"}
        assert evaluation.path.name == "eval-00001"
        actual = output_row["result"]["label"]
        expected = input_row["expected"]
        score = 1.0 if actual == expected else 0.0
        return RowWiseScore(
            score=score,
            trajectory={
                "scores": {"gepa_score": score, "exact": score},
                "expected": expected,
                "actual": actual,
            },
            objective_scores={"gepa_score": score, "exact": score},
        )

    adapter = FastAgentRowWiseBatchAdapter(
        env_dir=tmp_path / "env",
        agent_card=tmp_path / "card.md",
        candidate_variables={"policy": "policy"},
        template="{{row_json}}",
        row_scorer=row_scorer,
        run_dir=tmp_path / "runs",
        id_field="id",
        batch_runner_factory=runner_factory,
    )

    batch = [
        {"id": "row-1", "expected": "A"},
        {"id": "row-2", "expected": "B"},
    ]
    result = adapter.evaluate(batch, {"policy": "route carefully"}, capture_traces=True)

    assert result.scores == [0.0, 1.0]
    assert result.objective_scores == [
        {"gepa_score": 0.0, "exact": 0.0},
        {"gepa_score": 1.0, "exact": 1.0},
    ]
    assert result.num_metric_calls == 2
    assert result.outputs[0]["id"] == "row-1"
    assert result.outputs[1]["id"] == "row-2"
    assert runner.calls[0]["variables"] == {"policy": "route carefully"}
    assert runner.calls[0]["id_field"] == "id"
    assert (tmp_path / "runs" / "row-wise-evals" / "eval-00001" / "input.jsonl").exists()
    row_wise_score_path = (
        tmp_path / "runs" / "row-wise-evals" / "eval-00001" / "row-wise-score.json"
    )
    assert row_wise_score_path.exists()
    row_wise_score = json.loads(row_wise_score_path.read_text(encoding="utf-8"))
    assert row_wise_score["objective_averages"] == {"exact": 0.5, "gepa_score": 0.5}

    reflective = adapter.make_reflective_dataset(
        {"policy": "route carefully"},
        result,
        ["policy"],
    )

    assert list(reflective) == ["policy"]
    assert reflective["policy"][0]["Scores (Higher is Better)"] == {
        "gepa_score": 0.0,
        "exact": 0.0,
    }
    assert reflective["policy"][0]["selected_row_score"] == 0.0


def test_row_wise_batch_adapter_logs_batch_usage_and_cache(monkeypatch, tmp_path):
    logged: list[dict[str, Any]] = []
    runner = RecordingBatchRunner(
        summary={
            "processed_rows": 2,
            "failed_rows": 0,
            "skipped_rows": 0,
            "selected_rows": 2,
            "duration_ms": 1000,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "billing_tokens": 120,
                "reasoning_tokens": 7,
                "rows_with_usage": 2,
                "usage_coverage_percent": 100.0,
            },
            "cache": {
                "served_tokens": 60,
                "hit_tokens": 40,
                "write_tokens": 20,
                "effective_input_tokens": 40,
                "hit_rate_percent": 60.0,
            },
        }
    )

    def runner_factory(env_dir, *, backend):
        return runner

    def row_scorer(output_row, input_row, candidate, evaluation):
        score = 1.0 if output_row["result"]["label"] == input_row["expected"] else 0.0
        return RowWiseScore(
            score=score,
            objective_scores={"gepa_score": score},
        )

    monkeypatch.setattr(
        "fast_agent.integrations.gepa.safe_trackio_log",
        lambda payload, **kwargs: logged.append(dict(payload)) or True,
    )

    adapter = FastAgentRowWiseBatchAdapter(
        env_dir=tmp_path / "env",
        agent_card=tmp_path / "card.md",
        candidate_variables={"policy": "policy"},
        template="{{row_json}}",
        row_scorer=row_scorer,
        run_dir=tmp_path / "runs",
        id_field="id",
        batch_runner_factory=runner_factory,
    )

    adapter.evaluate(
        [{"id": "row-1", "expected": "A"}, {"id": "row-2", "expected": "B"}],
        {"policy": "route carefully"},
    )

    assert not logged
    FastAgentGEPATrackioCallback(row_wise_adapter=adapter).on_evaluation_end(
        {
            "iteration": 7,
            "candidate_idx": 4,
            "batch_size": 2,
            "capture_traces": True,
            "parent_ids": [1],
            "scores": [1.0, 0.0],
            "has_trajectories": False,
            "outputs": [],
            "trajectories": None,
            "objective_scores": None,
            "is_seed_candidate": False,
        }
    )
    assert logged
    payload = logged[0]
    assert payload["gepa/iteration"] == 7
    assert payload["gepa/candidate_idx"] == 4
    assert payload["fast_agent/gepa_context/parent_count"] == 1
    assert payload["fast_agent/gepa_context/score_mean"] == 0.5
    assert payload["fast_agent/eval/eval_index"] == 1
    assert payload["fast_agent/eval/batch_size"] == 2
    assert "fast_agent/eval/objective/gepa_score" not in payload
    assert payload["fast_agent/eval/usage/billing_tokens_per_row"] == 60
    assert payload["fast_agent/eval/cache/served_tokens"] == 60
    assert payload["fast_agent/eval/cache/hit_rate_percent"] == 60.0


def test_trackio_callback_can_log_operational_metrics_only(monkeypatch, tmp_path):
    logged: list[dict[str, Any]] = []
    runner = RecordingBatchRunner(
        summary={
            "processed_rows": 2,
            "failed_rows": 0,
            "duration_ms": 1000,
            "usage": {
                "billing_tokens": 120,
                "rows_with_usage": 2,
            },
            "cache": {
                "served_tokens": 60,
                "hit_rate_percent": 60.0,
            },
        }
    )

    def runner_factory(env_dir, *, backend):
        return runner

    def row_scorer(output_row, input_row, candidate, evaluation):
        return RowWiseScore(score=1.0, objective_scores={"gepa_score": 1.0})

    monkeypatch.setattr(
        "fast_agent.integrations.gepa.safe_trackio_log",
        lambda payload, **kwargs: logged.append(dict(payload)) or True,
    )

    adapter = FastAgentRowWiseBatchAdapter(
        env_dir=tmp_path / "env",
        agent_card=tmp_path / "card.md",
        candidate_variables={"policy": "policy"},
        template="{{row_json}}",
        row_scorer=row_scorer,
        run_dir=tmp_path / "runs",
        id_field="id",
        batch_runner_factory=runner_factory,
    )
    adapter.evaluate([{"id": "row-1"}], {"policy": "route carefully"})

    FastAgentGEPATrackioCallback(
        row_wise_adapter=adapter,
        include_gepa_context=False,
        include_eval_score_summary=False,
    ).on_evaluation_end(
        {
            "iteration": 7,
            "candidate_idx": 4,
            "batch_size": 1,
            "capture_traces": True,
            "parent_ids": [1],
            "scores": [1.0],
            "has_trajectories": False,
            "outputs": [],
            "trajectories": None,
            "objective_scores": None,
            "is_seed_candidate": False,
        }
    )

    assert logged
    payload = logged[0]
    assert payload["gepa/iteration"] == 7
    assert payload["gepa/candidate_idx"] == 4
    assert "fast_agent/gepa_context/score_mean" not in payload
    assert "fast_agent/gepa_context/batch_size" not in payload
    assert "fast_agent/eval/batch_size" not in payload
    assert "fast_agent/eval/avg_score" not in payload
    assert "fast_agent/eval/num_metric_calls" not in payload
    assert payload["fast_agent/eval/usage/billing_tokens_per_row"] == 60
    assert payload["fast_agent/eval/cache/served_tokens"] == 60


def test_gepa_numeric_metrics_flattens_scores_and_details():
    metrics = gepa_numeric_metrics(
        {
            "scores": {
                "gepa_score": 0.8,
                "valid_json": 1,
                "accepted": True,
                "note": "ignored",
            },
            "score_details": {
                "failure_count": 3,
                "latency_seconds": 1.25,
                "nested": {"ignored": 1},
            },
            "raw_metrics": {
                "tokens": 42,
            },
        }
    )

    assert metrics == {
        "candidate/gepa_score": 0.8,
        "candidate/valid_json": 1,
        "candidate/detail/failure_count": 3,
        "candidate/detail/latency_seconds": 1.25,
        "candidate/raw/tokens": 42,
    }


def test_gepa_trackio_kwargs_are_sensible_defaults():
    init_kwargs = gepa_trackio_init_kwargs(
        name="classifier",
        group="smoke",
        config={"agent": "classifier"},
        space_id="owner/space",
    )

    assert init_kwargs == {
        "project": "fast-agent-gepa",
        "name": "classifier",
        "group": "smoke",
        "embed": False,
        "auto_log_gpu": False,
        "config": {"agent": "classifier"},
        "space_id": "owner/space",
    }

    api_kwargs = gepa_api_trackio_kwargs(
        project="demo",
        name="row-wise",
        config={"mode": "row-wise"},
        attach_existing=True,
    )

    assert api_kwargs == {
        "use_trackio": True,
        "trackio_init_kwargs": {
            "project": "demo",
            "name": "row-wise",
            "embed": False,
            "auto_log_gpu": False,
            "config": {"mode": "row-wise"},
        },
        "trackio_attach_existing": True,
        "trackio_step_metric": "gepa/iteration",
        "tracking_key_prefix": "gepa/",
    }


def test_evaluation_batch_falls_back_when_gepa_is_not_installed():
    batch = _evaluation_batch(
        outputs=[{"ok": True}],
        scores=[1.0],
        trajectories=None,
        objective_scores=[{"gepa_score": 1.0}],
        num_metric_calls=1,
    )

    assert batch.outputs == [{"ok": True}]
    assert batch.scores == [1.0]
    assert batch.objective_scores == [{"gepa_score": 1.0}]
    assert batch.num_metric_calls == 1


def test_evaluation_batch_rejects_incompatible_gepa_package(tmp_path):
    package_root = tmp_path / "packages"
    adapter_path = package_root / "gepa" / "core" / "adapter.py"
    adapter_path.parent.mkdir(parents=True)
    (package_root / "gepa" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "gepa" / "core" / "__init__.py").write_text("", encoding="utf-8")
    adapter_path.write_text("class NotEvaluationBatch:\n    pass\n", encoding="utf-8")

    code = """
from fast_agent.integrations.gepa import GEPAIntegrationError, _evaluation_batch
try:
    _evaluation_batch(
        outputs=[],
        scores=[],
        trajectories=None,
        objective_scores=[],
        num_metric_calls=0,
    )
except GEPAIntegrationError as exc:
    print(str(exc))
else:
    raise SystemExit("expected GEPAIntegrationError")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[4],
        env={
            **os.environ,
            "PYTHONPATH": (
                f"{package_root}{os.pathsep}{os.environ['PYTHONPATH']}"
                if "PYTHONPATH" in os.environ
                else str(package_root)
            ),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "gepa.core.adapter.EvaluationBatch is unavailable" in completed.stdout
