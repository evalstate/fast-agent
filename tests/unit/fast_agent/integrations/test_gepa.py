import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fast_agent.batch import BatchRunResult
from fast_agent.integrations.gepa import (
    FastAgentBatchEvaluator,
    FastAgentReflectionLM,
    FastAgentRowWiseBatchAdapter,
    RowWiseEvaluationRun,
    RowWiseScore,
    _evaluation_batch,
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
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
            summary={"processed_rows": 2},
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
    assert (tmp_path / "runs" / "row-wise-evals" / "eval-00001" / "row-wise-score.json").exists()

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
