import json
import os
import subprocess
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fast_agent.batch import BatchRunner, BatchRunResult
from fast_agent.integrations.gepa import (
    FastAgentBatchEvaluator,
    FastAgentGEPATrackioCallback,
    FastAgentReflectionLM,
    FastAgentRowWiseBatchAdapter,
    FastAgentSingleTaskAdapter,
    GEPATrackioDashboard,
    RowWiseEvaluationRun,
    RowWiseScore,
    _budget_event_metrics,
    _evaluation_batch,
    _evaluation_event_metrics,
    gepa_api_trackio_kwargs,
    gepa_numeric_metrics,
    gepa_trackio_init_kwargs,
    make_gepa_trackio_dashboard,
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
        home=tmp_path / "env",
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
    assert "--home" in commands[0][0]
    assert "--home-dir" not in commands[0][0]


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
        home=tmp_path / "env",
        model="passthrough",
        audit_dir=tmp_path / "audit",
        command_runner=fake_runner,
    )

    assert lm("think about this") == "reflection"
    assert not logged
    FastAgentGEPATrackioCallback(reflection_lm=lm, include_gepa_context=True).on_proposal_end(
        {
            "iteration": 3,
            "new_instructions": {"policy": "new"},
            "prompts": {"policy": "prompt"},
            "raw_lm_outputs": {"policy": "raw"},
        }
    )
    assert logged
    assert logged[0]["gepa/iteration"] == 3
    assert "gepa/total_metric_calls" not in logged[0]
    assert logged[0]["fast_agent/gepa_context/proposed_components"] == 1
    assert logged[0]["fast_agent/reflection/usage/cumulative_billing_tokens"] == 12
    assert logged[0]["fast_agent/reflection/usage/billing_tokens_per_turn"] == 12
    assert logged[0]["fast_agent/reflection/usage/input_tokens_per_turn"] == 10
    assert logged[0]["fast_agent/reflection/usage/cache_hit_rate_percent"] == 40.0
    assert "fast_agent/reflection/call_index" not in logged[0]
    assert "fast_agent/reflection/usage/cumulative_input_tokens" not in logged[0]


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
        home=tmp_path / "env",
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

    def runner_factory(home, *, backend):
        assert home == tmp_path / "env"
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
        home=tmp_path / "env",
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


def test_single_task_adapter_evaluates_batch_of_one_and_exposes_metrics(tmp_path):
    class Runner(BatchRunner):
        async def run(self, **kwargs: Any) -> BatchRunResult:
            output_path = Path(kwargs["output_path"])
            output_path.write_text(
                json.dumps({"ok": True, "result": "Once upon a moon."}) + "\n",
                encoding="utf-8",
            )
            return BatchRunResult(
                rows=[{"ok": True, "result": "Once upon a moon."}],
                output_path=output_path,
                summary={
                    "processed_rows": 1,
                    "duration_ms": 500,
                    "timing_ms": {
                        "duration": {"count": 1, "mean": 400},
                        "ttft": {"count": 1, "mean": 100},
                    },
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "billing_tokens": 15,
                        "rows_with_usage": 1,
                    },
                },
                telemetry_path=kwargs["telemetry_path"],
                error_output_path=None,
                summary_path=kwargs["summary_path"],
            )

    def runner_factory(home, *, backend):
        return Runner()

    adapter = FastAgentSingleTaskAdapter(
        home=tmp_path / "env",
        agent_card=tmp_path / "card.md",
        model="test-model",
        input_builder=lambda candidate, example=None: candidate["prompt"],
        scorer=lambda output_row, candidate, evaluation, example=None: RowWiseScore(
            score=0.75,
            trajectory={"response": output_row["result"]},
            objective_scores={"gepa_score": 0.75},
        ),
        run_dir=tmp_path / "runs",
        batch_runner_factory=runner_factory,
    )

    score, side_info = adapter({"prompt": "Write a story."})

    assert score == 0.75
    assert side_info == {"response": "Once upon a moon."}
    metrics = adapter.pop_pending_gepa_eval_metrics()
    assert metrics is not None
    assert metrics["fast_agent/eval/batch_size"] == 1
    assert metrics["fast_agent/eval/duration_seconds"] == 0.5
    assert metrics["fast_agent/eval/duration_seconds_per_row"] == 0.5
    assert metrics["fast_agent/eval/rows_per_second"] == 2
    assert metrics["fast_agent/eval/ttft_mean_seconds"] == 0.1
    assert metrics["fast_agent/eval/usage/input_tokens_per_row"] == 10
    assert metrics["fast_agent/eval/usage/output_tokens_per_second"] == 5 / 0.3
    assert "fast_agent/eval/usage/output_tokens_per_generation_second" not in metrics
    assert "fast_agent/eval/objective_avg/gepa_score" not in metrics


def test_single_task_prompt_adapter_uses_default_worker(tmp_path):
    calls = []

    class Runner(BatchRunner):
        async def run(self, **kwargs: Any) -> BatchRunResult:
            calls.append(kwargs)
            output_path = Path(kwargs["output_path"])
            output_path.write_text(json.dumps({"result": "ok"}) + "\n", encoding="utf-8")
            return BatchRunResult(
                rows=[{"result": "ok"}],
                output_path=output_path,
                summary={"processed_rows": 1},
                telemetry_path=kwargs["telemetry_path"],
                error_output_path=None,
                summary_path=kwargs["summary_path"],
            )

    adapter = FastAgentSingleTaskAdapter.prompt(
        model="test-model",
        scorer=lambda output_row, candidate, evaluation, example=None: 1.0,
        run_dir=tmp_path / "runs",
        batch_runner_factory=lambda home, *, backend: Runner(),
    )

    assert adapter({"prompt": "hello"}) == (1.0, {})
    assert calls[0]["agent_card"] is None
    assert calls[0]["agent"] is None
    assert calls[0]["model"] == "test-model"
    assert calls[0]["template"] == "{{prompt}}"


def test_row_wise_batch_adapter_logs_batch_usage_and_cache(monkeypatch, tmp_path):
    logged: list[dict[str, Any]] = []
    runner = RecordingBatchRunner(
        summary={
            "processed_rows": 2,
            "failed_rows": 0,
            "skipped_rows": 0,
            "selected_rows": 2,
            "duration_ms": 1000,
            "timing_ms": {
                "duration": {"count": 2, "mean": 400},
                "ttft": {"count": 2, "mean": 100},
            },
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "billing_tokens": 120,
                "reasoning_tokens": 7,
                "tool_use_tokens": 3,
                "tool_calls": 1,
                "rows_with_usage": 2,
                "usage_coverage_percent": 100.0,
            },
            "cache": {
                "served_tokens": 60,
                "activity_tokens": 80,
                "hit_tokens": 40,
                "write_tokens": 20,
                "effective_input_tokens": 40,
                "hit_rate_percent": 60.0,
                "rows_with_cache_activity": 2,
                "row_cache_activity_percent": 100.0,
            },
        }
    )

    def runner_factory(home, *, backend):
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
        home=tmp_path / "env",
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
    FastAgentGEPATrackioCallback(eval_adapter=adapter, include_gepa_context=True).on_evaluation_end(
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
            "metric_calls_before": 10,
            "metric_calls_delta": 2,
            "metric_calls_after": 12,
        }
    )
    assert logged
    payload = logged[0]
    assert payload["gepa/iteration"] == 7
    assert payload["gepa/total_metric_calls"] == 12
    assert payload["gepa/metric_calls_delta"] == 2
    assert payload["fast_agent/eval/gepa_iteration"] == 7
    assert payload["fast_agent/eval/gepa_candidate_idx"] == 4
    assert payload["fast_agent/gepa_context/parent_count"] == 1
    assert payload["fast_agent/gepa_context/score_mean"] == 0.5
    assert payload["fast_agent/eval/step"] == 1
    assert payload["fast_agent/eval/batch_size"] == 2
    assert payload["fast_agent/eval/objective_avg/gepa_score"] == 0.5
    assert payload["fast_agent/eval/duration_seconds"] == 1
    assert payload["fast_agent/eval/duration_seconds_per_row"] == 0.5
    assert payload["fast_agent/eval/rows_per_second"] == 2
    assert payload["fast_agent/eval/ttft_mean_seconds"] == 0.1
    assert payload["fast_agent/eval/failed_rows"] == 0
    assert payload["fast_agent/eval/error_rate_percent"] == 0
    assert payload["fast_agent/eval/usage/billing_tokens_per_row"] == 60
    assert payload["fast_agent/eval/usage/input_tokens_per_row"] == 50
    assert payload["fast_agent/eval/usage/output_tokens_per_second"] == 20 / 0.3
    assert "fast_agent/eval/usage/output_tokens_per_generation_second" not in payload
    assert payload["fast_agent/eval/usage/reasoning_tokens_per_row"] == 3.5
    assert payload["fast_agent/eval/usage/tool_use_tokens_per_row"] == 1.5
    assert payload["fast_agent/eval/usage/tool_calls_per_row"] == 0.5
    assert payload["fast_agent/eval/cache/row_cache_activity_percent"] == 100.0
    assert payload["fast_agent/eval/cache/hit_rate_percent"] == 60.0
    assert "fast_agent/eval/eval_index" not in payload
    assert "fast_agent/eval/num_metric_calls" not in payload
    assert "fast_agent/eval/duration_ms" not in payload
    assert "fast_agent/eval/processed_rows" not in payload
    assert "fast_agent/eval/selected_rows" not in payload
    assert "fast_agent/eval/usage/input_tokens" not in payload
    assert "fast_agent/eval/usage/usage_coverage_percent" not in payload
    assert "fast_agent/eval/usage/total_tokens_per_row" not in payload
    assert "fast_agent/eval/cache/served_tokens" not in payload
    assert "fast_agent/eval/cache/write_rate_percent" not in payload


def test_trackio_callback_omits_score_summaries_for_single_row_batches(monkeypatch, tmp_path):
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

    def runner_factory(home, *, backend):
        return runner

    def row_scorer(output_row, input_row, candidate, evaluation):
        return RowWiseScore(score=1.0, objective_scores={"gepa_score": 1.0})

    monkeypatch.setattr(
        "fast_agent.integrations.gepa.safe_trackio_log",
        lambda payload, **kwargs: logged.append(dict(payload)) or True,
    )

    adapter = FastAgentRowWiseBatchAdapter(
        home=tmp_path / "env",
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
        eval_adapter=adapter,
        include_gepa_context=False,
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
            "metric_calls_before": 10,
            "metric_calls_delta": 1,
            "metric_calls_after": 11,
        }
    )

    assert logged
    payload = logged[0]
    assert payload["gepa/iteration"] == 7
    assert payload["gepa/total_metric_calls"] == 11
    assert payload["fast_agent/eval/gepa_iteration"] == 7
    assert payload["fast_agent/eval/gepa_candidate_idx"] == 4
    assert "fast_agent/gepa_context/score_mean" not in payload
    assert "fast_agent/gepa_context/batch_size" not in payload
    assert payload["fast_agent/eval/step"] == 1
    assert payload["fast_agent/eval/batch_size"] == 1
    assert "fast_agent/eval/avg_score" not in payload
    assert "fast_agent/eval/num_metric_calls" not in payload
    assert "fast_agent/eval/objective_avg/gepa_score" not in payload
    assert payload["fast_agent/eval/error_rate_percent"] == 0
    assert payload["fast_agent/eval/usage/billing_tokens_per_row"] == 60
    assert payload["fast_agent/eval/cache/hit_rate_percent"] == 60.0


def test_eval_metrics_include_error_rate_for_failed_rows(monkeypatch, tmp_path):
    logged: list[dict[str, Any]] = []
    runner = RecordingBatchRunner(
        summary={
            "processed_rows": 4,
            "failed_rows": 1,
            "skipped_rows": 0,
            "duration_ms": 1000,
        }
    )

    def runner_factory(home, *, backend):
        return runner

    def row_scorer(output_row, input_row, candidate, evaluation):
        return RowWiseScore(score=1.0)

    monkeypatch.setattr(
        "fast_agent.integrations.gepa.safe_trackio_log",
        lambda payload, **kwargs: logged.append(dict(payload)) or True,
    )

    adapter = FastAgentRowWiseBatchAdapter(
        home=tmp_path / "env",
        agent_card=tmp_path / "card.md",
        candidate_variables={"policy": "policy"},
        template="{{row_json}}",
        row_scorer=row_scorer,
        run_dir=tmp_path / "runs",
        batch_runner_factory=runner_factory,
    )
    adapter.evaluate([{}, {}, {}, {}], {"policy": "route carefully"})

    FastAgentGEPATrackioCallback(eval_adapter=adapter).on_evaluation_end(
        {"iteration": 1, "metric_calls_before": 4, "metric_calls_delta": 4, "metric_calls_after": 8}
    )

    assert logged[0]["fast_agent/eval/step"] == 1
    assert logged[0]["gepa/iteration"] == 1
    assert logged[0]["gepa/total_metric_calls"] == 8
    assert logged[0]["fast_agent/eval/failed_rows"] == 1
    assert logged[0]["fast_agent/eval/error_rate_percent"] == 25


def test_trackio_callback_adds_monotonic_fast_agent_eval_step(monkeypatch):
    logged: list[dict[str, Any]] = []

    class Adapter:
        def __init__(self) -> None:
            self.pending = [
                {"fast_agent/eval/batch_size": 1},
                {"fast_agent/eval/batch_size": 1},
            ]

        def pop_pending_gepa_eval_metrics(self):
            return self.pending.pop(0) if self.pending else None

    monkeypatch.setattr(
        "fast_agent.integrations.gepa.safe_trackio_log",
        lambda payload, **kwargs: logged.append(dict(payload)) or True,
    )

    callback = FastAgentGEPATrackioCallback(eval_adapter=Adapter())
    callback.on_valset_evaluated(
        {"iteration": 0, "candidate_idx": 0, "metric_calls_before": 0, "metric_calls_delta": 1, "metric_calls_after": 1}
    )
    callback.on_evaluation_end({"iteration": 1, "candidate_idx": 1})
    callback.on_budget_updated(
        {"iteration": 1, "metric_calls_used": 2, "metric_calls_delta": 1, "metric_calls_remaining": None}
    )

    eval_rows = [row for row in logged if "fast_agent/eval/step" in row]
    assert [row["fast_agent/eval/step"] for row in eval_rows] == [1, 2]
    assert [row["fast_agent/eval/gepa_iteration"] for row in eval_rows] == [0, 1]
    assert [row["fast_agent/eval/gepa_candidate_idx"] for row in eval_rows] == [0, 1]
    assert [row["gepa/iteration"] for row in eval_rows] == [0, 1]
    assert [row["gepa/total_metric_calls"] for row in eval_rows] == [1, 2]


def test_trackio_callback_omits_trackio_global_step(monkeypatch):
    calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    class Adapter:
        def pop_pending_gepa_eval_metrics(self):
            return {"fast_agent/eval/ttft_mean_seconds": 0.1}

    class ReflectionLM(FastAgentReflectionLM):
        def __init__(self) -> None:
            pass

        def pop_pending_gepa_reflection_metrics(self):
            return [{"fast_agent/reflection/duration_seconds": 1.2}]

    monkeypatch.setattr(
        "fast_agent.integrations.gepa.safe_trackio_log",
        lambda payload, **kwargs: calls.append((dict(payload), dict(kwargs))) or True,
    )

    callback = FastAgentGEPATrackioCallback(eval_adapter=Adapter(), reflection_lm=ReflectionLM())
    callback.on_budget_updated({"iteration": 1, "metric_calls_used": 2, "metric_calls_delta": 1})
    callback.on_evaluation_end({"iteration": 2, "metric_calls_after": 3})
    callback.on_proposal_end({"iteration": 2})

    assert [kwargs for _, kwargs in calls] == [{}, {}, {}]
    assert [payload["gepa/iteration"] for payload, _ in calls] == [1, 2, 2]


def test_evaluation_event_metrics_maps_authoritative_gepa_axes():
    payload = _evaluation_event_metrics(
        {
            "iteration": 7,
            "candidate_idx": 4,
            "total_metric_calls": 123,
            "metric_calls_after": 124,
            "metric_calls_delta": 2,
            "num_metric_calls": 2,
        }
    )

    assert payload["gepa/iteration"] == 7
    assert payload["gepa/total_metric_calls"] == 124
    assert payload["gepa/metric_calls_delta"] == 2
    assert payload["fast_agent/eval/gepa_iteration"] == 7
    assert payload["fast_agent/eval/gepa_candidate_idx"] == 4
    assert "gepa/num_metric_calls" not in payload


def test_budget_event_metrics_maps_authoritative_gepa_budget_fields():
    payload = _budget_event_metrics(
        {
            "iteration": 7,
            "metric_calls_used": 123,
            "metric_calls_delta": 2,
            "metric_calls_remaining": 9,
        }
    )

    assert payload == {
        "gepa/iteration": 7,
        "gepa/total_metric_calls": 123,
        "gepa/metric_calls_delta": 2,
        "gepa/metric_calls_remaining": 9,
    }


def test_trackio_callback_logs_budget_updates(monkeypatch):
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "fast_agent.integrations.gepa.safe_trackio_log",
        lambda payload, **kwargs: logged.append(dict(payload)) or True,
    )

    FastAgentGEPATrackioCallback().on_budget_updated(
        {
            "iteration": 3,
            "metric_calls_used": 12,
            "metric_calls_delta": 4,
            "metric_calls_remaining": None,
        }
    )

    assert logged == [
        {
            "gepa/iteration": 3,
            "gepa/total_metric_calls": 12,
            "gepa/metric_calls_delta": 4,
        }
    ]


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


def test_make_gepa_trackio_dashboard_builds_tracking_and_callbacks(monkeypatch, tmp_path):
    optimize_anything = types.ModuleType("gepa.optimize_anything")

    class TrackingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    optimize_anything_any: Any = optimize_anything
    optimize_anything_any.TrackingConfig = TrackingConfig
    gepa = types.ModuleType("gepa")
    gepa.__path__ = []
    monkeypatch.setitem(sys.modules, "gepa", gepa)
    monkeypatch.setitem(sys.modules, "gepa.optimize_anything", optimize_anything)

    def fake_command_runner(
        command: Sequence[str],
        cwd: Path | None,
        timeout_seconds: float | None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    lm = FastAgentReflectionLM(
        model="passthrough",
        audit_dir=tmp_path / "reflection",
        command_runner=fake_command_runner,
    )
    dashboard = make_gepa_trackio_dashboard(
        project="demo",
        name="run",
        group="group",
        config={"regime": "story"},
        reflection_lm=lm,
    )

    assert isinstance(dashboard, GEPATrackioDashboard)
    assert dashboard.tracking.kwargs == {
        "use_trackio": True,
        "trackio_init_kwargs": {
            "project": "demo",
            "name": "run",
            "group": "group",
            "embed": False,
            "auto_log_gpu": False,
            "config": {"regime": "story"},
        },
        "trackio_attach_existing": False,
        "trackio_step_metric": "gepa/iteration",
        "key_prefix": "gepa/",
    }
    assert len(dashboard.callbacks) == 1
    assert isinstance(dashboard.callbacks[0], FastAgentGEPATrackioCallback)
    assert dashboard.callbacks[0].include_gepa_context is False


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
