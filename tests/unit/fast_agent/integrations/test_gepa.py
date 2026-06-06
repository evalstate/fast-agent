import subprocess

from fast_agent.batch import BatchRunResult
from fast_agent.integrations.gepa import FastAgentBatchEvaluator, FastAgentReflectionLM


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
