import json
import sys

from fast_agent.eval import ArtifactRun, EvalRun


def test_artifact_run_materializes_candidate_command_and_score(tmp_path):
    artifact_run = ArtifactRun(tmp_path / "runs")
    assert isinstance(artifact_run, EvalRun)
    candidate = artifact_run.candidate()

    candidate.materialize_candidate(
        {"skill": "seed"},
        variables={"policy": "concise"},
    )
    result = candidate.run_command(
        [sys.executable, "-c", "print('ok')"],
        timeout_seconds=5,
        log_prefix="generate",
    )
    score = candidate.write_score(0.75, "useful feedback", metadata={"lines": 12})

    assert result.ok is True
    assert result.stdout_path.read_text(encoding="utf-8").strip() == "ok"
    assert score.score == 0.75
    assert json.loads((candidate.path / "candidate.json").read_text(encoding="utf-8")) == {
        "skill": "seed"
    }
    assert (candidate.path / "reports").is_dir()
    assert (candidate.path / "score.json").exists()
