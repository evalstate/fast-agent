from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fast_agent.session.atif_models import (
    AtifAgent,
    AtifMetrics,
    AtifObservation,
    AtifObservationResult,
    AtifStep,
    AtifSubagentTrajectoryRef,
    AtifToolCall,
    AtifTrajectory,
)


def _trajectory(*steps: AtifStep, trajectory_id: str = "traj_root") -> AtifTrajectory:
    return AtifTrajectory(
        session_id="run_1",
        trajectory_id=trajectory_id,
        agent=AtifAgent(name="fast-agent", version="test"),
        steps=list(steps),
    )


def test_atif_models_reject_unknown_standard_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AtifAgent.model_validate({"name": "fast-agent", "version": "test", "secret": True})


def test_atif_trajectory_requires_sequential_step_ids() -> None:
    with pytest.raises(ValidationError, match="step_id must be sequential"):
        _trajectory(AtifStep(step_id=2, source="user", message="hello"))


def test_atif_trajectory_rejects_unknown_observation_call_id() -> None:
    with pytest.raises(ValidationError, match="has no tool call"):
        _trajectory(
            AtifStep(
                step_id=1,
                source="agent",
                message="done",
                observation=AtifObservation(
                    results=[AtifObservationResult(source_call_id="missing", content="no")]
                ),
            )
        )


def test_atif_zero_llm_dispatch_forbids_metrics() -> None:
    with pytest.raises(ValidationError, match="must be absent"):
        AtifStep(
            step_id=1,
            source="agent",
            message="dispatch",
            llm_call_count=0,
            metrics=AtifMetrics(prompt_tokens=1),
        )


def test_atif_embedded_trajectories_require_unique_ids() -> None:
    child = _trajectory(
        AtifStep(step_id=1, source="user", message="child"),
        trajectory_id="traj_child",
    )
    with pytest.raises(ValidationError, match="must be unique"):
        AtifTrajectory(
            session_id="run_1",
            trajectory_id="traj_root",
            agent=AtifAgent(name="fast-agent", version="test"),
            steps=[AtifStep(step_id=1, source="user", message="root")],
            subagent_trajectories=[child, child.model_copy(deep=True)],
        )


def test_atif_subagent_reference_requires_resolution_key() -> None:
    with pytest.raises(ValidationError, match="require trajectory_id or trajectory_path"):
        AtifSubagentTrajectoryRef(session_id="run_1")


def test_atif_embedded_subagent_reference_must_resolve() -> None:
    with pytest.raises(ValidationError, match="must resolve"):
        _trajectory(
            AtifStep(
                step_id=1,
                source="agent",
                message="delegate",
                tool_calls=[
                    AtifToolCall(
                        tool_call_id="call_1",
                        function_name="worker",
                        arguments={},
                    )
                ],
                observation=AtifObservation(
                    results=[
                        AtifObservationResult(
                            source_call_id="call_1",
                            subagent_trajectory_ref=[
                                AtifSubagentTrajectoryRef(trajectory_id="missing")
                            ],
                        )
                    ]
                ),
            )
        )


def test_harbor_multiagent_golden_fixture_is_valid_atif_v17() -> None:
    fixture = (
        Path(__file__).parents[3]
        / "fixtures"
        / "atif"
        / "harbor_multiagent_v1_7.json"
    )
    trajectory = AtifTrajectory.model_validate_json(fixture.read_text(encoding="utf-8"))

    assert trajectory.schema_version == "ATIF-v1.7"
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_steps == len(trajectory.steps)
    assert trajectory.subagent_trajectories is not None
    assert trajectory.subagent_trajectories[0].trajectory_id == "traj_child_golden"
