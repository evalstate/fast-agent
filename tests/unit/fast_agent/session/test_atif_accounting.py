"""Wire contracts for observed accounting; no provider execution is required."""

import json

import pytest
from mcp_types import TextContent

from fast_agent.constants import FAST_AGENT_RETRY, FAST_AGENT_USAGE
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.session.trace_export_atif import (
    AtifRunSource,
    _final_metrics_from_usage_summary,
    build_atif_fanout_trajectory,
    build_atif_trajectory,
)


def _source(attempts: list[dict[str, object]], *, interrupted: bool = False) -> AtifRunSource:
    channels = {}
    if attempts:
        channels[FAST_AGENT_USAGE] = [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "schema": "fast-agent.usage/v2",
                        "provider_attempts": attempts,
                    }
                ),
            )
        ]
    if interrupted:
        channels[FAST_AGENT_RETRY] = [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "schema": "fast-agent.retry/v1",
                        "provider_attempts": len(attempts) + 1,
                        "retries": [],
                    }
                ),
            )
        ]
    return AtifRunSource(
        session_id="synthetic",
        agent_name="agent",
        model_name="model",
        provider="openai",
        history=[PromptMessageExtended(role="assistant", channels=channels)],
        message_timestamps=(None,),
    )


def _attempt(*, known: bool) -> dict[str, object]:
    return {
        "provider": "openai",
        "usage_schema": "openai-chat",
        "model": "model",
        "prompt": {"total": 10, "cache_read": 0} if known else {},
        "completion": {"total": 2} if known else {},
        "tool_calls": 0,
        "cost_usd": 0.01 if known else None,
    }


@pytest.mark.parametrize(
    "attempts,available",
    [
        ([], False),
        ([_attempt(known=False)], False),
        ([_attempt(known=True)], True),
    ],
)
@pytest.mark.parametrize("fanout", [False, True])
def test_absent_vs_observed_zero(
    attempts: list[dict[str, object]], available: bool, fanout: bool
) -> None:
    source = _source(attempts)
    trajectory = (
        build_atif_fanout_trajectory(session_id="synthetic", sources=[source])
        if fanout
        else build_atif_trajectory(source)
    )
    payload = trajectory.to_json_dict()
    extra = payload["final_metrics"]["extra"]
    assert extra["observed_cached_tokens_lower_bound"] == 0
    assert extra["accounting"] == {
        "schema": "fast-agent.accounting/v1",
        "scope": "observed",
        "provider_usage_complete": extra["llm_usage_calls_complete"],
        "observed_token_availability": {
            "prompt_tokens": available,
            "completion_tokens": available,
            "cached_tokens": available,
        },
    }


@pytest.mark.parametrize("interrupted", [False, True])
def test_partial_retry_fields_preserve_observations(interrupted: bool) -> None:
    # A successful response, an attempt with no token fields, and optionally
    # an interrupted retry without a usage report. Retry metadata is not usage.
    source = _source([_attempt(known=True), _attempt(known=False)], interrupted=interrupted)
    trajectory = build_atif_fanout_trajectory(session_id="synthetic", sources=[source])
    child = trajectory.subagent_trajectories
    assert child is not None
    step_metrics = child[0].steps[0].metrics
    assert step_metrics is not None
    assert step_metrics.prompt_tokens is None  # unchanged complete-data semantics
    assert step_metrics.extra is not None
    assert step_metrics.extra["observed_prompt_tokens_lower_bound"] == 10
    for item in (trajectory, child[0]):
        metrics = item.final_metrics
        assert metrics is not None and metrics.extra is not None
        assert metrics.total_prompt_tokens is None
        assert metrics.total_completion_tokens is None
        assert metrics.total_cached_tokens is None
        assert metrics.total_cost_usd is None
        extra = metrics.extra
        assert extra["observed_prompt_tokens_lower_bound"] == 10
        assert extra["observed_completion_tokens_lower_bound"] == 2
        assert extra["observed_cached_tokens_lower_bound"] == 0
        assert extra["observed_cost_usd_lower_bound"] == pytest.approx(0.01)
        assert extra["llm_usage_expected_call_count"] == (3 if interrupted else 2)
        assert extra["llm_usage_observed_call_count"] == 2
        assert extra["accounting"]["provider_usage_complete"] is (not interrupted)
        assert all(extra["accounting"]["observed_token_availability"].values())


def test_child_complete_data_summary_does_not_erase_step_observations() -> None:
    trajectory = build_atif_trajectory(_source([_attempt(known=True), _attempt(known=False)]))
    metrics = _final_metrics_from_usage_summary(
        {"prompt": {}, "completion": {}, "provider_attempts": 2, "tool_calls": 0},
        total_steps=len(trajectory.steps),
        steps=trajectory.steps,
    )
    assert metrics.total_prompt_tokens is None
    assert metrics.extra is not None
    assert metrics.extra["observed_prompt_tokens_lower_bound"] == 10
    assert metrics.extra["observed_completion_tokens_lower_bound"] == 2
    assert metrics.extra["accounting"]["observed_token_availability"]["cached_tokens"] is True


def test_availability_is_per_field_not_call_or_field_completeness() -> None:
    attempt = _attempt(known=False)
    attempt["completion"] = {"total": 0}
    trajectory = build_atif_trajectory(_source([attempt], interrupted=True))
    metrics = trajectory.final_metrics
    assert metrics is not None and metrics.extra is not None
    assert metrics.extra["accounting"] == {
        "schema": "fast-agent.accounting/v1",
        "scope": "observed",
        "provider_usage_complete": False,
        "observed_token_availability": {
            "prompt_tokens": False,
            "completion_tokens": True,
            "cached_tokens": False,
        },
    }
    assert metrics.extra["observed_completion_tokens_lower_bound"] == 0
    assert metrics.total_completion_tokens is None
