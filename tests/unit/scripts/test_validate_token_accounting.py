from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

import pytest
from mcp_types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    TextContent,
)

from fast_agent.constants import FAST_AGENT_USAGE
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import save_json
from fast_agent.session import (
    SessionAgentSnapshot,
    SessionContinuationSnapshot,
    SessionSnapshot,
    SessionTraceExporter,
)
from fast_agent.session.session_manager import SessionManager
from fast_agent.session.token_accounting_validation import (
    ArtifactValidationError,
    validate_artifacts,
)
from fast_agent.session.trace_export_models import ExportRequest

if TYPE_CHECKING:
    from pathlib import Path


def _usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    reasoning_tokens: int,
    cumulative_input: int,
    cumulative_output: int,
    cumulative_cached: int,
    cumulative_reasoning: int,
) -> TextContent:
    return TextContent(
        type="text",
        text=json.dumps(
            {
                "schema": "fast-agent.usage/v2",
                "provider_attempts": [
                    {
                        "provider": "openai",
                        "usage_schema": "openai-chat",
                        "model": "gpt-5.6",
                        "prompt": {
                            "total": input_tokens,
                            "cache_read": cached_tokens,
                            "cache_write": 0,
                        },
                        "completion": {
                            "total": output_tokens,
                            "reasoning": reasoning_tokens,
                        },
                        "tool_calls": 0,
                        "raw_usage": {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "prompt_tokens_details": {
                                "cached_tokens": cached_tokens,
                                "cache_write_tokens": 0,
                            },
                            "completion_tokens_details": {
                                "reasoning_tokens": reasoning_tokens,
                            },
                        },
                    },
                ],
            }
        ),
    )


def _write_session(tmp_path: Path) -> tuple[SessionManager, Path]:
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session_id = "2607151200-accounting"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    call_id = "call-1"
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="run the tool")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="running")],
            tool_calls={
                call_id: CallToolRequest(
                    params=CallToolRequestParams(
                        name="execute",
                        arguments={"command": "printf ok"},
                    )
                )
            },
            channels={
                FAST_AGENT_USAGE: [
                    _usage(
                        input_tokens=100,
                        output_tokens=20,
                        cached_tokens=10,
                        reasoning_tokens=5,
                        cumulative_input=100,
                        cumulative_output=20,
                        cumulative_cached=10,
                        cumulative_reasoning=5,
                    )
                ]
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                call_id: CallToolResult(
                    content=[TextContent(type="text", text="ok")],
                    is_error=False,
                )
            },
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="done")],
            channels={
                FAST_AGENT_USAGE: [
                    _usage(
                        input_tokens=150,
                        output_tokens=10,
                        cached_tokens=20,
                        reasoning_tokens=0,
                        cumulative_input=250,
                        cumulative_output=30,
                        cumulative_cached=30,
                        cumulative_reasoning=5,
                    )
                ]
            },
        ),
    ]
    save_json(messages, str(session_dir / "history_agent.json"))
    snapshot = SessionSnapshot(
        session_id=session_id,
        created_at=datetime(2026, 7, 15, 12, 0),
        last_activity=datetime(2026, 7, 15, 12, 1),
        continuation=SessionContinuationSnapshot(
            active_agent="agent",
            agents={
                "agent": SessionAgentSnapshot(
                    history_file="history_agent.json",
                    model="gpt-5.6",
                    model_spec="codexresponses.gpt-5.6?reasoning=low",
                    provider="openai",
                )
            },
        ),
    )
    (session_dir / "session.json").write_text(
        json.dumps(snapshot.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return manager, session_dir


def _export(
    manager: SessionManager,
    session_dir: Path,
    path: Path,
    export_format: str,
) -> None:
    SessionTraceExporter(session_manager=manager).export(
        ExportRequest(
            target=session_dir,
            agent_name="agent",
            output_path=path,
            format=export_format,
        )
    )


def test_validate_token_accounting_artifacts(tmp_path: Path) -> None:
    manager, session_dir = _write_session(tmp_path)
    codex_path = tmp_path / "trace.jsonl"
    atif_path = tmp_path / "trace.json"
    _export(manager, session_dir, codex_path, "codex")
    _export(manager, session_dir, atif_path, "atif")

    report = validate_artifacts(
        session_dir=session_dir,
        codex_path=codex_path,
        atif_path=atif_path,
        require_cache=True,
        require_tool=True,
    )

    assert report.session.usage_records == 2
    assert report.codex.token_records == 2
    assert report.codex.tool_calls == 1
    assert report.atif.metric_steps == 2
    assert report.atif.cached_tokens == 30


def test_validate_token_accounting_rejects_incorrect_codex_total(tmp_path: Path) -> None:
    manager, session_dir = _write_session(tmp_path)
    codex_path = tmp_path / "trace.jsonl"
    atif_path = tmp_path / "trace.json"
    _export(manager, session_dir, codex_path, "codex")
    _export(manager, session_dir, atif_path, "atif")

    records = [json.loads(line) for line in codex_path.read_text(encoding="utf-8").splitlines()]
    token_record = next(
        record
        for record in records
        if record["type"] == "event_msg" and record["payload"]["type"] == "token_count"
    )
    token_record["payload"]["info"]["last_token_usage"]["total_tokens"] = 999
    codex_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="total_tokens"):
        validate_artifacts(
            session_dir=session_dir,
            codex_path=codex_path,
            atif_path=atif_path,
        )


def _metric_trajectory(
    *, root_prompt: int, child_prompt: int | None, total_prompt: int | None
) -> dict[str, object]:
    def step(prompt: int | None) -> dict[str, object]:
        return {
            "step_id": 1,
            "source": "agent",
            "message": "ok",
            "metrics": {"prompt_tokens": prompt, "completion_tokens": 1, "cached_tokens": 0},
            "llm_call_count": 1,
        }

    def final(prompt: int | None, completion: int | None) -> dict[str, object]:
        return {
            "total_prompt_tokens": prompt,
            "total_completion_tokens": completion,
            "total_cached_tokens": 0 if prompt is not None else None,
            "total_steps": 1,
        }

    child = {
        "trajectory_id": "compaction-step-1",
        "agent": {"name": "fast-agent", "version": "test"},
        "steps": [step(child_prompt)],
        "final_metrics": final(child_prompt, 1 if child_prompt is not None else None),
    }
    return {
        "agent": {"name": "fast-agent", "version": "test"},
        "steps": [step(root_prompt)],
        "final_metrics": final(total_prompt, 2 if total_prompt is not None else None),
        "subagent_trajectories": [child],
    }


@pytest.mark.parametrize(
    ("child_prompt", "total_prompt", "valid"),
    [
        # Embedded summary/compaction calls count toward final metrics.
        (40, 140, True),
        (40, 100, False),
        # An incomplete child makes the total incomplete; claiming a sum is invalid.
        (None, None, True),
        (None, 100, False),
    ],
)
def test_validate_atif_counts_embedded_subagent_metrics(
    tmp_path: Path, child_prompt: int | None, total_prompt: int | None, valid: bool
) -> None:
    from fast_agent.session.token_accounting_validation import validate_atif

    path = tmp_path / "trajectory.json"
    path.write_text(
        json.dumps(
            _metric_trajectory(
                root_prompt=100, child_prompt=child_prompt, total_prompt=total_prompt
            )
        ),
        encoding="utf-8",
    )

    if valid:
        validate_atif(path)
    else:
        with pytest.raises(ArtifactValidationError):
            validate_atif(path)


@pytest.mark.parametrize("provider", ["anthropic", "copilot"])
def test_raw_anthropic_signature_includes_compaction_iterations(provider: str) -> None:
    """Canonical totals fold billed compaction iterations; the raw check must agree."""
    from anthropic.types.beta import BetaUsage

    from fast_agent.llm.provider_types import Provider
    from fast_agent.llm.usage_tracking import usage_from_anthropic
    from fast_agent.session.token_accounting_validation import _raw_usage_signature

    iteration = {"cache_read_input_tokens": 40, "cache_creation_input_tokens": 0}
    usage = BetaUsage.model_validate(
        {
            "input_tokens": 300,
            "cache_read_input_tokens": 1_000,
            "cache_creation_input_tokens": 50,
            "output_tokens": 80,
            "iterations": [
                {**iteration, "type": "compaction", "input_tokens": 120_000, "output_tokens": 900},
                {**iteration, "type": "message", "input_tokens": 300, "output_tokens": 80},
            ],
        }
    )
    turn = usage_from_anthropic(usage, provider=Provider(provider), model="claude-sonnet-5.5")

    assert _raw_usage_signature(turn.raw_usage, provider=provider, label="usage") == (
        turn.prompt.total,
        turn.completion.total,
        turn.prompt.cache_read,
        turn.completion.reasoning,
    )
    assert turn.prompt.total == 300 + 1_000 + 50 + 120_000 + 40
