"""Synthetic audit coverage: no provider calls or historical artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.constants import FAST_AGENT_COMPACTION_CHANNEL, FAST_AGENT_USAGE
from fast_agent.history.atif_reconstruction import (
    COMPACTION_BOUNDARY,
    HistoryReconstructionError,
    is_export_marker,
    reconcile_transient,
    reconstruct_history,
)
from fast_agent.history.compaction import build_summary_message
from fast_agent.mcp.prompt_serialization import save_messages
from fast_agent.session.trace_export_atif import (
    AtifRunSource,
    build_atif_trajectory,
    live_export_history,
)
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.session.atif_models import AtifStep

BASE = datetime(2026, 9, 28, 12, tzinfo=timezone.utc)


def _message(index: int, *, template: bool = False) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text=f"message {index}")],
        timestamp=BASE + timedelta(seconds=index),
        is_template=template,
    )


def _exchange(index: int) -> list[PromptMessageExtended]:
    call_id = f"call-{index}"
    return [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text=f"request {index}")],
            timestamp=BASE + timedelta(seconds=index),
            tool_calls={
                call_id: CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="shell", arguments={"command": f"echo {index}"}
                    ),
                )
            },
        ),
        PromptMessageExtended(
            role="user",
            content=[],
            timestamp=BASE + timedelta(seconds=index, microseconds=1),
            tool_results={
                call_id: CallToolResult(
                    content=[TextContent(type="text", text=f"output {index}\n")],
                    is_error=False,
                )
            },
        ),
    ]


def _checkpoint(
    directory: Path,
    history: list[PromptMessageExtended],
    *,
    templates: int = 0,
    tail: int = 1,
    serial: int = 1,
    linked: bool = True,
) -> list[PromptMessageExtended]:
    stamp = BASE + timedelta(minutes=serial)
    path = directory / f"compacted_{stamp:%Y%m%d-%H%M%S}_agent.json"
    save_messages(history, str(path))
    summary = build_summary_message(
        f"summary {serial}",
        prompt_text="summarize",
        instructions=None,
        messages_compacted=len(history) - templates - tail,
        tokens_before=100,
        context_window=200,
        model="test",
        archive_metadata=(
            {
                "archive_file": path.name,
                "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "template_messages": templates,
                "retained_messages": tail,
            }
            if linked
            else None
        ),
    )
    assert summary.channels is not None
    block = summary.channels[FAST_AGENT_COMPACTION_CHANNEL][0]
    assert isinstance(block, TextContent)
    metadata = json.loads(block.text)
    metadata["compacted_at"] = stamp.isoformat()
    block.text = json.dumps(metadata)
    return history[:templates] + [summary] + (history[-tail:] if tail else [])


def _originals(history: list[PromptMessageExtended]) -> list[PromptMessageExtended]:
    return [m for m in history if not is_export_marker(m)]


def _trajectory(history: list[PromptMessageExtended], directory: Path):
    return build_atif_trajectory(
        AtifRunSource(
            session_id="test",
            agent_name="agent",
            model_name="test",
            provider="test",
            history=history,
            message_timestamps=tuple(m.timestamp for m in history),
            parent_session_dir=directory,
        )
    )


@pytest.mark.parametrize("linked", [True, False])
def test_exact_messages_and_results_across_repeated_compaction(tmp_path: Path, linked: bool):
    template = _message(0, template=True)
    repeated = _message(1)
    # Identical legitimate messages must survive; there is no global deduplication.
    first = [template, repeated, repeated.model_copy(deep=True), *_exchange(2), _message(3)]
    current = _checkpoint(tmp_path, first, templates=1, linked=linked)
    second = [*current, *_exchange(4), _message(5)]
    current = _checkpoint(tmp_path, second, templates=1, serial=2, linked=linked)
    final = [*current, *_exchange(6)]
    restored = reconstruct_history(final, tmp_path, "agent")
    assert _originals(restored) == first + _exchange(4) + [_message(5)] + _exchange(6)
    assert sum(COMPACTION_BOUNDARY in (m.channels or {}) for m in restored) == 2
    # Pure, repeatable recovery; original snapshots and final state stay unchanged.
    assert reconstruct_history(final, tmp_path, "agent") == restored
    trajectory = _trajectory(final, tmp_path)
    originals = [
        step for step in trajectory.steps if step.source != "system" and not step.is_copied_context
    ]
    calls = [call for step in originals for call in step.tool_calls or []]
    assert [call.tool_call_id for call in calls] == ["call-2", "call-4", "call-6"]
    outputs = [
        result.content
        for step in originals
        if step.observation
        for result in step.observation.results
    ]
    assert outputs == ["output 2\n", "output 4\n", "output 6\n"]
    boundaries = [step for step in trajectory.steps if (step.extra or {}).get("context_management")]
    assert len(boundaries) == 2
    assert all(step.source == "system" and step.metrics is None for step in boundaries)
    assert all(step.llm_call_count is None for step in boundaries)
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.extra is not None
    assert trajectory.final_metrics.extra["llm_usage_expected_call_count"] == 3
    assert trajectory.final_metrics.extra["summary_compaction_usage_complete"] is False


def test_linked_summary_only_and_extra_transient_evidence(tmp_path: Path):
    archived = [_message(0), *_exchange(1)]
    current = _checkpoint(tmp_path, archived, tail=0)
    transient = [*archived, *_exchange(2)]
    restored = live_export_history(current, transient, tmp_path, "agent")
    assert _originals(restored) == transient
    assert COMPACTION_BOUNDARY in (restored[-1].channels or {})
    later = _message(120)
    restored_later = live_export_history(current, archived + [later], tmp_path, "agent")
    assert restored_later[-1] == later
    assert COMPACTION_BOUNDARY in (restored_later[-2].channels or {})
    assert live_export_history(current, archived, tmp_path, "agent") == reconstruct_history(
        current, tmp_path, "agent"
    )


def test_transient_tail_extends_current_without_repeating_persisted_results(tmp_path: Path):
    archived = [_message(0), *_exchange(1), _message(2)]
    current = _checkpoint(tmp_path, archived) + _exchange(3)
    transient = [_message(2), *_exchange(3), *_exchange(4)]
    restored = live_export_history(current, transient, tmp_path, "agent")
    assert _originals(restored) == archived + _exchange(3) + _exchange(4)
    assert live_export_history([], transient, None, "agent") == transient


def test_transient_conflicts_and_ambiguous_repeated_messages_fail():
    same = _message(1)
    with pytest.raises(HistoryReconstructionError, match="unique"):
        reconcile_transient([same, same], [same])
    with pytest.raises(HistoryReconstructionError, match="unique"):
        reconcile_transient([same], [_message(2)])


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "malformed",
        "invalid_message",
        "digest",
        "tail",
        "count",
        "path",
        "disabled",
        "summary_metadata",
    ],
)
def test_unverifiable_archive_fails_without_writing_output(tmp_path: Path, damage: str):
    current = _checkpoint(tmp_path, [_message(0), *_exchange(1), _message(2)])
    path = next(tmp_path.glob("compacted_*.json"))
    summary = current[0]
    assert summary.channels is not None
    block = summary.channels[FAST_AGENT_COMPACTION_CHANNEL][0]
    assert isinstance(block, TextContent)
    metadata = json.loads(block.text)
    if damage == "missing":
        path.unlink()
    elif damage == "malformed":
        path.write_text("{")
    elif damage == "invalid_message":
        path.write_text('{"messages": [{"role": "invalid", "content": []}]}')
    elif damage == "digest":
        path.write_text(path.read_text() + " ")
    elif damage == "tail":
        current[-1] = _message(99)
    elif damage == "count":
        metadata["retained_messages"] = 100
    elif damage == "path":
        metadata["archive_file"] = "../outside.json"
    elif damage == "disabled":
        metadata["archive_file"] = None
    if damage in {"malformed", "invalid_message"}:
        metadata["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    block.text = "not json" if damage == "summary_metadata" else json.dumps(metadata)
    with pytest.raises(HistoryReconstructionError):
        _trajectory(current, tmp_path)


def test_legacy_requires_unique_timestamped_positive_overlap(tmp_path: Path):
    archived = [_message(0), *_exchange(1), _message(2)]
    current = _checkpoint(tmp_path, archived, linked=False)
    path = next(tmp_path.glob("compacted_*.json"))
    duplicate = path.with_name(path.stem.replace("_agent", "_duplicate_agent") + ".json")
    duplicate.write_bytes(path.read_bytes())
    with pytest.raises(HistoryReconstructionError, match="ambiguous"):
        reconstruct_history(current, tmp_path, "agent")
    duplicate.unlink()
    current = _checkpoint(tmp_path, archived, linked=False, tail=0)
    with pytest.raises(HistoryReconstructionError):
        reconstruct_history(current, tmp_path, "agent")
    current = _checkpoint(tmp_path, archived, linked=False)
    archived[0].timestamp = None
    save_messages(archived, str(path))
    with pytest.raises(HistoryReconstructionError):
        reconstruct_history(current, tmp_path, "agent")


def test_full_export_without_archive_directory_is_explicit_error(tmp_path: Path):
    current = _checkpoint(tmp_path, [_message(0), _message(1)])
    with pytest.raises(HistoryReconstructionError, match="requires"):
        reconstruct_history(current, None, "agent")


def _spec_context(trajectory, step_id: int) -> list[object]:
    """Model context for a step, per the ATIF v1.7 normative replace-boundary rule."""
    prior = [step for step in trajectory.steps if step.step_id < step_id]
    replace_boundaries = [
        index
        for index, step in enumerate(prior)
        if ((step.extra or {}).get("context_management") or {}).get("boundary") == "replace"
    ]
    if not replace_boundaries:
        return [step.message for step in prior]
    boundary = prior[replace_boundaries[-1]]
    assert boundary.observation is not None
    return [
        *(result.content for result in boundary.observation.results),
        *(step.message for step in prior[replace_boundaries[-1] + 1 :]),
    ]


def _context_management(step: AtifStep) -> dict[str, Any]:
    management = (step.extra or {})["context_management"]
    assert isinstance(management, dict)
    return management


def _with_summary_call(messages: list[PromptMessageExtended]) -> list[PromptMessageExtended]:
    usage = {
        "schema": "fast-agent.usage/v2",
        "provider_attempts": [
            {
                "provider": "openai",
                "usage_schema": "openai-chat",
                "model": "summarizer",
                "prompt": {"total": 100, "cache_read": 0},
                "completion": {"total": 20},
                "tool_calls": 0,
                "cost_usd": 0.01,
            }
        ],
    }
    response = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="generated summary")],
        channels={FAST_AGENT_USAGE: [TextContent(type="text", text=json.dumps(usage))]},
    )
    summary = next(m for m in messages if FAST_AGENT_COMPACTION_CHANNEL in (m.channels or {}))
    assert summary.channels is not None
    block = summary.channels[FAST_AGENT_COMPACTION_CHANNEL][0]
    assert isinstance(block, TextContent)
    metadata = json.loads(block.text)
    metadata["summary_request"] = "summarize"
    metadata["summary_response"] = response.model_dump(mode="json", exclude_none=True)
    block.text = json.dumps(metadata)
    return messages


def test_replace_boundary_carries_model_visible_context(tmp_path: Path):
    first = [_message(0, template=True), _message(1), *_exchange(2), _message(3), *_exchange(4)]
    current = _checkpoint(tmp_path, first, templates=1, tail=3)
    second = [*current, _message(20), *_exchange(21), _message(22), *_exchange(23)]
    current = _checkpoint(tmp_path, second, templates=1, tail=3, serial=2)
    final = [*current, _message(40)]
    trajectory = build_atif_trajectory(
        AtifRunSource(
            session_id="test",
            agent_name="agent",
            model_name="test",
            provider="test",
            history=final,
            message_timestamps=tuple(m.timestamp for m in final),
            parent_session_dir=tmp_path,
            system_prompt="system prompt",
        )
    )

    # A spec consumer sees exactly what fast-agent sent the model after compaction.
    last = trajectory.steps[-1]
    summary = current[1].first_text()
    assert _spec_context(trajectory, last.step_id) == [
        summary,
        "system prompt",
        "message 0",
        "message 22",
        "request 23",
    ]
    boundaries = [s for s in trajectory.steps if (s.extra or {}).get("context_management")]
    assert [s.message for s in boundaries] == ["Context compaction performed"] * 2
    by_id = {step.step_id: step for step in trajectory.steps}
    managements = [_context_management(boundary) for boundary in boundaries]
    for management in managements:
        copied = [by_id[i] for i in management["copied_step_ids"]]
        retained = management["retained_step_ids"]
        # Copies restate the retained steps, carry no usage, and never overlap removals.
        assert all(step.is_copied_context for step in copied)
        assert all(step.metrics is None and step.llm_call_count in (None, 0) for step in copied)
        assert [(s.extra or {})["copied_from_step_id"] for s in copied] == retained
        assert [by_id[i].message for i in retained] == [s.message for s in copied]
        assert not set(retained) & set(management["removed_step_ids"])
    # The second boundary removes the first boundary's summary with the compacted steps.
    assert boundaries[0].step_id in managements[1]["removed_step_ids"]
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.extra is not None
    assert trajectory.final_metrics.extra["llm_usage_expected_call_count"] == 4


def test_summary_call_is_embedded_and_accounted(tmp_path: Path):
    archived = [_message(0), *_exchange(1), _message(2), *_exchange(3)]
    current = _with_summary_call(_checkpoint(tmp_path, archived, tail=3))
    trajectory = _trajectory([*current, _message(10)], tmp_path)

    boundary = next(s for s in trajectory.steps if (s.extra or {}).get("context_management"))
    management = _context_management(boundary)
    assert "summary_call" not in management
    assert boundary.observation is not None
    [reference] = boundary.observation.results[0].subagent_trajectory_ref or []
    [child] = trajectory.subagent_trajectories or []
    assert reference.trajectory_id == child.trajectory_id
    assert [step.message for step in child.steps] == ["summarize", "generated summary"]
    assert child.extra is not None
    assert child.extra["input_step_ids"] == management["removed_step_ids"]
    metrics = trajectory.final_metrics
    assert metrics is not None and metrics.extra is not None
    assert metrics.extra["subagent_prompt_tokens"] == 100
    assert metrics.extra["summary_compaction_usage_complete"] is True
    assert "accounting_scope" not in metrics.extra
