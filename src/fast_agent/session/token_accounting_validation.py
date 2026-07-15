#!/usr/bin/env python3
"""Validate persisted sessions, Codex rollouts, and ATIF token accounting artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Never

from mcp.types import TextContent

from fast_agent.constants import FAST_AGENT_USAGE
from fast_agent.llm.usage_tracking import UsageReport
from fast_agent.mcp.prompt_serialization import load_messages
from fast_agent.session.atif_models import AtifMetrics, AtifStep, AtifTrajectory
from fast_agent.session.snapshot import load_session_snapshot


class ArtifactValidationError(ValueError):
    """Raised when a token-accounting artifact is invalid."""


@dataclass(frozen=True, slots=True)
class SessionReport:
    session_id: str
    agents: int
    messages: int
    usage_records: int


@dataclass(frozen=True, slots=True)
class CodexReport:
    records: int
    turns: int
    token_records: int
    tool_calls: int


@dataclass(frozen=True, slots=True)
class AtifReport:
    steps: int
    metric_steps: int
    tool_calls: int
    cached_tokens: int


@dataclass(frozen=True, slots=True)
class ValidationReport:
    session: SessionReport
    codex: CodexReport
    atif: AtifReport


UsageSignature = tuple[int | None, int | None, int | None, int | None]


def _fail(message: str) -> Never:
    raise ArtifactValidationError(message)


def _json_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"failed to read {path}: {exc}") from exc
    return _json_object(value, label=str(path))


def _integer(mapping: dict[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(f"{key} must be an integer or null")
    if value < 0:
        _fail(f"{key} must be nonnegative")
    return value


def _nested_integer(mapping: dict[str, object], parent: str, key: str) -> int | None:
    nested = mapping.get(parent)
    if nested is None:
        return None
    return _integer(_json_object(nested, label=parent), key)


def _raw_usage_signature(
    raw_usage: object,
    *,
    provider: object,
    label: str,
) -> UsageSignature:
    raw = _json_object(raw_usage, label=f"{label}.raw_usage")
    provider_name = provider if isinstance(provider, str) else ""

    if provider_name.startswith("anthropic") or "cache_creation_input_tokens" in raw:
        uncached = _integer(raw, "input_tokens")
        cache_read = _integer(raw, "cache_read_input_tokens")
        cache_write = _integer(raw, "cache_creation_input_tokens")
        prompt = None
        if uncached is not None and cache_read is not None and cache_write is not None:
            prompt = uncached + cache_read + cache_write
        return (
            prompt,
            _integer(raw, "output_tokens"),
            cache_read,
            _nested_integer(raw, "output_tokens_details", "thinking_tokens"),
        )

    if "prompt_token_count" in raw:
        prompt = _integer(raw, "prompt_token_count")
        tool_use = _integer(raw, "tool_use_prompt_token_count")
        if prompt is not None and tool_use is not None:
            prompt += tool_use
        visible = _integer(raw, "candidates_token_count")
        if visible is None:
            visible = _integer(raw, "response_token_count")
        reasoning = _integer(raw, "thoughts_token_count")
        completion = visible + reasoning if visible is not None and reasoning is not None else None
        if completion is None:
            total = _integer(raw, "total_token_count")
            if total is not None and prompt is not None and total >= prompt:
                completion = total - prompt
        return (
            prompt,
            completion,
            _integer(raw, "cached_content_token_count"),
            reasoning,
        )

    if "input_tokens" in raw:
        return (
            _integer(raw, "input_tokens"),
            _integer(raw, "output_tokens"),
            _nested_integer(raw, "input_tokens_details", "cached_tokens"),
            _nested_integer(raw, "output_tokens_details", "reasoning_tokens"),
        )

    if "prompt_tokens" in raw:
        return (
            _integer(raw, "prompt_tokens"),
            _integer(raw, "completion_tokens"),
            _nested_integer(raw, "prompt_tokens_details", "cached_tokens"),
            _nested_integer(raw, "completion_tokens_details", "reasoning_tokens"),
        )

    _fail(f"{label}.raw_usage: unsupported provider usage shape")


def validate_session(session_dir: Path) -> tuple[SessionReport, list[UsageSignature]]:
    session_dir = session_dir.expanduser().resolve()
    snapshot_path = session_dir / "session.json"
    snapshot = load_session_snapshot(_read_json_object(snapshot_path))

    messages = 0
    usage_records = 0
    histories = 0
    raw_signatures: list[UsageSignature] = []
    for agent_name, agent in snapshot.continuation.agents.items():
        if agent.history_file is None:
            continue
        history_path = session_dir / agent.history_file
        if not history_path.is_file():
            _fail(f"history for agent {agent_name!r} does not exist: {history_path}")
        histories += 1
        history = load_messages(str(history_path))
        messages += len(history)
        for message in history:
            for block in (message.channels or {}).get(FAST_AGENT_USAGE, ()):
                if not isinstance(block, TextContent):
                    _fail(f"{history_path}: {FAST_AGENT_USAGE} contains non-text content")
                try:
                    payload = json.loads(block.text)
                except json.JSONDecodeError as exc:
                    raise ArtifactValidationError(
                        f"{history_path}: invalid {FAST_AGENT_USAGE} JSON: {exc}"
                    ) from exc
                usage = _json_object(payload, label=f"{history_path}:{FAST_AGENT_USAGE}")
                try:
                    report = UsageReport.model_validate(usage)
                except ValueError as exc:
                    raise ArtifactValidationError(
                        f"{history_path}: invalid canonical usage: {exc}"
                    ) from exc
                for index, attempt in enumerate(report.provider_attempts):
                    raw_signature = _raw_usage_signature(
                        attempt.raw_usage,
                        provider=attempt.provider.value,
                        label=f"{history_path}:{FAST_AGENT_USAGE}.provider_attempts[{index}]",
                    )
                    canonical_signature = (
                        attempt.prompt.total,
                        attempt.completion.total,
                        attempt.prompt.cache_read,
                        attempt.completion.reasoning,
                    )
                    if canonical_signature != raw_signature:
                        _fail(
                            f"{history_path}: canonical usage disagrees with raw provider usage"
                        )
                consumed = report.consumed
                raw_signatures.append(
                    (
                        consumed.prompt.total,
                        consumed.completion.total,
                        consumed.prompt.cache_read,
                        consumed.completion.reasoning,
                    )
                )
                usage_records += 1

    if histories == 0:
        _fail(f"{snapshot_path}: no exportable agent histories")
    if histories != 1:
        _fail(f"{snapshot_path}: accounting validation requires exactly one agent history")
    if messages == 0:
        _fail(f"{snapshot_path}: agent histories are empty")
    if usage_records == 0:
        _fail(f"{snapshot_path}: no {FAST_AGENT_USAGE} records")

    return (
        SessionReport(
            session_id=snapshot.session_id,
            agents=histories,
            messages=messages,
            usage_records=usage_records,
        ),
        raw_signatures,
    )


def _codex_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArtifactValidationError(f"failed to read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            _fail(f"{path}:{line_number}: blank JSONL record")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(f"{path}:{line_number}: {exc}") from exc
        record = _json_object(value, label=f"{path}:{line_number}")
        if not isinstance(record.get("type"), str):
            _fail(f"{path}:{line_number}: record type must be a string")
        _json_object(record.get("payload"), label=f"{path}:{line_number}.payload")
        records.append(record)
    if not records:
        _fail(f"{path}: empty Codex rollout")
    return records


def _validate_codex_usage(usage: dict[str, object], *, label: str) -> None:
    input_tokens = _integer(usage, "input_tokens")
    cached_tokens = _integer(usage, "cached_input_tokens")
    output_tokens = _integer(usage, "output_tokens")
    reasoning_tokens = _integer(usage, "reasoning_output_tokens")
    total_tokens = _integer(usage, "total_tokens")

    if cached_tokens is not None and input_tokens is not None and cached_tokens > input_tokens:
        _fail(f"{label}: cached input exceeds input total")
    if reasoning_tokens is not None and output_tokens is not None:
        if reasoning_tokens > output_tokens:
            _fail(f"{label}: reasoning output exceeds output total")
    if None not in (input_tokens, output_tokens, total_tokens):
        assert input_tokens is not None
        assert output_tokens is not None
        assert total_tokens is not None
        if total_tokens != input_tokens + output_tokens:
            _fail(f"{label}: total_tokens is not input_tokens + output_tokens")


def validate_codex(path: Path) -> tuple[CodexReport, list[dict[str, object]]]:
    records = _codex_records(path.expanduser().resolve())
    if records[0]["type"] != "session_meta":
        _fail(f"{path}: first Codex record must be session_meta")

    started: set[str] = set()
    completed: set[str] = set()
    calls: set[str] = set()
    outputs: set[str] = set()
    token_usages: list[dict[str, object]] = []

    for index, record in enumerate(records, start=1):
        payload = _json_object(record["payload"], label=f"{path}:{index}.payload")
        record_type = record["type"]
        payload_type = payload.get("type")

        if record_type == "event_msg" and payload_type == "task_started":
            turn_id = payload.get("turn_id")
            if not isinstance(turn_id, str) or not turn_id:
                _fail(f"{path}:{index}: task_started requires turn_id")
            if turn_id in started:
                _fail(f"{path}:{index}: duplicate task_started for {turn_id}")
            started.add(turn_id)
        elif record_type == "event_msg" and payload_type == "task_complete":
            turn_id = payload.get("turn_id")
            if not isinstance(turn_id, str) or not turn_id:
                _fail(f"{path}:{index}: task_complete requires turn_id")
            if turn_id not in started:
                _fail(f"{path}:{index}: task_complete precedes task_started")
            completed.add(turn_id)
        elif record_type == "event_msg" and payload_type == "token_count":
            info = _json_object(payload.get("info"), label=f"{path}:{index}.payload.info")
            last = _json_object(
                info.get("last_token_usage"),
                label=f"{path}:{index}.payload.info.last_token_usage",
            )
            _validate_codex_usage(last, label=f"{path}:{index}.last_token_usage")
            total = _json_object(
                info.get("total_token_usage"),
                label=f"{path}:{index}.payload.info.total_token_usage",
            )
            _validate_codex_usage(total, label=f"{path}:{index}.total_token_usage")
            token_usages.append(last)
        elif record_type == "response_item" and payload_type == "function_call":
            call_id = payload.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                _fail(f"{path}:{index}: function_call requires call_id")
            calls.add(call_id)
        elif record_type == "response_item" and payload_type == "function_call_output":
            call_id = payload.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                _fail(f"{path}:{index}: function_call_output requires call_id")
            outputs.add(call_id)

    if started != completed:
        _fail(f"{path}: started/completed turn ids differ")
    missing_calls = outputs - calls
    if missing_calls:
        _fail(f"{path}: tool outputs without calls: {sorted(missing_calls)}")
    if not token_usages:
        _fail(f"{path}: no token_count records")

    return (
        CodexReport(
            records=len(records),
            turns=len(started),
            token_records=len(token_usages),
            tool_calls=len(calls),
        ),
        token_usages,
    )


def _metric_extra_integer(metrics: AtifMetrics, key: str) -> int | None:
    extra = metrics.extra or {}
    value = extra.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(f"ATIF metrics.extra.{key} must be a nonnegative integer or null")
    return value


def _validate_atif_step(step: AtifStep) -> None:
    metrics = step.metrics
    if metrics is None:
        return
    counts = {
        "prompt_tokens": metrics.prompt_tokens,
        "completion_tokens": metrics.completion_tokens,
        "cached_tokens": metrics.cached_tokens,
    }
    for name, value in counts.items():
        if value is not None and value < 0:
            _fail(f"ATIF step {step.step_id}: {name} must be nonnegative")
    if (
        metrics.cached_tokens is not None
        and metrics.prompt_tokens is not None
        and metrics.cached_tokens > metrics.prompt_tokens
    ):
        _fail(f"ATIF step {step.step_id}: cached_tokens exceeds prompt_tokens")
    reasoning = _metric_extra_integer(metrics, "reasoning_tokens")
    cache_write = _metric_extra_integer(metrics, "cache_write_tokens")
    if (
        reasoning is not None
        and metrics.completion_tokens is not None
        and reasoning > metrics.completion_tokens
    ):
        _fail(f"ATIF step {step.step_id}: reasoning_tokens exceeds completion_tokens")
    if (
        cache_write is not None
        and metrics.prompt_tokens is not None
        and cache_write > metrics.prompt_tokens
    ):
        _fail(f"ATIF step {step.step_id}: cache_write_tokens exceeds prompt_tokens")


def _complete_sum(steps: list[AtifStep], field: str) -> int | None:
    values: list[int] = []
    for step in steps:
        metrics = step.metrics
        if metrics is None:
            return None
        value = getattr(metrics, field)
        if value is None:
            return None
        values.append(value)
    return sum(values)


def _validate_final_metrics(trajectory: AtifTrajectory) -> None:
    final = trajectory.final_metrics
    if final is None:
        _fail("ATIF trajectory is missing final_metrics")
    relevant = [
        step
        for step in trajectory.steps
        if step.source == "agent" and (step.llm_call_count is None or step.llm_call_count > 0)
    ]
    expected = {
        "total_prompt_tokens": _complete_sum(relevant, "prompt_tokens"),
        "total_completion_tokens": _complete_sum(relevant, "completion_tokens"),
        "total_cached_tokens": _complete_sum(relevant, "cached_tokens"),
    }
    for field, value in expected.items():
        actual = getattr(final, field)
        if actual is not None and actual != value:
            _fail(f"ATIF final_metrics.{field}={actual} does not match complete step sum {value}")
        if actual is not None and value is None:
            _fail(f"ATIF final_metrics.{field} presents an incomplete sum as complete")
    if final.total_steps is not None and final.total_steps != len(trajectory.steps):
        _fail("ATIF final_metrics.total_steps does not match trajectory steps")


def validate_atif(
    path: Path,
    *,
    require_cache: bool = False,
    require_tool: bool = False,
    expect_reasoning: str | None = None,
) -> tuple[AtifReport, AtifTrajectory]:
    path = path.expanduser().resolve()
    try:
        trajectory = AtifTrajectory.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(f"failed to validate {path}: {exc}") from exc

    for step in trajectory.steps:
        _validate_atif_step(step)
    _validate_final_metrics(trajectory)

    metric_steps: list[tuple[AtifStep, AtifMetrics]] = []
    for step in trajectory.steps:
        if step.metrics is not None:
            metric_steps.append((step, step.metrics))
    cached_tokens = sum(metrics.cached_tokens or 0 for _, metrics in metric_steps)
    tool_calls = sum(len(step.tool_calls or []) for step in trajectory.steps)

    if require_cache and cached_tokens == 0:
        _fail(f"{path}: expected positive cached token usage")
    if require_tool and tool_calls == 0:
        _fail(f"{path}: expected at least one tool call")
    if expect_reasoning is not None:
        agent_steps = [
            step
            for step in trajectory.steps
            if step.source == "agent" and (step.llm_call_count is None or step.llm_call_count > 0)
        ]
        mismatched = [
            step.step_id for step in agent_steps if step.reasoning_effort != expect_reasoning
        ]
        if mismatched:
            _fail(
                f"{path}: expected reasoning_effort={expect_reasoning!r} "
                f"on agent steps {mismatched}"
            )
        if expect_reasoning == "none":
            nonzero = [
                step.step_id
                for step, metrics in metric_steps
                if _metric_extra_integer(metrics, "reasoning_tokens") not in {None, 0}
            ]
            if nonzero:
                _fail(f"{path}: reasoning=none produced reasoning tokens on steps {nonzero}")

    return (
        AtifReport(
            steps=len(trajectory.steps),
            metric_steps=len(metric_steps),
            tool_calls=tool_calls,
            cached_tokens=cached_tokens,
        ),
        trajectory,
    )


def _codex_usage_signature(usage: dict[str, object]) -> UsageSignature:
    return (
        _integer(usage, "input_tokens"),
        _integer(usage, "output_tokens"),
        _integer(usage, "cached_input_tokens"),
        _integer(usage, "reasoning_output_tokens"),
    )


def _atif_usage_signature(step: AtifStep) -> UsageSignature:
    assert step.metrics is not None
    return (
        step.metrics.prompt_tokens,
        step.metrics.completion_tokens,
        step.metrics.cached_tokens,
        _metric_extra_integer(step.metrics, "reasoning_tokens"),
    )


def validate_cross_format(
    codex_usage: list[dict[str, object]],
    trajectory: AtifTrajectory,
    raw_usage: list[UsageSignature],
) -> None:
    atif_usage = [
        _atif_usage_signature(step) for step in trajectory.steps if step.metrics is not None
    ]
    codex_signatures = [_codex_usage_signature(usage) for usage in codex_usage]
    if codex_signatures != atif_usage:
        _fail(
            "Codex token_count records do not match ATIF metric steps:\n"
            f"Codex: {codex_signatures}\n"
            f"ATIF: {atif_usage}"
        )
    if raw_usage != atif_usage:
        _fail(
            "Raw provider usage does not match exported token accounting:\n"
            f"Raw: {raw_usage}\n"
            f"ATIF: {atif_usage}"
        )


def validate_artifacts(
    *,
    session_dir: Path,
    codex_path: Path,
    atif_path: Path,
    require_cache: bool = False,
    require_tool: bool = False,
    expect_reasoning: str | None = None,
) -> ValidationReport:
    session, raw_usage = validate_session(session_dir)
    codex, codex_usage = validate_codex(codex_path)
    atif, trajectory = validate_atif(
        atif_path,
        require_cache=require_cache,
        require_tool=require_tool,
        expect_reasoning=expect_reasoning,
    )
    validate_cross_format(codex_usage, trajectory, raw_usage)
    return ValidationReport(session=session, codex=codex, atif=atif)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True, help="Persisted session directory.")
    parser.add_argument("--codex", type=Path, required=True, help="Exported Codex JSONL path.")
    parser.add_argument("--atif", type=Path, required=True, help="Exported ATIF JSON path.")
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--require-tool", action="store_true")
    parser.add_argument("--expect-reasoning")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = validate_artifacts(
        session_dir=args.session,
        codex_path=args.codex,
        atif_path=args.atif,
        require_cache=args.require_cache,
        require_tool=args.require_tool,
        expect_reasoning=args.expect_reasoning,
    )
    payload = asdict(report)
    if args.json_output:
        print(json.dumps(payload, indent=2))
        return
    print(
        "token accounting valid: "
        f"{report.session.usage_records} usage records, "
        f"{report.codex.turns} Codex turns, "
        f"{report.atif.metric_steps} ATIF metric steps, "
        f"{report.atif.cached_tokens} cached tokens"
    )


if __name__ == "__main__":
    main()
