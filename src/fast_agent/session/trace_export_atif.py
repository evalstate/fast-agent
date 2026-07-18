"""ATIF v1.7 writer for persisted Fast-Agent sessions."""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import parse_qs, urlparse

from mcp.types import CallToolResult, ImageContent, TextContent

from fast_agent.constants import (
    FAST_AGENT_PROCESS_POLL_FOLD,
    FAST_AGENT_SHELL_PROCESS_METADATA,
    FAST_AGENT_TIMING,
    FAST_AGENT_TOOL_METADATA,
    FAST_AGENT_TOOL_TIMING,
    FAST_AGENT_USAGE,
    REASONING,
)
from fast_agent.history.process_poll_fold_audit import (
    ArchivedContextRewrite,
    ProcessPollFoldAudit,
)
from fast_agent.llm.usage_tracking import UsageReport, UsageSummary
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.privacy.sanitizer import RedactionAccumulator
from fast_agent.session.atif_models import (
    AtifAgent,
    AtifContent,
    AtifContentPart,
    AtifFinalMetrics,
    AtifImageSource,
    AtifMetrics,
    AtifObservation,
    AtifObservationResult,
    AtifStep,
    AtifSubagentTrajectoryRef,
    AtifToolCall,
    AtifTrajectory,
)
from fast_agent.session.trace_export_models import ExportResult

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime
    from typing import Literal, TypeGuard

    from fast_agent.privacy.sanitizer import RedactionSummary, TraceSanitizer
    from fast_agent.session.trace_export_models import ResolvedSessionExport


@dataclass(frozen=True, slots=True)
class AtifRunSource:
    """Provider-independent input used by both live and persisted exports."""

    session_id: str
    agent_name: str
    model_name: str | None
    provider: str | None
    history: list[PromptMessageExtended]
    message_timestamps: tuple[datetime | None, ...]
    child_trajectory_dir: Path | None = None
    tool_definitions: list[dict[str, object]] | None = None
    extra: dict[str, object] | None = None
    notes: str | None = None
    system_prompt: str | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True, slots=True)
class _AuditMessage:
    message: PromptMessageExtended
    timestamp: datetime | None


@dataclass(frozen=True, slots=True)
class _ContextRewrite:
    timestamp: datetime | None
    summary: str
    fold: dict[str, object]
    removed_call_ids: tuple[str, ...]
    retained_call_ids: tuple[str, ...]


type _AuditHistoryItem = _AuditMessage | _ContextRewrite


@cache
def _package_version() -> str:
    try:
        return version("fast-agent-mcp")
    except PackageNotFoundError:
        return "unknown"


def _channel_text(message: PromptMessageExtended, channel: str) -> str | None:
    blocks = (message.channels or {}).get(channel, ())
    texts = [block.text for block in blocks if isinstance(block, TextContent)]
    return "\n".join(texts) or None


def _json_channel_mapping(
    message: PromptMessageExtended,
    channel: str,
) -> dict[str, object] | None:
    for block in reversed((message.channels or {}).get(channel, ())):
        if not isinstance(block, TextContent):
            continue
        try:
            value = json.loads(block.text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
    return None


def _usage(message: PromptMessageExtended) -> AtifMetrics | None:
    values: list[object] = []
    for block in (message.channels or {}).get(FAST_AGENT_USAGE, ()):
        if not isinstance(block, TextContent):
            continue
        try:
            values.append(json.loads(block.text))
        except json.JSONDecodeError:
            continue
    if not values:
        return None
    value = values[-1]
    if not isinstance(value, dict):
        return None
    try:
        report = UsageReport.model_validate(value)
    except ValueError:
        return None
    turn = report.final_attempt
    usage = report.consumed
    costs = [attempt.cost_usd for attempt in report.provider_attempts]
    cost_usd = (
        sum(cost for cost in costs if cost is not None)
        if all(cost is not None for cost in costs)
        else None
    )
    metric_extra = {
        key: value
        for key, value in {
            "provider": turn.provider.value,
            "upstream_provider": turn.upstream_provider,
            "usage_schema": turn.usage_schema.value,
            "model": turn.model,
            "cache_write_tokens": usage.prompt.cache_write,
            "reasoning_tokens": usage.completion.reasoning,
            "tool_use_prompt_tokens": usage.prompt.tool_use,
            "tool_calls": usage.tool_calls,
            "reasoning_effort": turn.reasoning_effort,
            "requested_service_tier": turn.requested_service_tier,
            "service_tier": turn.service_tier,
            "raw_usage": [attempt.raw_usage for attempt in report.provider_attempts],
        }.items()
        if value is not None
    }
    return AtifMetrics(
        prompt_tokens=usage.prompt.total,
        completion_tokens=usage.completion.total,
        cached_tokens=usage.prompt.cache_read,
        cost_usd=cost_usd,
        extra=metric_extra or None,
    )


def _tool_calls(message: PromptMessageExtended) -> list[AtifToolCall] | None:
    calls = []
    for call_id, request in (message.tool_calls or {}).items():
        arguments = request.params.arguments or {}
        calls.append(
            AtifToolCall(
                tool_call_id=call_id,
                function_name=request.params.name,
                arguments=arguments,
            )
        )
    return calls or None


def _reasoning_effort(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    query = parse_qs(urlparse(model_name).query)
    for key in ("reasoning_effort", "reasoning"):
        values = query.get(key)
        if values:
            return values[-1]
    return None


type AtifImageMime = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


def _is_atif_image_mime(value: str) -> TypeGuard[AtifImageMime]:
    return value in {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _atif_content(blocks: list[object]) -> AtifContent:
    images = [
        block
        for block in blocks
        if isinstance(block, ImageContent) and _is_atif_image_mime(block.mimeType)
    ]
    texts = [block.text for block in blocks if isinstance(block, TextContent)]
    if not images:
        return "\n".join(texts)
    parts = [AtifContentPart(type="text", text=text) for text in texts]
    parts.extend(
        AtifContentPart(
            type="image",
            source=AtifImageSource(
                media_type=cast("AtifImageMime", image.mimeType),
                path=f"data:{image.mimeType};base64,{image.data}",
            ),
        )
        for image in images
    )
    return parts


def _result_content(result: CallToolResult) -> AtifContent:
    return _atif_content(list(result.content))


def _attach_tool_result(
    steps: list[AtifStep],
    call_id: str,
    result: CallToolResult,
    extra: dict[str, object],
) -> bool:
    for step in reversed(steps):
        if call_id not in {call.tool_call_id for call in step.tool_calls or []}:
            continue
        observation_result = AtifObservationResult(
            source_call_id=call_id,
            content=_result_content(result),
            extra=extra,
        )
        if step.observation is None:
            step.observation = AtifObservation(results=[observation_result])
        else:
            step.observation.results.append(observation_result)
        return True
    return False


def _tool_result_extra(
    message: PromptMessageExtended,
    call_id: str,
    result: CallToolResult,
) -> dict[str, object]:
    extra: dict[str, object] = {"is_error": bool(result.isError)}
    timing = (_json_channel_mapping(message, FAST_AGENT_TOOL_TIMING) or {}).get(call_id)
    if isinstance(timing, dict):
        timing_mapping: dict[str, object] = {
            str(key): value for key, value in timing.items()
        }
        for key in ("timing_ms", "transport_channel"):
            value = timing_mapping.get(key)
            if value is not None:
                extra[key] = value
    elif isinstance(timing, int | float) and not isinstance(timing, bool):
        extra["timing_ms"] = timing
    metadata = (_json_channel_mapping(message, FAST_AGENT_TOOL_METADATA) or {}).get(call_id)
    if isinstance(metadata, dict):
        extra["tool_metadata"] = metadata
    process_metadata = (
        _json_channel_mapping(message, FAST_AGENT_SHELL_PROCESS_METADATA) or {}
    ).get(call_id)
    if isinstance(process_metadata, dict):
        extra["process_metadata"] = process_metadata
    fold_metadata = _json_channel_mapping(message, FAST_AGENT_PROCESS_POLL_FOLD)
    if fold_metadata:
        extra["process_poll_fold"] = fold_metadata
    return extra


def _step_timing_extra(message: PromptMessageExtended) -> dict[str, object]:
    timing = _json_channel_mapping(message, FAST_AGENT_TIMING) or {}
    return {
        key: value
        for key in ("duration_ms", "ttft_ms", "time_to_response_ms")
        if (value := timing.get(key)) is not None
    }


def _process_poll_folds(
    history: list[PromptMessageExtended],
) -> list[dict[str, object]]:
    return [
        metadata
        for message in history
        if (metadata := _json_channel_mapping(message, FAST_AGENT_PROCESS_POLL_FOLD))
    ]


def _parse_process_poll_fold_audit(
    fold: dict[str, object],
) -> ProcessPollFoldAudit:
    if "audit" not in fold:
        raise ValueError("Managed-process poll fold is missing its audit archive")
    try:
        return ProcessPollFoldAudit.model_validate(fold.get("audit"))
    except ValueError as exc:
        raise ValueError(
            "Managed-process poll fold audit archive is invalid"
        ) from exc


def _context_rewrite(
    rewrite: ArchivedContextRewrite,
    *,
    timestamp: datetime | None,
) -> _ContextRewrite:
    return _ContextRewrite(
        timestamp=timestamp,
        summary=rewrite.summary,
        fold=rewrite.fold,
        removed_call_ids=tuple(rewrite.removed_call_ids),
        retained_call_ids=tuple(rewrite.retained_call_ids),
    )


def _reconstruct_process_poll_audit_items(
    audit: ProcessPollFoldAudit,
    *,
    fallback_timestamp: datetime | None,
) -> list[_AuditHistoryItem]:
    items: list[_AuditHistoryItem] = []
    rewrite_index = 0
    exchanges = [
        *audit.removed_exchanges,
        *audit.retained_exchanges,
    ]
    for exchange in exchanges:
        for archived in (exchange.request, exchange.result):
            archived_timestamp = archived.timestamp or fallback_timestamp
            items.append(
                _AuditMessage(
                    message=archived,
                    timestamp=archived_timestamp,
                )
            )
            if (
                rewrite_index < len(audit.context_rewrites)
                and audit.context_rewrites[rewrite_index].after_call_id
                in (archived.tool_results or {})
            ):
                items.append(
                    _context_rewrite(
                        audit.context_rewrites[rewrite_index],
                        timestamp=archived_timestamp,
                    )
                )
                rewrite_index += 1
    if rewrite_index != len(audit.context_rewrites):
        raise ValueError(
            "Managed-process poll fold context rewrite placement is inconsistent"
        )
    return items


def _expand_process_poll_folds(
    source: AtifRunSource,
) -> list[_AuditHistoryItem]:
    items: list[_AuditHistoryItem] = []
    history = source.history
    index = 0

    while index < len(history):
        message = history[index]
        timestamp = source.message_timestamps[index]
        if index + 1 >= len(history):
            items.append(_AuditMessage(message=message, timestamp=timestamp))
            index += 1
            continue

        result_message = history[index + 1]
        fold = _json_channel_mapping(result_message, FAST_AGENT_PROCESS_POLL_FOLD)
        if fold is None:
            items.append(_AuditMessage(message=message, timestamp=timestamp))
            index += 1
            continue
        audit = _parse_process_poll_fold_audit(fold)
        retained_call_ids = [
            exchange.call_id for exchange in audit.retained_exchanges
        ]
        retained_call_id = retained_call_ids[-1]
        if (
            retained_call_id not in (message.tool_calls or {})
            or retained_call_id not in (result_message.tool_results or {})
        ):
            raise ValueError("Managed-process poll fold retained exchange is invalid")

        earlier_retained_call_ids = retained_call_ids[:-1]
        retained_suffix_items = len(earlier_retained_call_ids) * 2
        if retained_suffix_items > len(items):
            raise ValueError("Managed-process poll fold retained-step archive is inconsistent")
        if earlier_retained_call_ids:
            suffix = items[-retained_suffix_items:]
            suffix_call_ids = [
                call_id
                for archived_item in suffix
                if isinstance(archived_item, _AuditMessage)
                for call_id in (archived_item.message.tool_calls or {})
            ]
            if suffix_call_ids != earlier_retained_call_ids:
                raise ValueError(
                    "Managed-process poll fold retained call IDs are inconsistent"
                )
            del items[-retained_suffix_items:]

        fallback_timestamp = result_message.timestamp or source.message_timestamps[index + 1]
        items.extend(
            _reconstruct_process_poll_audit_items(
                audit,
                fallback_timestamp=fallback_timestamp,
            )
        )
        index += 2

    return items


def _resolve_call_step_ids(
    call_ids: tuple[str, ...],
    call_step_ids: dict[str, int],
) -> list[int]:
    missing = [call_id for call_id in call_ids if call_id not in call_step_ids]
    if missing:
        raise ValueError(
            "Managed-process context rewrite references unknown tool calls: "
            + ", ".join(missing)
        )
    return [call_step_ids[call_id] for call_id in call_ids]


def build_atif_trajectory(source: AtifRunSource) -> AtifTrajectory:
    model_name = source.model_name
    steps: list[AtifStep] = []
    call_step_ids: dict[str, int] = {}
    audit_items = _expand_process_poll_folds(source)
    context_boundary_count = 0
    if source.system_prompt:
        steps.append(
            AtifStep(
                step_id=1,
                source="system",
                message=source.system_prompt,
            )
        )
    for item in audit_items:
        if isinstance(item, _ContextRewrite):
            removed_step_ids = _resolve_call_step_ids(
                item.removed_call_ids,
                call_step_ids,
            )
            retained_step_ids = _resolve_call_step_ids(
                item.retained_call_ids,
                call_step_ids,
            )
            context_management: dict[str, object] = {
                "type": "compaction",
                "boundary": "truncate",
                "scope": "step_ids",
                "removed_step_ids": removed_step_ids,
                "replacement_source": "observation",
                "replacement_position": "prepend_to_retained_observation",
                "strategy": "managed_process_poll_fold",
            }
            if retained_step_ids:
                context_management["retained_step_ids"] = retained_step_ids
            timestamp_text = (
                item.timestamp.isoformat().replace("+00:00", "Z")
                if item.timestamp
                else None
            )
            context_boundary_count += 1
            steps.append(
                AtifStep(
                    step_id=len(steps) + 1,
                    timestamp=timestamp_text,
                    source="system",
                    message="Managed process polling context rewritten",
                    observation=AtifObservation(
                        results=[
                            AtifObservationResult(
                                content=item.summary,
                            )
                        ]
                    ),
                    extra={
                        "context_management": context_management,
                        "process_poll_fold": item.fold,
                    },
                )
            )
            continue

        message = item.message
        timestamp = item.timestamp
        timestamp_text = timestamp.isoformat().replace("+00:00", "Z") if timestamp else None
        if message.tool_results:
            for call_id, result in message.tool_results.items():
                _attach_tool_result(
                    steps,
                    call_id,
                    result,
                    _tool_result_extra(message, call_id, result),
                )
            continue
        step_source = "agent" if message.role == "assistant" else "user"
        calls = _tool_calls(message) if step_source == "agent" else None
        step_extra = _step_timing_extra(message)
        if message.stop_reason is not None:
            step_extra["stop_reason"] = str(message.stop_reason)
        if message.phase is not None:
            step_extra["phase"] = str(message.phase)
        steps.append(
            AtifStep(
                step_id=len(steps) + 1,
                timestamp=timestamp_text,
                source=step_source,
                model_name=model_name if step_source == "agent" else None,
                reasoning_effort=(
                    source.reasoning_effort or _reasoning_effort(model_name)
                    if step_source == "agent"
                    else None
                ),
                message=_atif_content(list(message.content)),
                reasoning_content=(
                    _channel_text(message, REASONING) if step_source == "agent" else None
                ),
                tool_calls=calls,
                metrics=_usage(message) if step_source == "agent" else None,
                llm_call_count=1 if step_source == "agent" else None,
                extra=step_extra if step_source == "agent" and step_extra else None,
            )
        )
        if calls:
            for call in calls:
                call_step_ids[call.tool_call_id] = steps[-1].step_id
    if not steps:
        raise ValueError("ATIF trajectories require at least one interaction step")
    metrics = [
        step.metrics
        for step in steps
        if step.source == "agent" and step.llm_call_count != 0
    ]
    total_reasoning_tokens = _sum_optional_int(
        _metric_extra_int(item, "reasoning_tokens") if item is not None else None
        for item in metrics
    )
    total_tool_use_tokens = _sum_optional_int(
        _metric_extra_int(item, "tool_use_prompt_tokens") if item is not None else None
        for item in metrics
    )
    folds = _process_poll_folds(source.history)
    folded_polls = sum(
        value
        for fold in folds
        if (value := fold.get("polls_folded")) is not None and type(value) is int
    )
    total_prompt_tokens = _sum_optional_int(
        item.prompt_tokens if item is not None else None for item in metrics
    )
    total_completion_tokens = _sum_optional_int(
        item.completion_tokens if item is not None else None for item in metrics
    )
    total_cached_tokens = _sum_optional_int(
        item.cached_tokens if item is not None else None for item in metrics
    )
    total = AtifFinalMetrics(
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_cached_tokens=total_cached_tokens,
        total_cost_usd=_sum_optional_float(
            item.cost_usd if item is not None else None for item in metrics
        ),
        total_steps=len(steps),
        extra={
            key: value
            for key, value in {
                "total_reasoning_tokens": total_reasoning_tokens,
                "total_tool_use_tokens": total_tool_use_tokens,
                "folded_process_poll_steps": folded_polls or None,
                "process_poll_context_rewrites": context_boundary_count or None,
            }.items()
            if value is not None
        }
        or None,
    )
    trajectory = AtifTrajectory(
        session_id=source.session_id,
        trajectory_id=f"traj_{uuid.uuid4().hex}",
        agent=AtifAgent(
            name="fast-agent",
            version=_package_version(),
            model_name=model_name,
            tool_definitions=source.tool_definitions,
            extra={
                key: value
                for key, value in {
                    "target_agent": source.agent_name,
                    "provider": source.provider,
                }.items()
                if value is not None
            },
        ),
        steps=steps,
        final_metrics=total,
        extra=source.extra,
        notes=_atif_notes(
            source.notes,
            context_boundary_count=context_boundary_count,
        ),
    )
    _embed_subagent_trajectories(trajectory, source)
    return AtifTrajectory.model_validate(trajectory.model_dump())


def _atif_notes(
    notes: str | None,
    *,
    context_boundary_count: int,
) -> str | None:
    additions: list[str] = []
    if context_boundary_count:
        additions.append(
            "Managed-process polling folds preserve every original LLM/tool step for "
            "auditability. System context-management steps identify the exact prior "
            "step IDs removed from subsequent model context; their observation summary "
            "is prepended to the retained poll observation."
        )
    if not additions:
        return notes
    addition = "\n\n".join(additions)
    return f"{notes}\n\n{addition}" if notes else addition


def _sum_optional_int(values: Iterable[int | None]) -> int | None:
    observations = list(values)
    if not observations or any(value is None for value in observations):
        return None
    return sum(value for value in observations if value is not None)


def _metric_extra_int(metrics: AtifMetrics, key: str) -> int | None:
    value = (metrics.extra or {}).get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _sum_optional_float(values: Iterable[float | None]) -> float | None:
    observations = list(values)
    if not observations or any(value is None for value in observations):
        return None
    return sum(value for value in observations if value is not None)


def build_atif_fanout_trajectory(
    *,
    session_id: str,
    sources: list[AtifRunSource],
) -> AtifTrajectory:
    """Build one Harbor artifact for a multi-model comparison run."""

    children = [build_atif_trajectory(source) for source in sources]
    calls = [
        AtifToolCall(
            tool_call_id=f"model_{index}",
            function_name="fast_agent_model",
            arguments={"agent_name": source.agent_name, "model_name": source.model_name},
        )
        for index, source in enumerate(sources, start=1)
    ]
    results = [
        AtifObservationResult(
            source_call_id=call.tool_call_id,
            content=child.steps[-1].message,
            subagent_trajectory_ref=[
                AtifSubagentTrajectoryRef(
                    trajectory_id=child.trajectory_id,
                    session_id=session_id,
                    extra={"agent_name": source.agent_name},
                )
            ],
        )
        for call, child, source in zip(calls, children, sources, strict=True)
    ]
    first_message = sources[0].history[0]
    root = AtifTrajectory(
        session_id=session_id,
        trajectory_id=f"traj_{uuid.uuid4().hex}",
        agent=AtifAgent(
            name="fast-agent",
            version=_package_version(),
            extra={"mode": "multi_model", "models": [source.model_name for source in sources]},
        ),
        steps=[
            AtifStep(
                step_id=1,
                source="user",
                message=_atif_content(list(first_message.content)),
                timestamp=(
                    first_message.timestamp.isoformat().replace("+00:00", "Z")
                    if first_message.timestamp is not None
                    else None
                ),
            ),
            AtifStep(
                step_id=2,
                source="agent",
                message="Dispatched the request to multiple model routes.",
                tool_calls=calls,
                observation=AtifObservation(results=results),
                llm_call_count=0,
                extra={"dispatch": "multi_model"},
            ),
        ],
        final_metrics=AtifFinalMetrics(
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_cached_tokens=0,
            total_cost_usd=0.0,
            total_steps=2,
            extra={
                "total_reasoning_tokens": 0,
                "total_tool_use_tokens": 0,
            },
        ),
        subagent_trajectories=children,
    )
    _include_subagent_metrics(root, children)
    return AtifTrajectory.model_validate(root.model_dump())


def _embed_subagent_trajectories(
    root: AtifTrajectory,
    source: AtifRunSource,
) -> None:
    directory = source.child_trajectory_dir
    if directory is None or not directory.is_dir():
        return
    embedded: list[AtifTrajectory] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("session_id") != source.session_id:
            continue
        parent_agent_name = payload.get("parent_agent_name")
        if isinstance(parent_agent_name, str) and parent_agent_name != source.agent_name:
            continue
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            continue
        messages = [PromptMessageExtended.model_validate(message) for message in raw_messages]
        child = build_atif_trajectory(
            AtifRunSource(
                session_id=source.session_id,
                agent_name=str(payload.get("agent_name") or "subagent"),
                model_name=None,
                provider=None,
                history=messages,
                message_timestamps=tuple(message.timestamp for message in messages),
            )
        )
        trajectory_id = payload.get("trajectory_id")
        if isinstance(trajectory_id, str) and trajectory_id:
            child.trajectory_id = trajectory_id
        child.agent.name = str(payload.get("agent_name") or "subagent")
        child.extra = {
            key: value
            for key, value in {
                "parent_agent_name": payload.get("parent_agent_name"),
                "parent_tool_call_id": payload.get("parent_tool_call_id"),
                "tool_name": payload.get("tool_name"),
                "use_history": payload.get("use_history"),
                "started_at": payload.get("started_at"),
                "completed_at": payload.get("completed_at"),
            }.items()
            if value is not None
        }
        usage_summary = payload.get("usage_summary")
        if isinstance(usage_summary, dict):
            child.final_metrics = _final_metrics_from_usage_summary(
                usage_summary,
                total_steps=len(child.steps),
            )
        embedded.append(child)
        parent_call_id = payload.get("parent_tool_call_id")
        if isinstance(parent_call_id, str) and child.trajectory_id is not None:
            _attach_subagent_ref(root, parent_call_id, child)
    if embedded:
        root.subagent_trajectories = embedded
        _include_subagent_metrics(root, embedded)


def _final_metrics_from_usage_summary(
    summary: dict[object, object],
    *,
    total_steps: int,
) -> AtifFinalMetrics:
    usage = UsageSummary.model_validate(summary)
    return AtifFinalMetrics(
        total_prompt_tokens=usage.prompt.total,
        total_completion_tokens=usage.completion.total,
        total_cached_tokens=usage.prompt.cache_read,
        total_steps=total_steps,
        extra={
            key: value
            for key, value in {
                "total_reasoning_tokens": usage.completion.reasoning,
                "total_tool_use_tokens": usage.prompt.tool_use,
            }.items()
            if value is not None
        }
    )


def _include_subagent_metrics(
    root: AtifTrajectory,
    embedded: list[AtifTrajectory],
) -> None:
    root_metrics = root.final_metrics or AtifFinalMetrics(total_steps=len(root.steps))
    child_metrics = [child.final_metrics for child in embedded if child.final_metrics is not None]
    root_prompt_tokens = root_metrics.total_prompt_tokens
    root_completion_tokens = root_metrics.total_completion_tokens
    root_cached_tokens = root_metrics.total_cached_tokens
    root_cost_usd = root_metrics.total_cost_usd
    root_reasoning_tokens = _final_metric_extra_int(
        root_metrics,
        "total_reasoning_tokens",
    )
    root_tool_use_tokens = _final_metric_extra_int(
        root_metrics,
        "total_tool_use_tokens",
    )
    subagent_prompt_tokens = _sum_optional_int(
        item.total_prompt_tokens for item in child_metrics
    )
    subagent_completion_tokens = _sum_optional_int(
        item.total_completion_tokens for item in child_metrics
    )
    subagent_cached_tokens = _sum_optional_int(
        item.total_cached_tokens for item in child_metrics
    )
    subagent_cost_usd = _sum_optional_float(
        item.total_cost_usd for item in child_metrics
    )
    subagent_reasoning_tokens = _sum_optional_int(
        _final_metric_extra_int(item, "total_reasoning_tokens")
        for item in child_metrics
    )
    subagent_tool_use_tokens = _sum_optional_int(
        _final_metric_extra_int(item, "total_tool_use_tokens")
        for item in child_metrics
    )
    root_metrics.total_prompt_tokens = _sum_optional_int(
        (root_prompt_tokens, subagent_prompt_tokens)
    )
    root_metrics.total_completion_tokens = _sum_optional_int(
        (root_completion_tokens, subagent_completion_tokens)
    )
    root_metrics.total_cached_tokens = _sum_optional_int(
        (root_cached_tokens, subagent_cached_tokens)
    )
    root_metrics.total_cost_usd = _sum_optional_float(
        (root_cost_usd, subagent_cost_usd)
    )
    total_reasoning_tokens = _sum_optional_int(
        (root_reasoning_tokens, subagent_reasoning_tokens)
    )
    total_tool_use_tokens = _sum_optional_int(
        (root_tool_use_tokens, subagent_tool_use_tokens)
    )
    root_metrics.extra = {
        key: value
        for key, value in {
            **(root_metrics.extra or {}),
            "total_reasoning_tokens": total_reasoning_tokens,
            "total_tool_use_tokens": total_tool_use_tokens,
            "root_prompt_tokens": root_prompt_tokens,
            "root_completion_tokens": root_completion_tokens,
            "subagent_prompt_tokens": subagent_prompt_tokens,
            "subagent_completion_tokens": subagent_completion_tokens,
        }.items()
        if value is not None
    }
    root.final_metrics = root_metrics


def _final_metric_extra_int(metrics: AtifFinalMetrics, key: str) -> int | None:
    value = (metrics.extra or {}).get(key)
    return value if type(value) is int else None


def _attach_subagent_ref(
    root: AtifTrajectory,
    parent_call_id: str,
    child: AtifTrajectory,
) -> None:
    reference = AtifSubagentTrajectoryRef(
        trajectory_id=child.trajectory_id,
        session_id=root.session_id,
        extra={"agent_name": child.agent.name},
    )
    for step in root.steps:
        if parent_call_id not in {call.tool_call_id for call in step.tool_calls or []}:
            continue
        if step.observation is None:
            step.observation = AtifObservation(results=[])
        for result in step.observation.results:
            if result.source_call_id == parent_call_id:
                refs = result.subagent_trajectory_ref or []
                refs.append(reference)
                result.subagent_trajectory_ref = refs
                return
        step.observation.results.append(
            AtifObservationResult(
                source_call_id=parent_call_id,
                subagent_trajectory_ref=[reference],
            )
        )
        return


class AtifTraceWriter:
    def __init__(self, sanitizer: TraceSanitizer | None = None) -> None:
        self._sanitizer = sanitizer

    def write(self, resolved: ResolvedSessionExport, output_path: Path) -> ExportResult:
        snapshot_agent = resolved.snapshot.continuation.agents[resolved.agent_name]
        trajectory = build_atif_trajectory(
            AtifRunSource(
                session_id=resolved.session_id,
                agent_name=resolved.agent_name,
                model_name=snapshot_agent.model_spec or snapshot_agent.model,
                provider=snapshot_agent.provider,
                history=resolved.history,
                message_timestamps=resolved.message_timestamps,
                child_trajectory_dir=resolved.session_dir / "trajectories",
                notes=(
                    "Converted from persisted Fast-Agent history; execution-only "
                    "metadata unavailable in older sessions may be omitted."
                ),
                system_prompt=snapshot_agent.resolved_prompt,
            )
        )
        redaction = _sanitize_trajectory(trajectory, self._sanitizer)
        write_atif_trajectory(trajectory, output_path)
        return ExportResult(
            session_id=resolved.session_id,
            agent_name=resolved.agent_name,
            format="atif",
            output_path=output_path,
            record_count=len(trajectory.steps),
            redaction=redaction,
        )


def _sanitize_text(
    value: str,
    sanitizer: TraceSanitizer,
    redactions: RedactionAccumulator,
) -> str:
    sanitized = sanitizer.sanitize_text(value)
    redactions.add(sanitized.spans)
    return sanitized.text


def _sanitize_value(
    value: object,
    sanitizer: TraceSanitizer,
    redactions: RedactionAccumulator,
) -> object:
    if isinstance(value, str):
        return _sanitize_text(value, sanitizer, redactions)
    if isinstance(value, list):
        return [_sanitize_value(item, sanitizer, redactions) for item in value]
    if isinstance(value, dict):
        return {
            key: _sanitize_value(item, sanitizer, redactions)
            for key, item in value.items()
        }
    return value


def _sanitize_content(
    content: AtifContent,
    sanitizer: TraceSanitizer,
    redactions: RedactionAccumulator,
) -> AtifContent:
    if isinstance(content, str):
        return _sanitize_text(content, sanitizer, redactions)
    for part in content:
        if part.type == "text" and part.text is not None:
            part.text = _sanitize_text(part.text, sanitizer, redactions)
    return content


def _sanitize_trajectory(
    trajectory: AtifTrajectory,
    sanitizer: TraceSanitizer | None,
) -> RedactionSummary | None:
    if sanitizer is None:
        return None
    redactions = RedactionAccumulator(model=sanitizer.model_info)
    for step in trajectory.steps:
        step.message = _sanitize_content(step.message, sanitizer, redactions)
        if step.reasoning_content is not None:
            step.reasoning_content = _sanitize_text(
                step.reasoning_content, sanitizer, redactions
            )
        for call in step.tool_calls or []:
            call.arguments = {
                key: _sanitize_value(value, sanitizer, redactions)
                for key, value in call.arguments.items()
            }
        for result in step.observation.results if step.observation else []:
            if result.content is not None:
                result.content = _sanitize_content(result.content, sanitizer, redactions)
    for child in trajectory.subagent_trajectories or []:
        child_summary = _sanitize_trajectory(child, sanitizer)
        if child_summary is not None:
            for label, count in child_summary.by_label.items():
                redactions.total += count
                redactions.by_label[label] = redactions.by_label.get(label, 0) + count
    return redactions.summary()


def write_atif_trajectory(trajectory: AtifTrajectory, output_path: Path) -> None:
    """Atomically write one self-contained ATIF document."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(trajectory.to_json_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
    temporary_path.replace(output_path)
