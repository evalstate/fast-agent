"""Durable identities for nested subagent runs."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from fast_agent.constants import FAST_AGENT_SYNTHETIC_FINAL_CHANNEL, FAST_AGENT_USAGE
from fast_agent.llm.usage_tracking import TurnUsage, UsageLedger, UsageReport, UserTurnUsage

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from fast_agent.session.session_manager import Session, SessionManager
    from fast_agent.session.snapshot import SessionExecutionStatus

SUBAGENT_ALIAS_KEY = "subagent_alias"
SUBAGENT_ORDINAL_KEY = "subagent_ordinal"
SUBAGENT_LABEL_KEY = "subagent_label"
SUBAGENT_TASK_PREVIEW_KEY = "subagent_task_preview"
SUBAGENT_ALIAS_SLUG_MAX_LENGTH = 32
SUBAGENT_TASK_PREVIEW_MAX_LENGTH = 80

_UNSAFE_ALIAS_CHARS = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class SubagentRun:
    alias: str
    ordinal: int
    child_session_id: str
    parent_agent_name: str
    label: str | None
    task_preview: str | None
    status: SessionExecutionStatus | Literal["interrupted"]
    started_at: datetime | None
    completed_at: datetime | None
    tool_name: Literal["subagent", "hc"] = "subagent"
    hc_terminal_id: str | None = None
    hc_name: str | None = None


@dataclass(frozen=True, slots=True)
class HcChildUsageRollup:
    """Aggregated persisted usage across ``hc`` child sessions for one parent agent."""

    parent_agent_name: str
    child_count: int
    model: str
    input_tokens: int | None
    cache_read_tokens: int | None
    output_tokens: int | None
    provider_attempts: int
    tool_calls: int
    cost_usd: float | None = None


def subagent_alias_slug(*, label: str | None, task: str) -> str:
    """Return a bounded, path-safe alias slug."""
    source = label or task
    ascii_source = (
        unicodedata.normalize("NFKD", source).encode("ascii", "ignore").decode("ascii").lower()
    )
    slug = _UNSAFE_ALIAS_CHARS.sub("_", ascii_source).strip("_")
    if len(slug) > SUBAGENT_ALIAS_SLUG_MAX_LENGTH:
        slug = slug[:SUBAGENT_ALIAS_SLUG_MAX_LENGTH].rstrip("_")
        if "_" in slug:
            slug = slug.rsplit("_", 1)[0]
    return slug or "subagent"


def format_subagent_alias(ordinal: int, slug: str) -> str:
    return f"{ordinal:02d}_{slug}"


def subagent_task_preview(task: str) -> str:
    return " ".join(task.split())[:SUBAGENT_TASK_PREVIEW_MAX_LENGTH]


def subagent_run_from_session(parent: "Session", child: "Session") -> SubagentRun | None:
    snapshot = child.load_snapshot_unlocked()
    link = snapshot.execution.child_link
    if link is None or link.parent_session_id != parent.info.name:
        return None

    metadata = snapshot.metadata.extras
    alias = metadata.get(SUBAGENT_ALIAS_KEY)
    ordinal = metadata.get(SUBAGENT_ORDINAL_KEY)
    if not isinstance(alias, str) or not isinstance(ordinal, int):
        alias = f"legacy_{child.info.name}"
        ordinal = 0

    label = metadata.get(SUBAGENT_LABEL_KEY)
    task_preview = metadata.get(SUBAGENT_TASK_PREVIEW_KEY)
    status: SessionExecutionStatus | Literal["interrupted"] = snapshot.execution.status
    if link.tool_name == "hc" and child.manager is not None:
        owned = child.manager.is_session_owned(child.info.name)
        if owned and status == "completed":
            status = "running"
        elif not owned and status == "running":
            status = "interrupted"

    return SubagentRun(
        alias=alias,
        ordinal=ordinal,
        child_session_id=child.info.name,
        parent_agent_name=link.parent_agent_name,
        label=label if isinstance(label, str) else None,
        task_preview=task_preview if isinstance(task_preview, str) else None,
        status=status,
        started_at=snapshot.execution.started_at,
        completed_at=snapshot.execution.completed_at,
        tool_name=link.tool_name,
        hc_terminal_id=link.hc_terminal_id,
        hc_name=link.hc_name,
    )


def _iter_hc_descendant_snapshots(
    manager: "SessionManager",
    parent: "Session",
    *,
    root_parent_agent_name: str | None = None,
) -> list[tuple[str, "Session"]]:
    """Collect ``(root_parent_agent_name, child)`` pairs for ``hc`` descendants only."""
    collected: list[tuple[str, Session]] = []
    for child in manager.list_child_sessions(parent):
        snapshot = child.load_snapshot_unlocked()
        link = snapshot.execution.child_link
        if link is None or link.tool_name != "hc" or link.parent_session_id != parent.info.name:
            continue
        owner_agent = root_parent_agent_name or link.parent_agent_name
        collected.append((owner_agent, child))
        collected.extend(
            _iter_hc_descendant_snapshots(
                manager,
                child,
                root_parent_agent_name=owner_agent,
            )
        )
    return collected


def collect_hc_child_usage_rollups(
    manager: "SessionManager | None",
    *,
    parent: "Session | None" = None,
) -> tuple[HcChildUsageRollup, ...]:
    """Aggregate persisted usage across ``hc`` child sessions without locking children.

    In-process ``subagent`` children are intentionally excluded because their
    usage is already merged into the parent agent's ``subagent_usage_accumulator``.
    """
    if manager is None:
        return ()
    target_parent = parent or manager.current_session
    if target_parent is None:
        return ()

    grouped: dict[str, list[Session]] = {}
    for owner_agent, child in _iter_hc_descendant_snapshots(manager, target_parent):
        grouped.setdefault(owner_agent, []).append(child)

    rollups: list[HcChildUsageRollup] = []
    for parent_agent_name, children in grouped.items():
        child_count = 0
        input_tokens = 0
        input_complete = True
        output_tokens = 0
        output_complete = True
        cache_read_tokens = 0
        cache_read_complete = True
        provider_attempts = 0
        tool_calls = 0
        cost_usd = 0.0
        cost_complete = True
        models: set[str] = set()

        for child in children:
            usage = child.load_snapshot_unlocked().analysis.usage_summary
            if usage is None:
                continue
            attempts = usage.provider_attempts if usage.provider_attempts is not None else 1
            if attempts <= 0:
                continue
            child_count += 1
            provider_attempts += attempts
            tool_calls += usage.tool_calls or 0
            if usage.prompt_tokens is None:
                input_complete = False
            else:
                input_tokens += usage.prompt_tokens
            if usage.completion_tokens is None:
                output_complete = False
            else:
                output_tokens += usage.completion_tokens
            if usage.cache_read_tokens is None:
                cache_read_complete = False
            else:
                cache_read_tokens += usage.cache_read_tokens
            if usage.cost_usd is None:
                cost_complete = False
            else:
                cost_usd += usage.cost_usd
            if usage.model:
                models.add(usage.model)

        if child_count <= 0:
            continue

        if len(models) == 1:
            resolved_model = next(iter(models))
        elif len(models) > 1:
            resolved_model = "multiple"
        else:
            resolved_model = "unknown"

        rollups.append(
            HcChildUsageRollup(
                parent_agent_name=parent_agent_name,
                child_count=child_count,
                model=resolved_model,
                input_tokens=input_tokens if input_complete else None,
                cache_read_tokens=cache_read_tokens if cache_read_complete else None,
                output_tokens=output_tokens if output_complete else None,
                provider_attempts=provider_attempts,
                tool_calls=tool_calls,
                cost_usd=cost_usd if cost_complete else None,
            )
        )

    return tuple(rollups)


_USAGE_SCHEMA_V2 = "fast-agent.usage/v2"


def _load_history_usage_attempts(history_path: "Path") -> tuple[TurnUsage, ...]:
    """Extract validated ``TurnUsage`` provider attempts from one persisted history file."""
    try:
        raw = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return ()
    if not isinstance(raw, dict):
        return ()
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return ()

    attempts: list[TurnUsage] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        channels = message.get("channels")
        if not isinstance(channels, dict):
            continue
        if FAST_AGENT_SYNTHETIC_FINAL_CHANNEL in channels:
            continue
        blocks = channels.get(FAST_AGENT_USAGE)
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if not isinstance(text, str):
                continue
            try:
                payload = json.loads(text)
            except ValueError:
                continue
            if not isinstance(payload, dict) or payload.get("schema") != _USAGE_SCHEMA_V2:
                continue
            try:
                report = UsageReport.model_validate(payload)
            except (ValidationError, TypeError, ValueError):
                continue
            attempts.extend(report.provider_attempts)
    return tuple(attempts)


def _hc_child_usage_ledger(child: "Session") -> UsageLedger | None:
    snapshot = child.load_snapshot_unlocked()
    link = snapshot.execution.child_link
    try:
        child_root = child.directory.resolve()
    except OSError:
        return None

    attempts: list[TurnUsage] = []
    seen_files: set[str] = set()
    for agent_snapshot in snapshot.continuation.agents.values():
        history_file = agent_snapshot.history_file
        if not history_file or history_file in seen_files:
            continue
        seen_files.add(history_file)
        try:
            candidate = (child.directory / history_file).resolve()
        except OSError:
            continue
        if not candidate.is_relative_to(child_root) or not candidate.is_file():
            continue
        attempts.extend(_load_history_usage_attempts(candidate))

    if not attempts:
        return None

    alias = snapshot.metadata.extras.get(SUBAGENT_ALIAS_KEY)
    name = (
        (link.hc_name if link is not None else None)
        or (alias if isinstance(alias, str) and alias else None)
        or (link.hc_terminal_id if link is not None else None)
        or child.info.name
    )
    return UsageLedger(label=f"hc:{name}", attempts=tuple(attempts))


def collect_hc_child_user_turn_usage(
    manager: "SessionManager | None",
    *,
    parent: "Session | None" = None,
) -> UserTurnUsage | None:
    """Collect persisted provider attempts across ``hc`` children as one ``UserTurnUsage``."""
    if manager is None:
        return None
    target_parent = parent or manager.current_session
    if target_parent is None:
        return None

    ordered_children = sorted(
        (child for _owner_agent, child in _iter_hc_descendant_snapshots(manager, target_parent)),
        key=lambda child: (child.info.created_at, child.info.name),
    )
    ledgers: list[UsageLedger] = []
    all_attempts: list[TurnUsage] = []
    for child in ordered_children:
        ledger = _hc_child_usage_ledger(child)
        if ledger is None:
            continue
        ledgers.append(ledger)
        all_attempts.extend(ledger.attempts)

    if not all_attempts:
        return None

    return UserTurnUsage(
        agent_name="hc",
        attempts=tuple(all_attempts),
        ledgers=tuple(ledgers),
    )


def with_hc_child_user_turn_usage(
    user_turn_usage: tuple[UserTurnUsage, ...],
    manager: "SessionManager | None",
    *,
    parent: "Session | None" = None,
) -> tuple[UserTurnUsage, ...]:
    """Append persisted ``hc`` child usage to plugin ``user_turn_usage`` when present."""
    hc_turn = collect_hc_child_user_turn_usage(manager, parent=parent)
    if hc_turn is None:
        return user_turn_usage
    return (*user_turn_usage, hc_turn)
