"""Durable identities for nested subagent runs."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from fast_agent.session.session_manager import Session
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
    status: SessionExecutionStatus
    started_at: datetime | None
    completed_at: datetime | None


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
    snapshot = child.load_snapshot()
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
    return SubagentRun(
        alias=alias,
        ordinal=ordinal,
        child_session_id=child.info.name,
        parent_agent_name=link.parent_agent_name,
        label=label if isinstance(label, str) else None,
        task_preview=task_preview if isinstance(task_preview, str) else None,
        status=snapshot.execution.status,
        started_at=snapshot.execution.started_at,
        completed_at=snapshot.execution.completed_at,
    )
