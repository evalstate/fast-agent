"""Spawn Registry — track lifecycle and status of spawned agents.

Pluggable storage via RegistryBackend abstraction:
  - JsonFileBackend  (default for standalone fast-agent)
  - SqliteBackend    (when SPAWN_REGISTRY_DB env var is set)

Three lifecycle models: persistent, resumable, oneshot.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .registry_backends import RegistryBackend, create_backend

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {"completed", "error", "timeout", "cancelled", "killed"}


class Lifecycle(Enum):
    """Agent lifecycle model.

    Only two modes after the 2026-05-20 merge: oneshot (auto-delete) and
    resumable (kept for resume). The legacy ``"persistent"`` value is no
    longer emitted by the spawner — entry points coerce it to ``"resumable"``
    (see ``spawn_and_run_isolated`` / ``spawn_and_run_background``), and
    SpawnRecord.from_dict() upgrades old DB rows on read.
    """

    RESUMABLE = "resumable"  # Can be stopped and resumed with context
    ONESHOT = "oneshot"  # Auto-delete after completion


class CleanupPolicy(Enum):
    """Cleanup policy after agent completion."""

    KEEP = "keep"  # Keep artifacts after completion
    DELETE = "delete"  # Auto-delete everything


class SpawnStatus(Enum):
    """Agent spawn status."""

    PENDING = "pending"
    RUNNING = "running"
    IDLE = "idle"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    KILLED = "killed"


@dataclass
class SpawnRecord:
    """Record of a spawned agent."""

    run_id: str = ""
    agent_name: str = ""
    role: str = ""
    team_name: str = ""  # Groups agents by team (e.g. "agile_team")
    task: str = ""
    status: str = SpawnStatus.RUNNING.value
    lifecycle: str = Lifecycle.ONESHOT.value
    cleanup: str = CleanupPolicy.DELETE.value
    session_id: str = ""
    pid: int | None = None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    # Updated by SpawnProgressBridge whenever the agent emits a thinking/
    # response/tool_call/tool_result event. Lets ``get_team_status``
    # distinguish "actively working" from "idle waiting on inbox" — the
    # visibility gap exposed by incident b61af7db (PM idled out after
    # observing identical ``running`` snapshots without knowing if the
    # team was making progress).
    last_active_at: float | None = None
    result: str = ""
    error: str = ""
    servers: list[str] = field(default_factory=list)
    original_config: dict[str, Any] = field(default_factory=dict)
    restart_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATES

    @property
    def should_auto_cleanup(self) -> bool:
        return (
            self.lifecycle == Lifecycle.ONESHOT.value
            and self.cleanup == CleanupPolicy.DELETE.value
            and self.is_terminal
        )

    @property
    def duration_seconds(self) -> float:
        if self.completed_at:
            return round(self.completed_at - self.started_at, 1)
        return round(time.time() - self.started_at, 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpawnRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        # Backward-compat: legacy DB rows had lifecycle="persistent". After
        # the 2026-05-20 merge persistent ≡ resumable everywhere — collapse
        # on read so downstream code can rely on the 2-value enum.
        if filtered.get("lifecycle") == "persistent":
            filtered["lifecycle"] = Lifecycle.RESUMABLE.value
        return cls(**filtered)


class SpawnRegistry:
    """Registry for spawned agents with pluggable storage backend.

    Storage is determined by the ``registry_backends.create_backend()``
    factory.  When ``SPAWN_REGISTRY_DB`` env var is set, data is stored
    in SQLite (shared across processes).  Otherwise, a JSON file is used.

    An optional ``on_change`` callback can be set to receive
    notifications on every write (register/update/remove).  Signature:
    ``(action: str, run_id: str, data: dict) -> None`` where *action*
    is one of ``"register"``, ``"update"``, ``"remove"``.
    """

    def __init__(
        self,
        registry_file: str | Path,
        on_change: Callable[[str, str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._backend: RegistryBackend = create_backend(registry_file)
        self._data: dict[str, dict[str, Any]] = {}
        self._on_change = on_change
        self._load()

    def _notify(self, action: str, run_id: str, data: dict[str, Any]) -> None:
        """Call on_change callback if set.  Never raises."""
        if self._on_change:
            try:
                self._on_change(action, run_id, data)
            except Exception:
                logger.warning(
                    "on_change callback failed: action=%s run_id=%s",
                    action, run_id, exc_info=True,
                )

    def _load(self) -> None:
        self._data = self._backend.load()

    def _save(self) -> None:
        self._backend.save(self._data)

    @staticmethod
    def _validate_for_registration(record: SpawnRecord) -> None:
        """Reject loose inputs that would silently fragment agent identity.

        Background — incident 2026-05-17: PM was registered twice in the
        running team, once as ``agent_name='Robin [PM]'`` (correct) and once
        as ``agent_name='pm'`` (just the role). The dashboard, which treats
        ``agent_name`` as the unique identity for grouping, rendered two
        separate cards for the same logical agent.

        Root cause was a defensive fallback in ``isolated_spawner`` —
        ``agent_name or role or "agent"`` — that silently substituted the
        role string when a caller forgot to pass ``agent_name`` to
        ``run_isolated_agent_background``. The mistake produced a successful
        register call instead of a loud error, so the bug only surfaced when
        a human noticed the duplicate card much later.

        This validator catches the bug class at the write boundary. For
        ad-hoc spawns (no ``team_name`` — generic ``_spawn_agent_background``
        MCP tool, one-off internal tasks) the fallback is legitimate and
        passes through unchanged. For team-managed spawns the human-readable
        identity must be present and distinct from the role.

        Why here and not in ``SpawnRecord.__post_init__``: ``from_dict()``
        (which loads existing rows from the DB) MUST be able to round-trip
        any historical row, including pre-fix orphans, without raising.
        Putting the guard in ``register()`` means "you cannot WRITE bad
        records going forward" without breaking the ability to READ legacy
        ones — which is the SSoT contract we actually want.
        """
        if not record.team_name:
            return  # ad-hoc spawn — fallback is legitimate
        if not record.agent_name:
            raise ValueError(
                f"Refuse to register team agent without agent_name "
                f"(team={record.team_name!r}, role={record.role!r}, "
                f"run_id={record.run_id!r}). Team-managed agents MUST be "
                f"registered under their team-assigned identity "
                f"(e.g. 'Robin [PM]') — silent fallback to role fragments "
                f"the dashboard's agent grouping."
            )
        if record.agent_name == record.role:
            raise ValueError(
                f"Refuse to register team agent whose agent_name equals "
                f"role ({record.agent_name!r}). team={record.team_name!r}, "
                f"run_id={record.run_id!r}. This indicates a caller "
                f"missed passing the distinct human-readable identity "
                f"(e.g. 'Robin [PM]') and fell back to the role string. "
                f"Fix the caller — do not loosen this check."
            )

    def register(self, record: SpawnRecord) -> None:
        self._validate_for_registration(record)
        data = record.to_dict()
        logger.warning(
            "[REGISTRY_DEBUG] register run_id=%s agent=%s lifecycle=%s team=%s backend=%s",
            record.run_id, record.agent_name, data.get("lifecycle"), data.get("team_name"),
            type(self._backend).__name__,
        )
        self._data[record.run_id] = data
        self._save()
        self._notify("register", record.run_id, self._data[record.run_id])

    def get(self, run_id: str) -> SpawnRecord | None:
        self._load()
        data = self._data.get(run_id)
        if data:
            return SpawnRecord.from_dict(data)
        return None

    def get_latest(self, run_id: str) -> SpawnRecord | None:
        """Follow the resume/restart chain and return the most current record.

        Given a root run_id, follows metadata.latest_resume_run_id and
        metadata.latest_restart_run_id links to find the final (leaf)
        record in the chain.
        """
        self._load()
        visited: set[str] = set()
        current_id = run_id

        while current_id and current_id not in visited:
            visited.add(current_id)
            data = self._data.get(current_id)
            if not data:
                break
            meta = data.get("metadata", {})
            next_id = meta.get("latest_resume_run_id") or meta.get("latest_restart_run_id")
            if next_id and next_id in self._data:
                current_id = next_id
            else:
                break

        data = self._data.get(current_id)
        return SpawnRecord.from_dict(data) if data else None

    def has_running_resume(self, agent_name: str) -> bool:
        """Check if this agent already has a running instance (guard double-resume)."""
        self._load()
        for d in self._data.values():
            if (
                d.get("agent_name") == agent_name
                and d.get("status") in ("running", "pending")
            ):
                return True
        return False

    def update_status(
        self,
        run_id: str,
        status: str | SpawnStatus,
        result: str = "",
        error: str = "",
    ) -> SpawnRecord | None:
        self._load()
        if run_id not in self._data:
            return None
        status_val = status.value if isinstance(status, SpawnStatus) else status
        self._data[run_id]["status"] = status_val
        if result:
            self._data[run_id]["result"] = result
        if error:
            self._data[run_id]["error"] = error
        if status_val in _TERMINAL_STATES:
            self._data[run_id]["completed_at"] = time.time()
        self._save()
        self._notify("update", run_id, self._data[run_id])
        return SpawnRecord.from_dict(self._data[run_id])

    def find_by_role(self, role: str) -> list[SpawnRecord]:
        self._load()
        return [SpawnRecord.from_dict(d) for d in self._data.values() if d.get("role") == role]

    def find_by_name(self, agent_name: str) -> SpawnRecord | None:
        """Find the latest agent record by agent_name (unique identity)."""
        self._load()
        matches = [
            SpawnRecord.from_dict(d)
            for d in self._data.values()
            if d.get("agent_name") == agent_name
        ]
        if not matches:
            return None
        # Return the most recent one
        return max(matches, key=lambda r: r.started_at)

    def find_by_session_id(self, session_id: str) -> SpawnRecord | None:
        self._load()
        for d in self._data.values():
            if d.get("session_id") == session_id:
                return SpawnRecord.from_dict(d)
        return None

    def list_active(self) -> list[SpawnRecord]:
        self._load()
        return [
            SpawnRecord.from_dict(d)
            for d in self._data.values()
            if d.get("status") in ("running", "idle", "pending")
        ]

    def list_all(self) -> list[SpawnRecord]:
        self._load()
        return [SpawnRecord.from_dict(d) for d in self._data.values()]

    def remove(self, run_id: str) -> bool:
        self._load()
        if run_id in self._data:
            removed = self._data.pop(run_id)
            self._save()
            self._notify("remove", run_id, removed)
            return True
        return False

    def find_by_team(self, team_name: str) -> list[SpawnRecord]:
        """Find all agents belonging to a team (exact match)."""
        self._load()
        return [
            SpawnRecord.from_dict(d)
            for d in self._data.values()
            if d.get("team_name") == team_name
        ]

    def remove_team(self, team_name: str) -> int:
        """Remove all registry entries for a team (exact match).
        Returns count removed."""
        self._load()
        to_remove = [
            rid for rid, d in self._data.items()
            if d.get("team_name") == team_name
        ]
        for rid in to_remove:
            removed = self._data.pop(rid)
            self._notify("remove", rid, removed)
        if to_remove:
            self._save()
        return len(to_remove)

    def cleanup_oneshots(self) -> list[str]:
        self._load()
        to_remove = []
        for run_id, d in self._data.items():
            rec = SpawnRecord.from_dict(d)
            if rec.should_auto_cleanup:
                to_remove.append(run_id)
        for run_id in to_remove:
            removed = self._data.pop(run_id)
            self._notify("remove", run_id, removed)
        if to_remove:
            self._save()
        return to_remove

    def to_summary(self) -> list[dict[str, str]]:
        self._load()
        return [
            {
                "run_id": d["run_id"],
                "agent_name": d.get("agent_name", ""),
                "role": d.get("role", ""),
                "status": d.get("status", ""),
                "lifecycle": d.get("lifecycle", ""),
                "task": d.get("task", "")[:80],
            }
            for d in self._data.values()
        ]
