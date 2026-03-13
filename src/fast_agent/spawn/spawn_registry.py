"""Spawn Registry — track lifecycle and status of spawned agents.

File-based JSON persistence for cross-process visibility.
Three lifecycle models: persistent, resumable, oneshot.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {"completed", "error", "timeout", "cancelled", "killed"}


class Lifecycle(Enum):
    """Agent lifecycle model."""

    PERSISTENT = "persistent"  # Stays alive, manual cleanup
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
    task: str = ""
    status: str = SpawnStatus.RUNNING.value
    lifecycle: str = Lifecycle.ONESHOT.value
    cleanup: str = CleanupPolicy.DELETE.value
    session_id: str = ""
    pid: int | None = None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
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
        return cls(**{k: v for k, v in data.items() if k in known})


class SpawnRegistry:
    """File-based registry for spawned agents.

    Stores to ``<project_dir>/.runtime/state/spawn_registry.json``
    for cross-process visibility.
    """

    def __init__(self, registry_file: str | Path) -> None:
        self._file = Path(registry_file)
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._file.exists():
            try:
                self._data = json.loads(self._file.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), "utf-8")
        tmp.rename(self._file)

    def register(self, record: SpawnRecord) -> None:
        self._data[record.run_id] = record.to_dict()
        self._save()

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
            del self._data[run_id]
            self._save()
            return True
        return False

    def cleanup_oneshots(self) -> list[str]:
        self._load()
        to_remove = []
        for run_id, d in self._data.items():
            rec = SpawnRecord.from_dict(d)
            if rec.should_auto_cleanup:
                to_remove.append(run_id)
        for run_id in to_remove:
            del self._data[run_id]
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
