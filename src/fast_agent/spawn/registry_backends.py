"""Storage backends for SpawnRegistry and TeamSessionStore.

Provides pluggable persistence for the spawn registry:
- JsonFileBackend: JSON file (default for standalone fast-agent)
- SqliteBackend:   SQLite DB (activated via SPAWN_REGISTRY_DB env var)

Also provides TeamSessionStore — SQLite-only storage for TeamSession state.
TeamSession requires SPAWN_REGISTRY_DB; raises RuntimeError if not configured.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RegistryBackend(ABC):
    """Abstract storage backend for spawn registry data."""

    @abstractmethod
    def load(self) -> dict[str, dict[str, Any]]:
        """Load all records from storage. Returns {run_id: record_dict}."""
        ...

    @abstractmethod
    def save(self, data: dict[str, dict[str, Any]]) -> None:
        """Persist all records to storage."""
        ...


class JsonFileBackend(RegistryBackend):
    """JSON file backend — default for standalone fast-agent."""

    def __init__(self, file_path: str | Path) -> None:
        self._file = Path(file_path)
        self._file.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict[str, Any]]:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def save(self, data: dict[str, dict[str, Any]]) -> None:
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
        tmp.rename(self._file)


class SqliteBackend(RegistryBackend):
    """SQLite backend — cross-process durable persistence.

    Activated by setting ``SPAWN_REGISTRY_DB`` env var to the DB file path.
    Uses stdlib ``sqlite3`` (no external deps). Schema::

        CREATE TABLE spawn_registry (
            run_id   TEXT PRIMARY KEY,
            data_json TEXT NOT NULL
        )

    WAL journal mode is enabled for concurrent read/write from
    multiple processes (parent server + spawned agent subprocesses).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_table(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS spawn_registry (
                        run_id    TEXT PRIMARY KEY,
                        data_json TEXT NOT NULL
                    )"""
                )
        except Exception as e:
            logger.warning("SqliteBackend: failed to init table: %s", e)

    def load(self) -> dict[str, dict[str, Any]]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT run_id, data_json FROM spawn_registry"
                ).fetchall()
                result: dict[str, dict[str, Any]] = {}
                for run_id, data_json in rows:
                    try:
                        result[run_id] = json.loads(data_json)
                    except json.JSONDecodeError:
                        continue
                return result
        except Exception as e:
            logger.warning("SqliteBackend.load failed: %s", e)
            return {}

    def save(self, data: dict[str, dict[str, Any]]) -> None:
        try:
            with self._connect() as conn:
                # Get existing run_ids
                existing = {
                    r[0]
                    for r in conn.execute(
                        "SELECT run_id FROM spawn_registry"
                    ).fetchall()
                }
                new_keys = set(data.keys())

                # Delete removed entries
                removed = existing - new_keys
                if removed:
                    placeholders = ",".join("?" * len(removed))
                    conn.execute(
                        f"DELETE FROM spawn_registry WHERE run_id IN ({placeholders})",
                        list(removed),
                    )

                # Merge-upsert: read existing record, overlay new data, write back.
                # This preserves fields written by other processes (e.g. bridge
                # writes runtime_config, subprocess writes lifecycle/team_name).
                for run_id, record in data.items():
                    if run_id in existing:
                        row = conn.execute(
                            "SELECT data_json FROM spawn_registry WHERE run_id = ?",
                            (run_id,),
                        ).fetchone()
                        if row:
                            merged = json.loads(row[0])
                            merged.update(record)
                            record = merged
                    conn.execute(
                        "INSERT OR REPLACE INTO spawn_registry (run_id, data_json) VALUES (?, ?)",
                        (run_id, json.dumps(record, ensure_ascii=False)),
                    )
        except Exception as e:
            logger.warning("SqliteBackend.save failed: %s", e)


def create_backend(registry_file: str | Path) -> RegistryBackend:
    """Factory: create the appropriate backend.

    If ``SPAWN_REGISTRY_DB`` env var is set → ``SqliteBackend(db_path)``.
    Otherwise → ``JsonFileBackend(registry_file)``.
    """
    db_path = os.environ.get("SPAWN_REGISTRY_DB")
    if db_path:
        return SqliteBackend(db_path)
    return JsonFileBackend(registry_file)


class TeamSessionStore:
    """SQLite-backed store for TeamSession state.

    Uses the same DB as spawn_registry (SPAWN_REGISTRY_DB env var).
    Schema::

        CREATE TABLE team_sessions (
            session_id TEXT PRIMARY KEY,
            data_json  TEXT NOT NULL
        )

    WAL mode inherited from the shared connection settings.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS team_sessions (
                    session_id TEXT PRIMARY KEY,
                    data_json  TEXT NOT NULL
                )"""
            )

    def upsert(self, session_id: str, data: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO team_sessions (session_id, data_json) VALUES (?, ?)",
                (session_id, json.dumps(data, ensure_ascii=False)),
            )

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM team_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_all(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT data_json FROM team_sessions ORDER BY session_id DESC"
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def delete(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM team_sessions WHERE session_id = ?", (session_id,)
            )


def create_team_store() -> TeamSessionStore:
    """Create TeamSessionStore from SPAWN_REGISTRY_DB env var.

    Raises RuntimeError if SPAWN_REGISTRY_DB is not set — TeamSession
    persistence requires SQLite; there is no JSON fallback.
    """
    db_path = os.environ.get("SPAWN_REGISTRY_DB")
    if not db_path:
        raise RuntimeError(
            "SPAWN_REGISTRY_DB env var is not set. "
            "TeamSession persistence requires SQLite."
        )
    return TeamSessionStore(db_path)
