"""Pluggable storage backend for meeting data.

Default implementation: ``JsonFileMeetingStorage`` — file-based, ships with
the library.  Applications can override with SQLite, Redis, etc.

Usage (standalone — zero config)::

    storage = JsonFileMeetingStorage()
    storage.create_meeting("mtg_abc", config_dict, state_dict)

Usage (injected by host application)::

    from my_app import SqliteMeetingStorage
    from fast_agent.spawn.servers.meeting_room_server import configure_meeting_room
    configure_meeting_room(storage=SqliteMeetingStorage(db_path="..."))
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MeetingStorage(Protocol):
    """Abstract interface for meeting data persistence.

    Implementations must be **thread-safe** for concurrent access from
    multiple agent subprocesses.
    """

    def meeting_exists(self, meeting_id: str) -> bool:
        """Return True if a meeting with *meeting_id* exists."""
        ...

    def create_meeting(
        self, meeting_id: str, config: dict, state: dict
    ) -> None:
        """Persist a new meeting (config + initial state + empty transcript)."""
        ...

    def get_config(self, meeting_id: str) -> dict | None:
        """Return the meeting config dict, or ``None`` if not found."""
        ...

    def get_state(self, meeting_id: str) -> dict | None:
        """Return the current state dict, or ``None``."""
        ...

    def get_transcript(self, meeting_id: str) -> list[dict]:
        """Return the full transcript (list of turn entries)."""
        ...

    def update_state(self, meeting_id: str, state: dict) -> None:
        """Overwrite the meeting state.

        Note: ``config_json`` is intentionally write-once (set during
        ``create_meeting``). All MUTABLE meeting data — participants,
        max_rounds, current_turn, current_round, joined, ended, outcome,
        read_cursors — lives in ``state_json``. This single-bucket-for-
        mutable-data design eliminates the dual-write deadlock that
        existed when ``participants``/``max_rounds`` were updated through
        a separate ``update_config`` path inside ``acquire_lock``.
        """
        ...

    def append_transcript(self, meeting_id: str, entry: dict) -> None:
        """Append a single entry to the transcript."""
        ...

    def set_transcript(
        self, meeting_id: str, transcript: list[dict]
    ) -> None:
        """Replace the entire transcript."""
        ...

    def acquire_lock(self, meeting_id: str) -> Any:
        """Return a **context manager** that holds an exclusive lock."""
        ...

    def list_meeting_ids(self) -> list[str]:
        """Return all known meeting IDs."""
        ...


# ────────────────────────────────────────────────────────────────────
# Default JSON-file implementation — backward compatible, zero config
# ────────────────────────────────────────────────────────────────────


class JsonFileMeetingStorage:
    """File-based meeting storage using JSON files + ``fcntl`` locking.

    This is the **default** backend that ships with the library.
    It preserves the exact same on-disk layout as the original
    ``meeting_room_server`` implementation::

        {workspace_dir}/meetings/{meeting_id}/
            config.json
            state.json
            transcript.json
            .lock
            audit.log
    """

    def __init__(self, workspace_dir: str | None = None) -> None:
        if workspace_dir:
            self._workspace = Path(workspace_dir)
        else:
            ws = os.environ.get("TEAM_WORKSPACE", "")
            if not ws:
                ws = str(
                    Path.cwd()
                    / ".runtime"
                    / "cache"
                    / "tmp"
                    / "meeting-workspace"
                )
            self._workspace = Path(ws)

    # ── helpers ────────────────────────────────────────────────────

    def _meeting_dir(self, meeting_id: str) -> Path:
        return self._workspace / "meetings" / meeting_id

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | list[Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Protocol implementation ───────────────────────────────────

    def meeting_exists(self, meeting_id: str) -> bool:
        return self._meeting_dir(meeting_id).exists()

    def create_meeting(
        self, meeting_id: str, config: dict, state: dict
    ) -> None:
        d = self._meeting_dir(meeting_id)
        d.mkdir(parents=True, exist_ok=True)
        self._write_json(d / "config.json", config)
        self._write_json(d / "state.json", state)
        self._write_json(d / "transcript.json", [])

    def get_config(self, meeting_id: str) -> dict | None:
        d = self._meeting_dir(meeting_id)
        if not d.exists():
            return None
        raw = self._read_json(d / "config.json")
        return raw if isinstance(raw, dict) else {}

    def get_state(self, meeting_id: str) -> dict | None:
        d = self._meeting_dir(meeting_id)
        if not d.exists():
            return None
        raw = self._read_json(d / "state.json")
        return raw if isinstance(raw, dict) else {}

    def get_transcript(self, meeting_id: str) -> list[dict]:
        d = self._meeting_dir(meeting_id)
        if not d.exists():
            return []
        raw = self._read_json(d / "transcript.json")
        return raw if isinstance(raw, list) else []

    def update_state(self, meeting_id: str, state: dict) -> None:
        self._write_json(
            self._meeting_dir(meeting_id) / "state.json", state
        )

    def append_transcript(self, meeting_id: str, entry: dict) -> None:
        d = self._meeting_dir(meeting_id)
        transcript = self.get_transcript(meeting_id)
        transcript.append(entry)
        self._write_json(d / "transcript.json", transcript)

    def set_transcript(
        self, meeting_id: str, transcript: list[dict]
    ) -> None:
        self._write_json(
            self._meeting_dir(meeting_id) / "transcript.json", transcript
        )

    @contextmanager
    def acquire_lock(self, meeting_id: str):  # type: ignore[override]
        d = self._meeting_dir(meeting_id)
        d.mkdir(parents=True, exist_ok=True)
        lock_path = d / ".lock"
        lock_path.touch(exist_ok=True)
        fd = open(lock_path, "w")  # noqa: SIM115
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    def list_meeting_ids(self) -> list[str]:
        meetings_dir = self._workspace / "meetings"
        if not meetings_dir.exists():
            return []
        return [
            d.name
            for d in meetings_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ]


# ────────────────────────────────────────────────────────────────────
# SQLite implementation — shared DB, cross-process safe
# ────────────────────────────────────────────────────────────────────


class SqliteMeetingStorage:
    """SQLite-backed meeting storage using the shared Jarvis DB.

    Uses raw ``sqlite3`` (no ORM dependency) so it can run inside fast-agent
    MCP subprocesses without importing the host app's SQLAlchemy stack.

    Tables are created idempotently on first use.  WAL mode + busy_timeout
    ensure safe cross-process concurrent access.
    """

    _CREATE_TABLES = """
    CREATE TABLE IF NOT EXISTS meetings (
        meeting_id TEXT PRIMARY KEY,
        config_json TEXT NOT NULL,
        state_json TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS meeting_transcripts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meeting_id TEXT NOT NULL,
        turn INTEGER NOT NULL,
        round INTEGER NOT NULL,
        agent TEXT NOT NULL,
        message TEXT,
        type TEXT DEFAULT 'speak',
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_mt_meeting ON meeting_transcripts(meeting_id);
    CREATE TABLE IF NOT EXISTS meeting_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        meeting_id TEXT NOT NULL,
        data_json TEXT,
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_me_meeting ON meeting_events(meeting_id);
    CREATE INDEX IF NOT EXISTS idx_me_created ON meeting_events(created_at);
    """

    def __init__(self, db_path: str) -> None:
        import sqlite3
        import time as _time

        self._db_path = db_path
        self._time = _time
        self._sqlite3 = sqlite3

        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(self._CREATE_TABLES)
        conn.commit()
        conn.close()

    def _conn(self):
        """Open a fresh connection (thread/process safe)."""
        conn = self._sqlite3.connect(self._db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.row_factory = self._sqlite3.Row
        return conn

    # ── MeetingStorage protocol ───────────────────────────────────

    def meeting_exists(self, meeting_id: str) -> bool:
        conn = self._conn()
        try:
            r = conn.execute(
                "SELECT 1 FROM meetings WHERE meeting_id = ?", (meeting_id,)
            ).fetchone()
            return r is not None
        finally:
            conn.close()

    def create_meeting(
        self, meeting_id: str, config: dict, state: dict
    ) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO meetings (meeting_id, config_json, state_json, created_at) VALUES (?, ?, ?, ?)",
                (meeting_id, json.dumps(config, ensure_ascii=False),
                 json.dumps(state, ensure_ascii=False), self._time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_config(self, meeting_id: str) -> dict | None:
        conn = self._conn()
        try:
            r = conn.execute(
                "SELECT config_json FROM meetings WHERE meeting_id = ?",
                (meeting_id,),
            ).fetchone()
            return json.loads(r["config_json"]) if r else None
        finally:
            conn.close()

    def get_state(self, meeting_id: str, _conn=None) -> dict | None:
        conn = _conn or self._conn()
        try:
            r = conn.execute(
                "SELECT state_json FROM meetings WHERE meeting_id = ?",
                (meeting_id,),
            ).fetchone()
            return json.loads(r["state_json"]) if r else None
        finally:
            if not _conn:
                conn.close()

    def get_transcript(self, meeting_id: str, _conn=None) -> list[dict]:
        conn = _conn or self._conn()
        try:
            rows = conn.execute(
                "SELECT turn, round, agent, message, type, created_at "
                "FROM meeting_transcripts WHERE meeting_id = ? ORDER BY id",
                (meeting_id,),
            ).fetchall()
            return [
                {
                    "turn": r["turn"],
                    "round": r["round"],
                    "agent": r["agent"],
                    "message": r["message"],
                    "type": r["type"],
                    "timestamp": r["created_at"],
                }
                for r in rows
            ]
        finally:
            if not _conn:
                conn.close()

    def update_state(self, meeting_id: str, state: dict, _conn=None) -> None:
        conn = _conn or self._conn()
        try:
            conn.execute(
                "UPDATE meetings SET state_json = ? WHERE meeting_id = ?",
                (json.dumps(state, ensure_ascii=False), meeting_id),
            )
            if not _conn:
                conn.commit()
        finally:
            if not _conn:
                conn.close()

    def append_transcript(self, meeting_id: str, entry: dict, _conn=None) -> None:
        conn = _conn or self._conn()
        try:
            conn.execute(
                "INSERT INTO meeting_transcripts "
                "(meeting_id, turn, round, agent, message, type, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    meeting_id,
                    entry.get("turn", 0),
                    entry.get("round", 0),
                    entry.get("agent", ""),
                    entry.get("message", ""),
                    entry.get("type", entry.get("entry_type", "speak")),
                    self._time.time(),
                ),
            )
            if not _conn:
                conn.commit()
        finally:
            if not _conn:
                conn.close()

    def set_transcript(
        self, meeting_id: str, transcript: list[dict]
    ) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "DELETE FROM meeting_transcripts WHERE meeting_id = ?",
                (meeting_id,),
            )
            for entry in transcript:
                conn.execute(
                    "INSERT INTO meeting_transcripts "
                    "(meeting_id, turn, round, agent, message, type, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        meeting_id,
                        entry.get("turn", 0),
                        entry.get("round", 0),
                        entry.get("agent", ""),
                        entry.get("message", ""),
                        entry.get("type", "speak"),
                        entry.get("timestamp", self._time.time()),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def acquire_lock(self, meeting_id: str):
        """Yield a connection holding a RESERVED lock.

        Code inside the ``with`` block should pass ``_conn=conn`` to
        storage methods so they reuse this connection instead of
        opening new ones (which would deadlock against the lock).
        """
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def list_meeting_ids(self) -> list[str]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT meeting_id FROM meetings ORDER BY created_at DESC"
            ).fetchall()
            return [r["meeting_id"] for r in rows]
        finally:
            conn.close()

    # ── Cross-process event bus ───────────────────────────────────

    def emit_event(
        self, event_type: str, meeting_id: str, data: dict | None = None
    ) -> None:
        """Write an event row for the bridge to poll."""
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO meeting_events "
                "(event_type, meeting_id, data_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    event_type,
                    meeting_id,
                    json.dumps(data or {}, ensure_ascii=False),
                    self._time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def poll_events(self, after_id: int = 0) -> list[dict]:
        """Read new events since *after_id*."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT id, event_type, meeting_id, data_json, created_at "
                "FROM meeting_events WHERE id > ? ORDER BY id",
                (after_id,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "event_type": r["event_type"],
                    "meeting_id": r["meeting_id"],
                    "data": json.loads(r["data_json"]) if r["data_json"] else {},
                    "ts": r["created_at"],
                }
                for r in rows
            ]
        finally:
            conn.close()

