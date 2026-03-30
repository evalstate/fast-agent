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

    def update_config(self, meeting_id: str, config: dict) -> None:
        """Overwrite the meeting config."""
        ...

    def update_state(self, meeting_id: str, state: dict) -> None:
        """Overwrite the meeting state."""
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

    def update_config(self, meeting_id: str, config: dict) -> None:
        self._write_json(
            self._meeting_dir(meeting_id) / "config.json", config
        )

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
