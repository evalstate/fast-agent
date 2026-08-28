"""Process-safe persisted session locks."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, cast

from filelock import FileLock, Timeout


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lock_key(session_id: str) -> str:
    readable = "".join(char if char.isalnum() or char in "-_" else "_" for char in session_id)
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:12]
    return f"{readable[:48] or 'session'}-{digest}"


@dataclass(frozen=True, slots=True)
class SessionOwner:
    host: str
    pid: int
    started_at: str
    acquired_at: str
    token: str
    surface: str | None = None

    @classmethod
    def from_dict(cls, payload: object) -> SessionOwner | None:
        if not isinstance(payload, dict):
            return None
        data = cast("dict[str, object]", payload)
        host = data.get("host")
        pid = data.get("pid")
        started_at = data.get("started_at")
        acquired_at = data.get("acquired_at")
        token = data.get("token")
        surface = data.get("surface")
        if (
            not isinstance(host, str)
            or not isinstance(pid, int)
            or isinstance(pid, bool)
            or not isinstance(started_at, str)
            or not isinstance(acquired_at, str)
            or not isinstance(token, str)
            or (surface is not None and not isinstance(surface, str))
        ):
            return None
        return cls(
            host=host,
            pid=pid,
            started_at=started_at,
            acquired_at=acquired_at,
            token=token,
            surface=surface,
        )

    def describe(self) -> str:
        surface = f", surface {self.surface}" if self.surface else ""
        return f"host {self.host}, pid {self.pid}{surface}, acquired {self.acquired_at}"


class SessionBusyError(RuntimeError):
    """Raised when a persisted session already has an exclusive owner."""

    def __init__(self, session_id: str, owner: SessionOwner | None) -> None:
        self.session_id = session_id
        self.owner = owner
        owner_text = owner.describe() if owner is not None else "owner details unavailable"
        super().__init__(
            f"Session '{session_id}' is active ({owner_text}). Wait for or close that process, "
            f"or fork its latest committed checkpoint with "
            f"'fast-agent session fork {session_id}'."
        )


class SessionCheckpointBusyError(RuntimeError):
    """Raised when a committed session checkpoint cannot be locked."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session '{session_id}' checkpoint is busy; retry after the save completes."
        )


@dataclass(slots=True)
class SessionOwnerLease:
    session_id: str
    owner: SessionOwner
    lock: FileLock
    metadata_path: Path
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        try:
            current = _read_owner(self.metadata_path)
            if current is not None and current.token == self.owner.token:
                try:
                    self.metadata_path.unlink(missing_ok=True)
                except OSError:
                    pass
        finally:
            self.lock.release()
            self._released = True


class SessionLockStore:
    """Own lock paths outside persisted session directories."""

    def __init__(self, sessions_dir: Path) -> None:
        self.directory = sessions_dir / ".locks"
        self.directory.mkdir(parents=True, exist_ok=True)

    def acquire_owner(
        self,
        session_id: str,
        *,
        started_at: str,
        surface: str | None,
        token: str,
    ) -> SessionOwnerLease:
        key = _lock_key(session_id)
        lock = FileLock(self.directory / f"{key}.owner.lock", timeout=0, thread_local=False)
        metadata_path = self.directory / f"{key}.owner.json"
        try:
            lock.acquire()
        except Timeout as exc:
            raise SessionBusyError(session_id, _read_owner(metadata_path)) from exc

        owner = SessionOwner(
            host=socket.gethostname(),
            pid=os.getpid(),
            started_at=started_at,
            acquired_at=_utc_now(),
            surface=surface,
            token=token,
        )
        try:
            _atomic_write_json(metadata_path, asdict(owner))
        except BaseException:
            lock.release()
            raise
        return SessionOwnerLease(session_id, owner, lock, metadata_path)

    def auxiliary(self, session_id: str, role: str, *, timeout: float = 0) -> FileLock:
        """Return a transient lock kept outside the session directory."""
        key = _lock_key(session_id)
        return FileLock(
            self.directory / f"{key}.{role}.lock",
            timeout=timeout,
            thread_local=False,
        )

    @contextmanager
    def checkpoint(self, session_id: str, *, timeout: float = 0) -> Iterator[None]:
        lock = self.auxiliary(session_id, "checkpoint", timeout=timeout)
        try:
            lock.acquire()
        except Timeout as exc:
            raise SessionCheckpointBusyError(session_id) from exc
        try:
            yield
        finally:
            lock.release()


def _read_owner(path: Path) -> SessionOwner | None:
    try:
        return SessionOwner.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


__all__ = [
    "SessionBusyError",
    "SessionCheckpointBusyError",
    "SessionLockStore",
    "SessionOwner",
    "SessionOwnerLease",
]
