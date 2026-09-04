"""Durable, POSIX-local shell process records and reader APIs."""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from filelock import FileLock

from fast_agent.constants import (
    DEFAULT_DURABLE_PROCESS_OUTPUT_RETENTION_BYTES,
    DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
    MAX_MANAGED_SHELL_PROCESSES,
    MAX_PROCESS_OUTPUT_QUERY_CHARS,
    MAX_RETAINED_DURABLE_PROCESS_RECORDS,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from typing import BinaryIO

_VERSION: Final = 1
_PROCESS_ID_PREFIX: Final = "process-"
_PROCESS_ID_LENGTH: Final = len(_PROCESS_ID_PREFIX) + 32
_MAX_OUTPUT_READ_BYTES: Final = 1024 * 1024
_OUTPUT_SEARCH_CHUNK_BYTES: Final = 64 * 1024
_NONTERMINAL_STATES: Final = frozenset({"created", "starting", "running", "stopping"})
_TERMINAL_STATES: Final = frozenset({"exited", "stopped", "unavailable"})

DurableProcessState = Literal[
    "created",
    "starting",
    "running",
    "stopping",
    "exited",
    "stopped",
    "unavailable",
]


class DurableProcessError(RuntimeError):
    """Base error for durable local process storage."""


class DurableProcessUnavailableError(DurableProcessError):
    """Raised when durable local processes are not supported by this platform."""


class DurableProcessRecordError(DurableProcessError):
    """Raised when an on-disk process record is missing or malformed."""


class DurableProcessStream(str, Enum):
    """A captured child output stream."""

    STDOUT = "stdout"
    STDERR = "stderr"
    COMBINED = "output"


@dataclass(frozen=True, slots=True)
class DurableProcessSpec:
    """The immutable, versioned process launch specification."""

    process_id: str
    command: str
    shell: str
    cwd: Path
    created_at: float
    origin_session_id: str | None
    agent_name: str | None
    output_byte_limit: int
    output_retention_byte_limit: int


@dataclass(frozen=True, slots=True)
class DurableProcessStatus:
    """The latest atomically-written supervisor state."""

    state: DurableProcessState
    exit_code: int | None
    updated_at: float
    heartbeat_at: float | None
    supervisor_pid: int | None
    child_pid: int | None
    started_at: float | None


@dataclass(frozen=True, slots=True)
class DurableProcessSnapshot:
    """A point-in-time view of one durable process."""

    spec: DurableProcessSpec
    status: DurableProcessStatus
    stdout_bytes: int
    stderr_bytes: int
    output_bytes: int
    stdout_total_bytes: int
    stderr_total_bytes: int
    output_total_bytes: int
    stdout_dropped_bytes: int
    stderr_dropped_bytes: int
    output_dropped_bytes: int
    session_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DurableProcessCapture:
    """Persisted raw and dropped output byte accounting."""

    stdout_total_bytes: int
    stderr_total_bytes: int
    output_total_bytes: int
    stdout_dropped_bytes: int
    stderr_dropped_bytes: int
    output_dropped_bytes: int


@dataclass(frozen=True, slots=True)
class DurableProcessOutput:
    """A bounded slice of one durable process output stream."""

    stream: DurableProcessStream
    offset: int
    next_offset: int
    text: str
    at_end: bool
    returned_bytes: int
    match_count: int | None = None


class DurableProcessStore:
    """Manage file-backed process records rooted in an explicit private directory."""

    def __init__(
        self,
        root: Path,
        *,
        heartbeat_timeout_seconds: float = 5.0,
        max_terminal_records: int = MAX_RETAINED_DURABLE_PROCESS_RECORDS,
    ) -> None:
        _require_posix()
        if heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat_timeout_seconds must be positive.")
        if max_terminal_records < 0:
            raise ValueError("max_terminal_records must not be negative.")
        self._root = root.resolve()
        self._heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self._max_terminal_records = max_terminal_records
        _ensure_private_directory(self._root)
        self.prune_terminal_records()

    @property
    def root(self) -> Path:
        """Return this store's explicit root directory."""

        return self._root

    def create(
        self,
        *,
        command: str,
        shell: Path,
        cwd: Path,
        origin_session_id: str | None = None,
        agent_name: str | None = None,
        output_byte_limit: int = DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
        output_retention_byte_limit: int = DEFAULT_DURABLE_PROCESS_OUTPUT_RETENTION_BYTES,
        max_active_processes: int = MAX_MANAGED_SHELL_PROCESSES,
    ) -> DurableProcessSnapshot:
        """Create a private durable record without launching its supervisor."""

        if not command:
            raise ValueError("command must not be empty.")
        if output_byte_limit <= 0:
            raise ValueError("output_byte_limit must be positive.")
        if output_retention_byte_limit <= 0:
            raise ValueError("output_retention_byte_limit must be positive.")
        if max_active_processes <= 0:
            raise ValueError("max_active_processes must be positive.")
        shell_path = _validate_shell(shell)
        working_directory = _validate_cwd(cwd)

        with FileLock(self._root / ".capacity.lock", mode=0o600):
            if self._active_process_count() >= max_active_processes:
                raise DurableProcessError(
                    f"at most {max_active_processes} managed shell processes may run at once"
                )
            return self._create_record(
                command=command,
                shell_path=shell_path,
                working_directory=working_directory,
                origin_session_id=origin_session_id,
                agent_name=agent_name,
                output_byte_limit=output_byte_limit,
                output_retention_byte_limit=output_retention_byte_limit,
            )

    def _create_record(
        self,
        *,
        command: str,
        shell_path: Path,
        working_directory: Path,
        origin_session_id: str | None,
        agent_name: str | None,
        output_byte_limit: int,
        output_retention_byte_limit: int,
    ) -> DurableProcessSnapshot:
        for _ in range(16):
            process_id = _new_process_id()
            directory = self._root / process_id
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                continue

            try:
                spec = DurableProcessSpec(
                    process_id=process_id,
                    command=command,
                    shell=str(shell_path),
                    cwd=working_directory,
                    created_at=time.time(),
                    origin_session_id=origin_session_id,
                    agent_name=agent_name,
                    output_byte_limit=output_byte_limit,
                    output_retention_byte_limit=output_retention_byte_limit,
                )
                _write_spec(directory, spec)
                for stream in DurableProcessStream:
                    _create_private_file(_output_path(directory, stream))
                status = DurableProcessStatus(
                    state="created",
                    exit_code=None,
                    updated_at=time.time(),
                    heartbeat_at=None,
                    supervisor_pid=None,
                    child_pid=None,
                    started_at=None,
                )
                _write_status(directory, status)
                return DurableProcessSnapshot(
                    spec,
                    status,
                    stdout_bytes=0,
                    stderr_bytes=0,
                    output_bytes=0,
                    stdout_total_bytes=0,
                    stderr_total_bytes=0,
                    output_total_bytes=0,
                    stdout_dropped_bytes=0,
                    stderr_dropped_bytes=0,
                    output_dropped_bytes=0,
                    session_ids=((origin_session_id,) if origin_session_id is not None else ()),
                )
            except BaseException:
                _remove_directory(directory)
                raise

        raise DurableProcessError("Could not allocate a unique durable process ID.")

    def _active_process_count(self) -> int:
        return sum(snapshot.status.state in _NONTERMINAL_STATES for snapshot in self._discover())

    def launch(self, process_id: str, *, environment: Mapping[str, str]) -> DurableProcessSnapshot:
        """Start a detached supervisor that inherits, but never persists, ``environment``."""

        directory = self._directory(process_id)
        _read_spec(directory)
        current = _read_status(directory)
        if current.state != "created":
            raise DurableProcessError(f"{process_id} has already been launched.")
        _validate_environment(environment)
        _claim_launch(directory)

        starting = DurableProcessStatus(
            state="starting",
            exit_code=None,
            updated_at=time.time(),
            heartbeat_at=time.time(),
            supervisor_pid=None,
            child_pid=None,
            started_at=None,
        )
        _write_status(directory, starting)
        try:
            with Path(os.devnull).open("rb") as stdin, Path(os.devnull).open("ab") as output:
                subprocess.Popen(
                    [
                        sys.executable,
                        "-P",
                        "-m",
                        "fast_agent.tools.durable_process_supervisor",
                        "--root",
                        str(self._root),
                        "--process-id",
                        process_id,
                        "--max-terminal-records",
                        str(self._max_terminal_records),
                    ],
                    stdin=stdin,
                    stdout=output,
                    stderr=output,
                    cwd=self._root,
                    env=dict(environment),
                    start_new_session=True,
                    close_fds=True,
                )
        except OSError:
            _write_status(
                directory,
                DurableProcessStatus(
                    state="unavailable",
                    exit_code=None,
                    updated_at=time.time(),
                    heartbeat_at=None,
                    supervisor_pid=None,
                    child_pid=None,
                    started_at=None,
                ),
            )
            raise
        finally:
            with suppress(OSError, DurableProcessRecordError):
                self.prune_terminal_records()
        return self.get(process_id)

    def discover(self) -> list[DurableProcessSnapshot]:
        """Return valid snapshots, persisting unavailable process tombstones."""

        with FileLock(self._root / ".capacity.lock", mode=0o600):
            return self._discover()

    def _discover(self) -> list[DurableProcessSnapshot]:
        snapshots: list[DurableProcessSnapshot] = []
        for entry in sorted(self._root.iterdir()):
            if entry.is_dir() and _is_process_id(entry.name):
                try:
                    snapshots.append(self.get(entry.name))
                except DurableProcessRecordError:
                    continue
        return snapshots

    def prune_terminal_records(self) -> int:
        """Remove oldest completed records beyond the configured retention count."""

        removed = 0
        with FileLock(self._root / ".capacity.lock", mode=0o600):
            terminal_records: list[tuple[float, float, str]] = []
            for entry in self._root.iterdir():
                if not entry.is_dir() or not _is_process_id(entry.name):
                    continue
                try:
                    directory = self._directory(entry.name)
                    spec = _read_spec(directory)
                    status = _read_status(directory)
                except DurableProcessRecordError:
                    continue
                status = self._persist_unavailable_status(directory, status)
                if self._status_is_prunable(status):
                    terminal_records.append((status.updated_at, spec.created_at, spec.process_id))

            terminal_records.sort(reverse=True)
            for _, _, process_id in terminal_records[self._max_terminal_records :]:
                try:
                    directory = self._directory(process_id)
                    status = _read_status(directory)
                except DurableProcessRecordError:
                    continue
                if not self._status_is_prunable(status):
                    continue
                _remove_directory(directory)
                removed += 1
        return removed

    def get(self, process_id: str) -> DurableProcessSnapshot:
        """Read one process snapshot, marking a stale supervisor as unavailable."""

        directory = self._directory(process_id)
        spec = _read_spec(directory)
        status = _read_status(directory)
        status = self._persist_unavailable_status(directory, status)
        if self._is_stale(status):
            status = DurableProcessStatus(
                state="unavailable",
                exit_code=None,
                updated_at=status.updated_at,
                heartbeat_at=status.heartbeat_at,
                supervisor_pid=status.supervisor_pid,
                child_pid=status.child_pid,
                started_at=status.started_at,
            )
        stdout_bytes = _file_size(_output_path(directory, DurableProcessStream.STDOUT))
        stderr_bytes = _file_size(_output_path(directory, DurableProcessStream.STDERR))
        output_bytes = _file_size(_output_path(directory, DurableProcessStream.COMBINED))
        capture = _read_capture(
            directory,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            output_bytes=output_bytes,
        )
        return DurableProcessSnapshot(
            spec=spec,
            status=status,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            output_bytes=output_bytes,
            stdout_total_bytes=capture.stdout_total_bytes,
            stderr_total_bytes=capture.stderr_total_bytes,
            output_total_bytes=capture.output_total_bytes,
            stdout_dropped_bytes=capture.stdout_dropped_bytes,
            stderr_dropped_bytes=capture.stderr_dropped_bytes,
            output_dropped_bytes=capture.output_dropped_bytes,
            session_ids=_read_session_links(directory, origin=spec.origin_session_id),
        )

    def poll(self, process_id: str) -> DurableProcessSnapshot:
        """Return the current process snapshot."""

        return self.get(process_id)

    def directory(self, process_id: str) -> Path:
        """Return the validated private directory for one process."""

        return self._directory(process_id)

    def read_output(
        self,
        process_id: str,
        *,
        stream: DurableProcessStream,
        offset: int,
        limit: int,
        query: str | None = None,
    ) -> DurableProcessOutput:
        """Read one bounded output range, optionally retaining matching text lines."""

        if offset < 0:
            raise ValueError("offset must not be negative.")
        if not 0 < limit <= _MAX_OUTPUT_READ_BYTES:
            raise ValueError(f"limit must be between 1 and {_MAX_OUTPUT_READ_BYTES}.")
        if query == "":
            raise ValueError("query must not be empty.")
        if query is not None and len(query) > MAX_PROCESS_OUTPUT_QUERY_CHARS:
            raise ValueError(f"query must be at most {MAX_PROCESS_OUTPUT_QUERY_CHARS} characters.")

        path = _output_path(self._directory(process_id), stream)
        offset = min(offset, _file_size(path))
        if query is not None:
            return _search_output(
                path,
                stream=stream,
                offset=offset,
                limit=limit,
                query=query,
            )
        with path.open("rb") as output:
            output.seek(offset)
            payload = output.read(limit)
        total_bytes = _file_size(path)
        text = payload.decode("utf-8", errors="replace")
        next_offset = offset + len(payload)
        return DurableProcessOutput(
            stream=stream,
            offset=offset,
            next_offset=next_offset,
            text=text,
            at_end=next_offset >= total_bytes,
            returned_bytes=len(payload),
        )

    def request_stop(self, process_id: str) -> bool:
        """Atomically request a stop, returning whether this call created the request."""

        directory = self._directory(process_id)
        _read_spec(directory)
        return _write_stop_request(directory)

    def link_session(self, process_id: str, session_id: str) -> DurableProcessSnapshot:
        """Persist a non-owning association between a process and a session."""

        if not session_id:
            raise ValueError("session_id must not be empty.")
        directory = self._directory(process_id)
        _read_spec(directory)
        links = directory / "links"
        links.mkdir(mode=0o700, exist_ok=True)
        _ensure_private_directory(links)
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        _atomic_replace(
            links / f"{digest}.json",
            _encode_json({"version": _VERSION, "session_id": session_id}),
        )
        return self.get(process_id)

    def wait_for_launch(
        self,
        process_id: str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 0.05,
    ) -> DurableProcessSnapshot:
        """Wait until a launched process leaves its initial starting state."""

        return self._wait_until(
            process_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            predicate=lambda snapshot: snapshot.status.state != "starting",
        )

    def wait(
        self,
        process_id: str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 0.05,
    ) -> DurableProcessSnapshot:
        """Wait for a terminal or unavailable process state."""

        return self._wait_until(
            process_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            predicate=lambda snapshot: (
                snapshot.status.state in {"exited", "stopped", "unavailable"}
            ),
        )

    def wait_for_change(
        self,
        process_id: str,
        *,
        previous: DurableProcessSnapshot,
        timeout_seconds: float,
        poll_interval_seconds: float = 0.05,
    ) -> DurableProcessSnapshot:
        """Wait for status or output-size changes from an earlier snapshot."""

        return self._wait_until(
            process_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            predicate=lambda snapshot: snapshot != previous,
        )

    def _wait_until(
        self,
        process_id: str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
        predicate: Callable[[DurableProcessSnapshot], bool],
    ) -> DurableProcessSnapshot:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative.")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive.")
        deadline = time.monotonic() + timeout_seconds
        while True:
            snapshot = self.get(process_id)
            if predicate(snapshot):
                return snapshot
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {process_id}.")
            time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))

    def _directory(self, process_id: str) -> Path:
        validate_process_id(process_id)
        directory = self._root / process_id
        try:
            details = directory.lstat()
        except FileNotFoundError as exc:
            raise DurableProcessRecordError(f"Unknown durable process: {process_id}.") from exc
        if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
            raise DurableProcessRecordError(f"Invalid durable process directory: {process_id}.")
        return directory

    def _is_stale(self, status: DurableProcessStatus) -> bool:
        last_seen_at = status.heartbeat_at or status.updated_at
        return (
            status.state in _NONTERMINAL_STATES
            and time.time() - last_seen_at > self._heartbeat_timeout_seconds
        )

    def _status_is_prunable(self, status: DurableProcessStatus) -> bool:
        return status.state in _TERMINAL_STATES

    def _persist_unavailable_status(
        self,
        directory: Path,
        status: DurableProcessStatus,
    ) -> DurableProcessStatus:
        abandoned_launch = (
            status.state in {"created", "starting"}
            and status.supervisor_pid is None
            and status.child_pid is None
            and self._is_stale(status)
        )
        if not self._status_has_disappeared(status) and not abandoned_launch:
            return status
        unavailable = DurableProcessStatus(
            state="unavailable",
            exit_code=None,
            updated_at=status.updated_at,
            heartbeat_at=status.heartbeat_at,
            supervisor_pid=status.supervisor_pid,
            child_pid=status.child_pid,
            started_at=status.started_at,
        )
        _write_status(directory, unavailable)
        return unavailable

    def _status_has_disappeared(self, status: DurableProcessStatus) -> bool:
        return (
            status.state in _NONTERMINAL_STATES
            and status.supervisor_pid is not None
            and status.child_pid is not None
            and not _process_is_running(status.supervisor_pid)
            and not _process_is_running(status.child_pid)
        )


def validate_process_id(process_id: str) -> None:
    """Reject process identifiers that are not stable, generated IDs."""

    if not _is_process_id(process_id):
        raise ValueError(f"Invalid durable process ID: {process_id!r}.")


def _is_process_id(process_id: str) -> bool:
    suffix = process_id.removeprefix(_PROCESS_ID_PREFIX)
    return (
        len(process_id) == _PROCESS_ID_LENGTH
        and len(suffix) == 32
        and all(character in "0123456789abcdef" for character in suffix)
    )


def _new_process_id() -> str:
    return f"{_PROCESS_ID_PREFIX}{uuid.uuid4().hex}"


def _require_posix() -> None:
    if os.name != "posix":
        raise DurableProcessUnavailableError("Durable local processes require POSIX.")


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    details = path.lstat()
    if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
        raise DurableProcessRecordError(f"Durable process root is not a directory: {path}.")
    if details.st_uid != os.getuid() or details.st_mode & 0o077:
        raise DurableProcessRecordError(f"Durable process root is not private: {path}.")


def _validate_shell(shell: Path) -> Path:
    path = shell.resolve()
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("shell must be an executable file.")
    return path


def _validate_cwd(cwd: Path) -> Path:
    path = cwd.resolve()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("cwd must be an existing directory.")
    return path


def _validate_environment(environment: Mapping[str, str]) -> None:
    for key, value in environment.items():
        if not key or "=" in key or "\x00" in key or "\x00" in value:
            raise ValueError("environment entries must be non-empty, NUL-free strings.")


def _process_is_running(process_id: int | None) -> bool:
    if process_id is None:
        return False
    try:
        os.kill(process_id, 0)
    except (OverflowError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _write_spec(directory: Path, spec: DurableProcessSpec) -> None:
    path = directory / "spec.json"
    payload = _encode_json(
        {
            "version": _VERSION,
            "process_id": spec.process_id,
            "command": spec.command,
            "shell": spec.shell,
            "cwd": str(spec.cwd),
            "created_at": spec.created_at,
            "origin_session_id": spec.origin_session_id,
            "agent_name": spec.agent_name,
            "output_byte_limit": spec.output_byte_limit,
            "output_retention_byte_limit": spec.output_retention_byte_limit,
        }
    )
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read_spec(directory: Path) -> DurableProcessSpec:
    value = _read_json(directory / "spec.json")
    legacy_keys = {
        "version",
        "process_id",
        "command",
        "shell",
        "cwd",
        "created_at",
        "origin_session_id",
        "agent_name",
        "output_byte_limit",
    }
    if frozenset(value) not in {
        frozenset(legacy_keys),
        frozenset({*legacy_keys, "output_retention_byte_limit"}),
    }:
        raise DurableProcessRecordError("Malformed durable process spec.")
    version = _required_int(value, "version", "spec")
    process_id = _required_str(value, "process_id", "spec")
    command = _required_str(value, "command", "spec")
    shell = _required_str(value, "shell", "spec")
    cwd = _required_str(value, "cwd", "spec")
    created_at = _required_float(value, "created_at", "spec")
    origin_session_id = _optional_str(value, "origin_session_id", "spec")
    agent_name = _optional_str(value, "agent_name", "spec")
    output_byte_limit = _required_int(value, "output_byte_limit", "spec")
    output_retention_byte_limit = (
        _required_int(value, "output_retention_byte_limit", "spec")
        if "output_retention_byte_limit" in value
        else DEFAULT_DURABLE_PROCESS_OUTPUT_RETENTION_BYTES
    )
    if version != _VERSION:
        raise DurableProcessRecordError("Unsupported durable process spec version.")
    validate_process_id(process_id)
    if (
        directory.name != process_id
        or not command
        or output_byte_limit <= 0
        or output_retention_byte_limit <= 0
    ):
        raise DurableProcessRecordError("Invalid durable process spec.")
    return DurableProcessSpec(
        process_id=process_id,
        command=command,
        shell=shell,
        cwd=Path(cwd),
        created_at=created_at,
        origin_session_id=origin_session_id,
        agent_name=agent_name,
        output_byte_limit=output_byte_limit,
        output_retention_byte_limit=output_retention_byte_limit,
    )


def _write_status(directory: Path, status: DurableProcessStatus) -> None:
    _validate_status(status)
    _atomic_replace(
        directory / "status.json",
        _encode_json(
            {
                "version": _VERSION,
                "state": status.state,
                "exit_code": status.exit_code,
                "updated_at": status.updated_at,
                "heartbeat_at": status.heartbeat_at,
                "supervisor_pid": status.supervisor_pid,
                "child_pid": status.child_pid,
                "started_at": status.started_at,
            }
        ),
    )


def _read_status(directory: Path) -> DurableProcessStatus:
    value = _read_json(directory / "status.json")
    _require_keys(
        value,
        {
            "version",
            "state",
            "exit_code",
            "updated_at",
            "heartbeat_at",
            "supervisor_pid",
            "child_pid",
            "started_at",
        },
        "status",
    )
    version = _required_int(value, "version", "status")
    state = _required_state(value, "state", "status")
    exit_code = _optional_int(value, "exit_code", "status")
    updated_at = _required_float(value, "updated_at", "status")
    heartbeat_at = _optional_float(value, "heartbeat_at", "status")
    supervisor_pid = _optional_int(value, "supervisor_pid", "status")
    child_pid = _optional_int(value, "child_pid", "status")
    started_at = _optional_float(value, "started_at", "status")
    if version != _VERSION:
        raise DurableProcessRecordError("Invalid durable process status.")
    status = DurableProcessStatus(
        state=state,
        exit_code=exit_code,
        updated_at=updated_at,
        heartbeat_at=heartbeat_at,
        supervisor_pid=supervisor_pid,
        child_pid=child_pid,
        started_at=started_at,
    )
    _validate_status(status)
    return status


def _write_capture(directory: Path, capture: DurableProcessCapture) -> None:
    _atomic_replace(
        directory / "capture.json",
        _encode_json(
            {
                "version": _VERSION,
                "stdout_total_bytes": capture.stdout_total_bytes,
                "stderr_total_bytes": capture.stderr_total_bytes,
                "output_total_bytes": capture.output_total_bytes,
                "stdout_dropped_bytes": capture.stdout_dropped_bytes,
                "stderr_dropped_bytes": capture.stderr_dropped_bytes,
                "output_dropped_bytes": capture.output_dropped_bytes,
            }
        ),
    )


def _read_capture(
    directory: Path,
    *,
    stdout_bytes: int,
    stderr_bytes: int,
    output_bytes: int,
) -> DurableProcessCapture:
    path = directory / "capture.json"
    if not path.exists():
        return DurableProcessCapture(
            stdout_total_bytes=stdout_bytes,
            stderr_total_bytes=stderr_bytes,
            output_total_bytes=output_bytes,
            stdout_dropped_bytes=0,
            stderr_dropped_bytes=0,
            output_dropped_bytes=0,
        )
    value = _read_json(path)
    _require_keys(
        value,
        {
            "version",
            "stdout_total_bytes",
            "stderr_total_bytes",
            "output_total_bytes",
            "stdout_dropped_bytes",
            "stderr_dropped_bytes",
            "output_dropped_bytes",
        },
        "capture",
    )
    if _required_int(value, "version", "capture") != _VERSION:
        raise DurableProcessRecordError("Unsupported durable process capture version.")
    stdout_total_bytes = _required_int(value, "stdout_total_bytes", "capture")
    stderr_total_bytes = _required_int(value, "stderr_total_bytes", "capture")
    output_total_bytes = _required_int(value, "output_total_bytes", "capture")
    stdout_dropped_bytes = _required_int(value, "stdout_dropped_bytes", "capture")
    stderr_dropped_bytes = _required_int(value, "stderr_dropped_bytes", "capture")
    output_dropped_bytes = _required_int(value, "output_dropped_bytes", "capture")
    if (
        min(
            stdout_total_bytes,
            stderr_total_bytes,
            output_total_bytes,
            stdout_dropped_bytes,
            stderr_dropped_bytes,
            output_dropped_bytes,
        )
        < 0
    ):
        raise DurableProcessRecordError("Invalid durable process capture.")
    return DurableProcessCapture(
        stdout_total_bytes=max(stdout_total_bytes, stdout_bytes),
        stderr_total_bytes=max(stderr_total_bytes, stderr_bytes),
        output_total_bytes=max(output_total_bytes, output_bytes),
        stdout_dropped_bytes=stdout_dropped_bytes,
        stderr_dropped_bytes=stderr_dropped_bytes,
        output_dropped_bytes=output_dropped_bytes,
    )


def _validate_status(status: DurableProcessStatus) -> None:
    if not math.isfinite(status.updated_at):
        raise DurableProcessRecordError("Invalid durable process status timestamp.")
    if status.heartbeat_at is not None and not math.isfinite(status.heartbeat_at):
        raise DurableProcessRecordError("Invalid durable process heartbeat.")
    if status.started_at is not None and not math.isfinite(status.started_at):
        raise DurableProcessRecordError("Invalid durable process start timestamp.")
    if status.supervisor_pid is not None and status.supervisor_pid <= 0:
        raise DurableProcessRecordError("Invalid durable process supervisor PID.")
    if status.child_pid is not None and status.child_pid <= 0:
        raise DurableProcessRecordError("Invalid durable process child PID.")
    if status.state in {"exited", "stopped"}:
        if status.exit_code is None:
            raise DurableProcessRecordError("Terminal durable process status has no exit code.")
    elif status.exit_code is not None:
        raise DurableProcessRecordError("Nonterminal durable process status has an exit code.")


def _output_path(directory: Path, stream: DurableProcessStream) -> Path:
    return directory / f"{stream.value}.log"


def _search_output(
    path: Path,
    *,
    stream: DurableProcessStream,
    offset: int,
    limit: int,
    query: str,
) -> DurableProcessOutput:
    selected = bytearray()
    match_count = 0
    scan_bytes = max(_file_size(path) - offset, 0)
    scan_end = offset + scan_bytes
    continuation_offset: int | None = None
    has_more = False
    with path.open("rb") as output:
        output.seek(offset)
        for matched, line_start, line_end, line_prefix, line_bytes in _iter_output_search_lines(
            output,
            base_offset=offset,
            scan_bytes=scan_bytes,
            query=query,
            prefix_limit=limit,
        ):
            if not matched:
                continue
            match_count += 1
            remaining = limit - len(selected)
            if remaining <= 0:
                if continuation_offset is None:
                    continuation_offset = line_start
                has_more = True
                continue
            if line_bytes <= remaining:
                selected.extend(line_prefix)
                continue
            if not selected:
                selected.extend(line_prefix[:remaining])
                continuation_offset = line_end
                continue
            if continuation_offset is None:
                continuation_offset = line_start
            has_more = True

    return DurableProcessOutput(
        stream=stream,
        offset=offset,
        next_offset=continuation_offset if continuation_offset is not None else scan_end,
        text=bytes(selected).decode("utf-8", errors="replace"),
        at_end=not has_more,
        returned_bytes=len(selected),
        match_count=match_count,
    )


class _OutputSearchLine:
    def __init__(self, *, query: str, prefix_limit: int) -> None:
        self._query = query
        self._query_overlap = max(len(query) - 1, 0)
        self._prefix_limit = prefix_limit
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._search_tail = ""
        self._prefix = bytearray()
        self._encoded_bytes = 0
        self._matched = False

    def feed(self, payload: bytes, *, final: bool) -> None:
        text = self._decoder.decode(payload, final=final)
        candidate = self._search_tail + text
        if not self._matched and self._query in candidate:
            self._matched = True
        self._search_tail = candidate[-self._query_overlap :] if self._query_overlap else ""
        encoded = text.encode("utf-8")
        self._encoded_bytes += len(encoded)
        remaining = self._prefix_limit - len(self._prefix)
        if remaining > 0:
            self._prefix.extend(encoded[:remaining])

    def result(self) -> tuple[bool, bytes, int]:
        return self._matched, bytes(self._prefix), self._encoded_bytes


def _iter_output_search_lines(
    output: BinaryIO,
    *,
    base_offset: int,
    scan_bytes: int,
    query: str,
    prefix_limit: int,
) -> Iterator[tuple[bool, int, int, bytes, int]]:
    line = _OutputSearchLine(query=query, prefix_limit=prefix_limit)
    line_has_data = False
    line_start = base_offset
    position = base_offset
    remaining_bytes = scan_bytes
    while remaining_bytes > 0:
        chunk = output.read(min(_OUTPUT_SEARCH_CHUNK_BYTES, remaining_bytes))
        if not chunk:
            break
        remaining_bytes -= len(chunk)
        cursor = 0
        while cursor < len(chunk):
            newline = chunk.find(b"\n", cursor)
            end = len(chunk) if newline < 0 else newline + 1
            fragment = chunk[cursor:end]
            line.feed(fragment, final=newline >= 0)
            line_has_data = True
            position += len(fragment)
            cursor = end
            if newline >= 0:
                matched, prefix, encoded_bytes = line.result()
                yield matched, line_start, position, prefix, encoded_bytes
                line = _OutputSearchLine(query=query, prefix_limit=prefix_limit)
                line_has_data = False
                line_start = position

    if line_has_data:
        line.feed(b"", final=True)
        matched, prefix, encoded_bytes = line.result()
        yield matched, line_start, position, prefix, encoded_bytes


def _read_session_links(directory: Path, *, origin: str | None) -> tuple[str, ...]:
    session_ids = {origin} if origin is not None else set()
    links = directory / "links"
    if not links.is_dir():
        return tuple(sorted(session_ids))
    for path in links.glob("*.json"):
        try:
            value = _read_json(path)
            _require_keys(value, {"version", "session_id"}, "session link")
            if _required_int(value, "version", "session link") != _VERSION:
                continue
            session_id = _required_str(value, "session_id", "session link")
            if session_id:
                session_ids.add(session_id)
        except DurableProcessRecordError:
            continue
    return tuple(sorted(session_ids))


def _create_private_file(path: Path) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)


def _claim_launch(directory: Path) -> None:
    path = directory / "launch"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400)
    except FileExistsError as exc:
        raise DurableProcessError(f"{directory.name} has already been launched.") from exc
    os.close(descriptor)
    _fsync_directory(directory)


def _write_stop_request(directory: Path) -> bool:
    control_directory = directory / "control"
    control_directory.mkdir(mode=0o700, exist_ok=True)
    _ensure_private_directory(control_directory)
    target = control_directory / "stop.json"
    temporary = control_directory / f".stop-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            stream.write(_encode_json({"version": _VERSION, "requested_at": time.time()}))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            return False
        _fsync_directory(control_directory)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _stop_requested(directory: Path) -> bool:
    path = directory / "control" / "stop.json"
    if not path.exists():
        return False
    value = _read_json(path)
    _require_keys(value, {"version", "requested_at"}, "stop request")
    if _required_int(value, "version", "stop request") != _VERSION:
        raise DurableProcessRecordError("Unsupported durable process stop request version.")
    _required_float(value, "requested_at", "stop request")
    return True


def _atomic_replace(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DurableProcessRecordError(f"Missing durable process record: {path.name}.") from exc
    except UnicodeDecodeError as exc:
        raise DurableProcessRecordError(f"Malformed durable process record: {path.name}.") from exc
    try:
        decoded: object = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DurableProcessRecordError(f"Malformed durable process record: {path.name}.") from exc
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise DurableProcessRecordError(f"Malformed durable process record: {path.name}.")
    return {key: value for key, value in decoded.items() if isinstance(key, str)}


def _require_keys(value: dict[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise DurableProcessRecordError(f"Malformed durable process {name}.")


def _required_str(value: dict[str, object], key: str, name: str) -> str:
    item = value[key]
    if not isinstance(item, str):
        raise DurableProcessRecordError(f"Malformed durable process {name}.")
    return item


def _optional_str(value: dict[str, object], key: str, name: str) -> str | None:
    item = value[key]
    if item is None:
        return None
    if not isinstance(item, str):
        raise DurableProcessRecordError(f"Malformed durable process {name}.")
    return item


def _required_state(
    value: dict[str, object],
    key: str,
    name: str,
) -> DurableProcessState:
    state = _required_str(value, key, name)
    match state:
        case "created":
            return "created"
        case "starting":
            return "starting"
        case "running":
            return "running"
        case "stopping":
            return "stopping"
        case "exited":
            return "exited"
        case "stopped":
            return "stopped"
        case "unavailable":
            return "unavailable"
        case _:
            raise DurableProcessRecordError(f"Malformed durable process {name}.")


def _required_int(value: dict[str, object], key: str, name: str) -> int:
    item = value[key]
    if not isinstance(item, int) or isinstance(item, bool):
        raise DurableProcessRecordError(f"Malformed durable process {name}.")
    return item


def _optional_int(value: dict[str, object], key: str, name: str) -> int | None:
    item = value[key]
    if item is None:
        return None
    if not isinstance(item, int) or isinstance(item, bool):
        raise DurableProcessRecordError(f"Malformed durable process {name}.")
    return item


def _required_float(value: dict[str, object], key: str, name: str) -> float:
    item = value[key]
    if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item):
        raise DurableProcessRecordError(f"Malformed durable process {name}.")
    return float(item)


def _optional_float(value: dict[str, object], key: str, name: str) -> float | None:
    item = value[key]
    if item is None:
        return None
    if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item):
        raise DurableProcessRecordError(f"Malformed durable process {name}.")
    return float(item)


def _encode_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError as exc:
        raise DurableProcessRecordError(f"Missing durable process output: {path.name}.") from exc


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_directory(directory: Path) -> None:
    shutil.rmtree(directory)
