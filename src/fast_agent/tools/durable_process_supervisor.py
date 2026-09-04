"""Detached supervisor entry point for durable POSIX-local processes."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, BinaryIO

from fast_agent.constants import MAX_RETAINED_DURABLE_PROCESS_RECORDS

if TYPE_CHECKING:
    from types import FrameType

from fast_agent.tools.durable_processes import (
    DurableProcessCapture,
    DurableProcessRecordError,
    DurableProcessStatus,
    DurableProcessStore,
    DurableProcessStream,
    _output_path,
    _read_spec,
    _stop_requested,
    _write_capture,
    _write_status,
    validate_process_id,
)

_HEARTBEAT_SECONDS = 1.0
_CAPTURE_PERSIST_SECONDS = 0.1
_POLL_SECONDS = 0.1
_STOP_GRACE_SECONDS = 2.0
_STREAM_CHUNK_BYTES = 65536
_OUTPUT_DRAIN_SECONDS = 2.0


class _ShutdownRequest:
    def __init__(self) -> None:
        self.requested = False


class _CaptureState:
    def __init__(self, *, byte_limit: int, lock: threading.Lock) -> None:
        self._byte_limit = byte_limit
        self._lock = lock
        self._stdout_total_bytes = 0
        self._stderr_total_bytes = 0
        self._output_total_bytes = 0
        self._stdout_dropped_bytes = 0
        self._stderr_dropped_bytes = 0
        self._output_dropped_bytes = 0
        self._generation = 0
        self._persisted_generation = -1

    def write(
        self,
        payload: bytes,
        *,
        stream: DurableProcessStream,
        stream_output: BinaryIO,
        combined_output: BinaryIO,
    ) -> None:
        with self._lock:
            stream_retained = min(
                len(payload),
                max(self._byte_limit - stream_output.tell(), 0),
            )
            stream_output.write(payload[:stream_retained])
            stream_dropped = len(payload) - stream_retained
            if stream is DurableProcessStream.STDOUT:
                self._stdout_total_bytes += len(payload)
                self._stdout_dropped_bytes += stream_dropped
            else:
                self._stderr_total_bytes += len(payload)
                self._stderr_dropped_bytes += stream_dropped

            combined_retained = min(
                len(payload),
                max(self._byte_limit - combined_output.tell(), 0),
            )
            combined_output.write(payload[:combined_retained])
            self._output_total_bytes += len(payload)
            self._output_dropped_bytes += len(payload) - combined_retained
            self._generation += 1

    def snapshot(self) -> DurableProcessCapture:
        with self._lock:
            return self._snapshot_locked()

    def persist(self, directory: Path, *, force: bool = False) -> bool:
        with self._lock:
            generation = self._generation
            if not force and generation == self._persisted_generation:
                return False
            snapshot = self._snapshot_locked()
        _write_capture(directory, snapshot)
        with self._lock:
            self._persisted_generation = max(self._persisted_generation, generation)
        return True

    def _snapshot_locked(self) -> DurableProcessCapture:
        return DurableProcessCapture(
            stdout_total_bytes=self._stdout_total_bytes,
            stderr_total_bytes=self._stderr_total_bytes,
            output_total_bytes=self._output_total_bytes,
            stdout_dropped_bytes=self._stdout_dropped_bytes,
            stderr_dropped_bytes=self._stderr_dropped_bytes,
            output_dropped_bytes=self._output_dropped_bytes,
        )


def main() -> int:
    """Run one process supervisor selected by explicit root and process ID."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--process-id", required=True)
    parser.add_argument(
        "--max-terminal-records",
        type=int,
        default=MAX_RETAINED_DURABLE_PROCESS_RECORDS,
    )
    arguments = parser.parse_args()
    root = Path(arguments.root)
    process_id = arguments.process_id
    shutdown = _ShutdownRequest()

    def request_shutdown(_signal_number: int, _frame: FrameType | None) -> None:
        shutdown.requested = True

    previous_handlers = {
        signal_number: signal.signal(signal_number, request_shutdown)
        for signal_number in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        validate_process_id(process_id)
        store = DurableProcessStore(
            root,
            max_terminal_records=arguments.max_terminal_records,
        )
        _supervise(store, process_id, shutdown=shutdown)
    except DurableProcessRecordError:
        return 2
    finally:
        for signal_number, previous_handler in previous_handlers.items():
            signal.signal(signal_number, previous_handler)
    return 0


def _supervise(
    store: DurableProcessStore,
    process_id: str,
    *,
    shutdown: _ShutdownRequest,
) -> None:
    directory = store.root / process_id
    spec = _read_spec(directory)
    started_at = time.time()
    child: subprocess.Popen[bytes] | None = None
    capture: _CaptureState | None = None
    try:
        with (
            Path(os.devnull).open("rb") as stdin,
            _output_path(directory, DurableProcessStream.STDOUT).open("ab", buffering=0) as stdout,
            _output_path(directory, DurableProcessStream.STDERR).open("ab", buffering=0) as stderr,
            _output_path(directory, DurableProcessStream.COMBINED).open(
                "ab", buffering=0
            ) as combined,
        ):
            child = subprocess.Popen(
                [spec.shell, "-c", spec.command],
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=spec.cwd,
                start_new_session=True,
                close_fds=True,
            )
            if child.stdout is None or child.stderr is None:
                raise OSError("Could not capture durable process output.")
            combined_lock = threading.Lock()
            capture = _CaptureState(
                byte_limit=spec.output_retention_byte_limit,
                lock=combined_lock,
            )
            capture.persist(directory, force=True)
            drain_failures: SimpleQueue[BaseException] = SimpleQueue()
            output_threads = (
                threading.Thread(
                    target=_drain_output,
                    args=(
                        child.stdout,
                        stdout,
                        combined,
                        combined_lock,
                        drain_failures,
                        capture,
                        DurableProcessStream.STDOUT,
                    ),
                    daemon=True,
                ),
                threading.Thread(
                    target=_drain_output,
                    args=(
                        child.stderr,
                        stderr,
                        combined,
                        combined_lock,
                        drain_failures,
                        capture,
                        DurableProcessStream.STDERR,
                    ),
                    daemon=True,
                ),
            )
            for thread in output_threads:
                thread.start()
            _write_status(
                directory,
                DurableProcessStatus(
                    state="running",
                    exit_code=None,
                    updated_at=started_at,
                    heartbeat_at=started_at,
                    supervisor_pid=os.getpid(),
                    child_pid=child.pid,
                    started_at=started_at,
                ),
            )
            stop_requested = _wait_for_child(
                directory,
                child,
                started_at=started_at,
                drain_failures=drain_failures,
                shutdown=shutdown,
                capture=capture,
            )
            if not _drain_output_threads(output_threads):
                raise OSError("Durable process output did not drain after process exit.")
            _raise_drain_failure(drain_failures)
            capture.persist(directory)
            now = time.time()
            _write_status(
                directory,
                DurableProcessStatus(
                    state="stopped" if stop_requested else "exited",
                    exit_code=child.returncode,
                    updated_at=now,
                    heartbeat_at=now,
                    supervisor_pid=os.getpid(),
                    child_pid=child.pid,
                    started_at=started_at,
                ),
            )
    except Exception:
        if child is not None:
            with suppress(OSError):
                _cleanup_child_process_group(child)
        now = time.time()
        if capture is not None:
            with suppress(OSError, DurableProcessRecordError):
                capture.persist(directory)
        with suppress(OSError, DurableProcessRecordError):
            _write_status(
                directory,
                DurableProcessStatus(
                    state="unavailable",
                    exit_code=None,
                    updated_at=now,
                    heartbeat_at=None,
                    supervisor_pid=os.getpid(),
                    child_pid=child.pid if child is not None else None,
                    started_at=started_at,
                ),
            )
    finally:
        if capture is not None:
            with suppress(OSError, DurableProcessRecordError):
                capture.persist(directory)
        with suppress(OSError, DurableProcessRecordError):
            store.prune_terminal_records()


def _wait_for_child(
    directory: Path,
    child: subprocess.Popen[bytes],
    *,
    started_at: float,
    drain_failures: SimpleQueue[BaseException],
    shutdown: _ShutdownRequest,
    capture: _CaptureState,
) -> bool:
    stop_requested = False
    next_heartbeat = time.monotonic() + _HEARTBEAT_SECONDS
    next_capture = time.monotonic() + _CAPTURE_PERSIST_SECONDS
    while child.poll() is None:
        _raise_drain_failure(drain_failures)
        if time.monotonic() >= next_capture:
            capture.persist(directory)
            next_capture = time.monotonic() + _CAPTURE_PERSIST_SECONDS
        if not stop_requested and (shutdown.requested or _stop_requested(directory)):
            stop_requested = True
            now = time.time()
            _write_status(
                directory,
                DurableProcessStatus(
                    state="stopping",
                    exit_code=None,
                    updated_at=now,
                    heartbeat_at=now,
                    supervisor_pid=os.getpid(),
                    child_pid=child.pid,
                    started_at=started_at,
                ),
            )
            _terminate_process_group(child)
        elif time.monotonic() >= next_heartbeat:
            now = time.time()
            _write_status(
                directory,
                DurableProcessStatus(
                    state="running",
                    exit_code=None,
                    updated_at=now,
                    heartbeat_at=now,
                    supervisor_pid=os.getpid(),
                    child_pid=child.pid,
                    started_at=started_at,
                ),
            )
            next_heartbeat = time.monotonic() + _HEARTBEAT_SECONDS
        time.sleep(_POLL_SECONDS)

    _raise_drain_failure(drain_failures)
    _terminate_remaining_process_group(child.pid)
    return stop_requested


def _terminate_process_group(child: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=_STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def _cleanup_child_process_group(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is None:
        _terminate_process_group(child)
    _terminate_remaining_process_group(child.pid)
    with suppress(subprocess.TimeoutExpired):
        child.wait(timeout=_STOP_GRACE_SECONDS)


def _terminate_remaining_process_group(process_group_id: int) -> None:
    if not _process_group_exists(process_group_id):
        return
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + _STOP_GRACE_SECONDS
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group_id):
            return
        time.sleep(_POLL_SECONDS)
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _drain_output(
    source: BinaryIO,
    stream_output: BinaryIO,
    combined_output: BinaryIO,
    combined_lock: threading.Lock,
    drain_failures: SimpleQueue[BaseException],
    capture: _CaptureState | None = None,
    stream: DurableProcessStream = DurableProcessStream.STDOUT,
) -> None:
    try:
        while payload := os.read(source.fileno(), _STREAM_CHUNK_BYTES):
            if capture is None:
                stream_output.write(payload)
                with combined_lock:
                    combined_output.write(payload)
            else:
                capture.write(
                    payload,
                    stream=stream,
                    stream_output=stream_output,
                    combined_output=combined_output,
                )
    except BaseException as exc:
        drain_failures.put(exc)


def _raise_drain_failure(drain_failures: SimpleQueue[BaseException]) -> None:
    try:
        failure = drain_failures.get_nowait()
    except Empty:
        return
    raise OSError("Could not drain durable process output.") from failure


def _drain_output_threads(threads: tuple[threading.Thread, threading.Thread]) -> bool:
    deadline = time.monotonic() + _OUTPUT_DRAIN_SECONDS
    for thread in threads:
        thread.join(timeout=max(deadline - time.monotonic(), 0))
    return not any(thread.is_alive() for thread in threads)


if __name__ == "__main__":
    raise SystemExit(main())
