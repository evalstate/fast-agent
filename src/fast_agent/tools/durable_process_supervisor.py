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
from typing import BinaryIO

from fast_agent.tools.durable_processes import (
    DurableProcessRecordError,
    DurableProcessStatus,
    DurableProcessStore,
    DurableProcessStream,
    _output_path,
    _read_spec,
    _stop_requested,
    _write_status,
    validate_process_id,
)

_HEARTBEAT_SECONDS = 1.0
_POLL_SECONDS = 0.1
_STOP_GRACE_SECONDS = 2.0
_STREAM_CHUNK_BYTES = 65536
_OUTPUT_DRAIN_SECONDS = 2.0


def main() -> int:
    """Run one process supervisor selected by explicit root and process ID."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--process-id", required=True)
    arguments = parser.parse_args()
    root = Path(arguments.root)
    process_id = arguments.process_id
    try:
        validate_process_id(process_id)
        store = DurableProcessStore(root)
        _supervise(store, process_id)
    except DurableProcessRecordError:
        return 2
    return 0


def _supervise(store: DurableProcessStore, process_id: str) -> None:
    directory = store.root / process_id
    spec = _read_spec(directory)
    started_at = time.time()
    child: subprocess.Popen[bytes] | None = None
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
            )
            if not _drain_output_threads(output_threads):
                raise OSError("Durable process output did not drain after process exit.")
            _raise_drain_failure(drain_failures)
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


def _wait_for_child(
    directory: Path,
    child: subprocess.Popen[bytes],
    *,
    started_at: float,
    drain_failures: SimpleQueue[BaseException],
) -> bool:
    stop_requested = False
    next_heartbeat = time.monotonic() + _HEARTBEAT_SECONDS
    while child.poll() is None:
        _raise_drain_failure(drain_failures)
        if not stop_requested and _stop_requested(directory):
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
) -> None:
    try:
        while payload := os.read(source.fileno(), _STREAM_CHUNK_BYTES):
            stream_output.write(payload)
            with combined_lock:
                combined_output.write(payload)
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
