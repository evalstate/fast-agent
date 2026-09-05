from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import signal
import stat
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING

import pytest
from mcp_types import TextContent

import fast_agent.tools.shell_runtime as shell_runtime_module
from fast_agent.config import Settings, ShellSettings
from fast_agent.constants import (
    DEFAULT_DURABLE_PROCESS_OUTPUT_RETENTION_BYTES,
    MAX_MANAGED_SHELL_PROCESSES,
)
from fast_agent.tools.durable_process_supervisor import (
    _drain_output,
    _raise_drain_failure,
)
from fast_agent.tools.durable_processes import (
    _OUTPUT_SEARCH_CHUNK_BYTES,
    DurableProcessError,
    DurableProcessRecordError,
    DurableProcessSnapshot,
    DurableProcessStore,
    DurableProcessStream,
)
from fast_agent.tools.local_shell_executor import LocalShellExecutor
from fast_agent.tools.shell_process import process_result_metadata
from fast_agent.tools.shell_runtime import ShellRuntime

if TYPE_CHECKING:
    from collections.abc import Buffer

_SHELL = Path("/bin/sh")


class _FailingWriter(io.BytesIO):
    def write(self, data: Buffer, /) -> int:
        del data
        raise OSError("simulated full output filesystem")


class _GatedLocalShellExecutor(LocalShellExecutor):
    def __init__(
        self,
        *,
        started: threading.Event,
        release: threading.Event,
        working_directory: Path,
    ) -> None:
        super().__init__(
            logger=logging.getLogger(__name__),
            working_directory=working_directory,
        )
        self._started = started
        self._release = release

    def start_durable_process(
        self,
        store: DurableProcessStore,
        *,
        command: str,
        cwd: Path,
        origin_session_id: str | None,
        agent_name: str | None,
        output_byte_limit: int,
        output_retention_byte_limit: int,
        max_active_processes: int,
    ) -> DurableProcessSnapshot:
        snapshot = super().start_durable_process(
            store,
            command=command,
            cwd=cwd,
            origin_session_id=origin_session_id,
            agent_name=agent_name,
            output_byte_limit=output_byte_limit,
            output_retention_byte_limit=output_retention_byte_limit,
            max_active_processes=max_active_processes,
        )
        self._started.set()
        if not self._release.wait(timeout=5):
            raise TimeoutError("test did not release durable launch")
        return snapshot


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_durable_process_completes_after_store_replacement_and_captures_logs(
    tmp_path: Path,
) -> None:
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(
        command="printf stdout; printf stderr >&2; exit 7",
        shell=_SHELL,
        cwd=tmp_path,
    )
    store.launch(created.spec.process_id, environment=dict(os.environ))

    replacement = DurableProcessStore(tmp_path / "durable")
    completed = replacement.wait(created.spec.process_id, timeout_seconds=5)
    stdout = replacement.read_output(
        created.spec.process_id,
        stream=DurableProcessStream.STDOUT,
        offset=0,
        limit=1024,
        query="stdout",
    )
    stderr = replacement.read_output(
        created.spec.process_id,
        stream=DurableProcessStream.STDERR,
        offset=0,
        limit=1024,
    )
    combined = replacement.read_output(
        created.spec.process_id,
        stream=DurableProcessStream.COMBINED,
        offset=0,
        limit=1024,
    )
    exact = replacement.read_output(
        created.spec.process_id,
        stream=DurableProcessStream.COMBINED,
        offset=0,
        limit=completed.output_bytes,
    )

    assert completed.status.state == "exited"
    assert completed.status.exit_code == 7
    assert replacement.discover() == [completed]
    assert stdout.text == "stdout"
    assert stdout.next_offset == len(b"stdout")
    assert stdout.match_count == 1
    assert stderr.text == "stderr"
    assert "stdout" in combined.text
    assert "stderr" in combined.text
    assert completed.output_bytes == len(combined.text.encode())
    assert exact.at_end is True


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_durable_process_caps_logs_while_continuing_to_drain(tmp_path: Path) -> None:
    script = tmp_path / "large_output.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'x' * 200000)\n"
        "sys.stdout.buffer.flush()\n"
        "sys.stderr.buffer.write(b'y' * 200000)\n"
        "sys.stderr.buffer.flush()\n",
        encoding="utf-8",
    )
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(
        command=f'exec "{sys.executable}" "{script}"',
        shell=_SHELL,
        cwd=tmp_path,
        output_retention_byte_limit=1024,
    )

    store.launch(created.spec.process_id, environment=dict(os.environ))
    completed = store.wait(created.spec.process_id, timeout_seconds=5)

    assert completed.status.state == "exited"
    assert completed.stdout_bytes == 1024
    assert completed.stderr_bytes == 1024
    assert completed.output_bytes == 1024
    assert completed.stdout_total_bytes == 200000
    assert completed.stderr_total_bytes == 200000
    assert completed.output_total_bytes == 400000
    assert completed.stdout_dropped_bytes == 200000 - 1024
    assert completed.stderr_dropped_bytes == 200000 - 1024
    assert completed.output_dropped_bytes == 400000 - 1024


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_silent_durable_process_does_not_rewrite_capture(tmp_path: Path) -> None:
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(command="sleep 30", shell=_SHELL, cwd=tmp_path)
    store.launch(created.spec.process_id, environment=dict(os.environ))
    store.wait_for_launch(created.spec.process_id, timeout_seconds=5)
    capture_path = store.directory(created.spec.process_id) / "capture.json"
    initial = capture_path.stat()

    try:
        time.sleep(0.35)
        unchanged = capture_path.stat()
    finally:
        store.request_stop(created.spec.process_id)
        store.wait(created.spec.process_id, timeout_seconds=5)

    assert unchanged.st_ino == initial.st_ino
    assert unchanged.st_mtime_ns == initial.st_mtime_ns


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_durable_poll_consumes_dropped_output_accounting(tmp_path: Path) -> None:
    root = tmp_path / "processes"
    script = tmp_path / "large_output.py"
    script.write_text(
        "import sys\nsys.stdout.buffer.write(b'x' * 200000)\nsys.stdout.buffer.flush()\n",
        encoding="utf-8",
    )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
        config=Settings(
            shell_execution=ShellSettings(durable_output_max_bytes=1024),
        ),
    )
    launch = await runtime.execute(
        {
            "command": f'exec "{sys.executable}" "{script}"',
            "background": True,
        }
    )
    launch_metadata = process_result_metadata(launch)
    assert launch_metadata is not None
    process_id = launch_metadata["process_id"]
    store = DurableProcessStore(root)
    await asyncio.to_thread(store.wait, process_id, timeout_seconds=5)

    try:
        first = await runtime.poll_process({"process_id": process_id, "wait_sec": 0})
        second = await runtime.poll_process({"process_id": process_id, "wait_sec": 0})
    finally:
        await runtime.close()

    first_metadata = process_result_metadata(first)
    second_metadata = process_result_metadata(second)
    assert first_metadata is not None
    assert second_metadata is not None
    assert first_metadata["output_bytes_since_last_poll"] == 200000
    assert first_metadata["retained_output_bytes_since_last_poll"] == 1024
    assert first_metadata["dropped_output_bytes_since_last_poll"] == 200000 - 1024
    assert first_metadata["output_truncated"] is True
    assert second_metadata["output_bytes_since_last_poll"] == 0
    assert second_metadata["retained_output_bytes_since_last_poll"] == 0
    assert second_metadata["dropped_output_bytes_since_last_poll"] == 0


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_durable_process_stop_is_file_backed_and_idempotent(tmp_path: Path) -> None:
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(
        command="trap 'printf stopped; exit 0' TERM; printf ready; while :; do sleep 1; done",
        shell=_SHELL,
        cwd=tmp_path,
    )
    store.launch(created.spec.process_id, environment=dict(os.environ))
    changed = store.wait_for_change(
        created.spec.process_id,
        previous=created,
        timeout_seconds=5,
    )

    assert changed.status.state in {"starting", "running"}
    assert store.request_stop(created.spec.process_id)
    assert not store.request_stop(created.spec.process_id)

    stopped = store.wait(created.spec.process_id, timeout_seconds=5)
    output = store.read_output(
        created.spec.process_id,
        stream=DurableProcessStream.STDOUT,
        offset=0,
        limit=1024,
    )

    assert stopped.status.state == "stopped"
    assert stopped.status.exit_code is not None
    assert "ready" in output.text


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_durable_process_rejects_malformed_records_and_stale_supervisors(tmp_path: Path) -> None:
    root = tmp_path / "durable"
    store = DurableProcessStore(root, heartbeat_timeout_seconds=0.01)
    malformed_id = f"process-{'a' * 32}"
    malformed = root / malformed_id
    malformed.mkdir()
    (malformed / "spec.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DurableProcessRecordError):
        store.get(malformed_id)
    with pytest.raises(ValueError, match="Invalid durable process ID"):
        store.get("process-../../etc")

    created = store.create(command="exit 0", shell=_SHELL, cwd=tmp_path)
    status_path = root / created.spec.process_id / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "version": 1,
                "state": "running",
                "exit_code": None,
                "updated_at": 0.0,
                "heartbeat_at": 0.0,
                "supervisor_pid": 1,
                "child_pid": 2,
                "started_at": 0.0,
            }
        ),
        encoding="utf-8",
    )

    stale = store.get(created.spec.process_id)

    assert stale.status.state == "unavailable"


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_invalid_utf8_record_is_skipped_as_malformed(tmp_path: Path) -> None:
    store = DurableProcessStore(tmp_path / "durable")
    valid = store.create(command="exit 0", shell=_SHELL, cwd=tmp_path)
    corrupt = store.create(command="exit 1", shell=_SHELL, cwd=tmp_path)
    status_path = store.directory(corrupt.spec.process_id) / "status.json"
    status_path.write_bytes(b"\x80")

    with pytest.raises(DurableProcessRecordError) as error:
        store.get(corrupt.spec.process_id)

    assert isinstance(error.value.__cause__, UnicodeDecodeError)
    assert [snapshot.spec.process_id for snapshot in store.discover()] == [valid.spec.process_id]
    additional = store.create(
        command="exit 2",
        shell=_SHELL,
        cwd=tmp_path,
        max_active_processes=2,
    )
    assert {snapshot.spec.process_id for snapshot in store.discover()} == {
        valid.spec.process_id,
        additional.spec.process_id,
    }


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_discovery_persists_unavailable_when_supervisor_and_child_disappeared(
    tmp_path: Path,
) -> None:
    root = tmp_path / "durable"
    store = DurableProcessStore(root)
    created = store.create(command="sleep 30", shell=_SHELL, cwd=tmp_path)
    status_path = store.directory(created.spec.process_id) / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "state": "running",
            "supervisor_pid": 99_999_999,
            "child_pid": 99_999_998,
        }
    )
    status_path.write_text(json.dumps(status), encoding="utf-8")

    discovered = store.discover()
    persisted = json.loads(status_path.read_text(encoding="utf-8"))

    assert len(discovered) == 1
    assert discovered[0].status.state == "unavailable"
    assert persisted["state"] == "unavailable"
    assert (root / created.spec.process_id).exists()


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_discovery_retains_record_with_surviving_orphaned_child(tmp_path: Path) -> None:
    store = DurableProcessStore(
        tmp_path / "durable",
        heartbeat_timeout_seconds=0.01,
        max_terminal_records=0,
    )
    created = store.create(command="sleep 30", shell=_SHELL, cwd=tmp_path)
    status_path = store.directory(created.spec.process_id) / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "state": "running",
            "updated_at": 0.0,
            "heartbeat_at": 0.0,
            "supervisor_pid": 99_999_999,
            "child_pid": os.getpid(),
        }
    )
    status_path.write_text(json.dumps(status), encoding="utf-8")

    discovered = store.discover()
    persisted = json.loads(status_path.read_text(encoding="utf-8"))

    assert len(discovered) == 1
    assert discovered[0].status.state == "unavailable"
    assert persisted["state"] == "running"
    assert store.prune_terminal_records() == 0
    assert store.directory(created.spec.process_id).exists()


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_discovery_handles_out_of_range_process_ids(tmp_path: Path) -> None:
    root = tmp_path / "durable"
    store = DurableProcessStore(root)
    created = store.create(command="sleep 30", shell=_SHELL, cwd=tmp_path)
    status_path = store.directory(created.spec.process_id) / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "state": "running",
            "supervisor_pid": 10**100,
            "child_pid": 10**100,
        }
    )
    status_path.write_text(json.dumps(status), encoding="utf-8")

    discovered = store.discover()

    assert len(discovered) == 1
    assert discovered[0].status.state == "unavailable"
    assert (root / created.spec.process_id).exists()


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_retention_prunes_stale_launch_with_unpublished_process_ids(
    tmp_path: Path,
) -> None:
    store = DurableProcessStore(
        tmp_path / "durable",
        heartbeat_timeout_seconds=0.01,
        max_terminal_records=0,
    )
    created = store.create(command="sleep 30", shell=_SHELL, cwd=tmp_path)
    status_path = store.directory(created.spec.process_id) / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "state": "starting",
            "updated_at": 0.0,
            "heartbeat_at": 0.0,
            "supervisor_pid": None,
            "child_pid": None,
        }
    )
    status_path.write_text(json.dumps(status), encoding="utf-8")

    assert store.prune_terminal_records() == 1
    assert not (store.root / created.spec.process_id).exists()


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_terminal_record_retention_prunes_oldest_records(tmp_path: Path) -> None:
    root = tmp_path / "durable"
    store = DurableProcessStore(root, max_terminal_records=10)
    stale = store.create(command="sleep 30", shell=_SHELL, cwd=tmp_path)
    stale_status_path = store.directory(stale.spec.process_id) / "status.json"
    stale_status = json.loads(stale_status_path.read_text(encoding="utf-8"))
    stale_status.update(
        {
            "state": "running",
            "updated_at": 0.0,
            "heartbeat_at": 0.0,
            "supervisor_pid": 99_999_999,
            "child_pid": 99_999_998,
            "started_at": 0.0,
        }
    )
    stale_status_path.write_text(json.dumps(stale_status), encoding="utf-8")
    terminal_ids: list[str] = []
    for index in range(3):
        created = store.create(command=f"exit {index}", shell=_SHELL, cwd=tmp_path)
        terminal_ids.append(created.spec.process_id)
        status_path = store.directory(created.spec.process_id) / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update(
            {
                "state": "exited",
                "exit_code": index,
                "updated_at": float(index + 1),
                "heartbeat_at": float(index + 1),
            }
        )
        status_path.write_text(json.dumps(status), encoding="utf-8")
    control = store.directory(terminal_ids[0]) / "control"
    control.mkdir()
    (control / "stop.json").write_text("{}", encoding="utf-8")
    active = store.create(command="sleep 30", shell=_SHELL, cwd=tmp_path)

    replacement = DurableProcessStore(root, max_terminal_records=2)
    discovered_ids = {snapshot.spec.process_id for snapshot in replacement.discover()}

    assert discovered_ids == {terminal_ids[1], terminal_ids[2], active.spec.process_id}
    assert not (root / stale.spec.process_id).exists()
    assert not (root / terminal_ids[0]).exists()


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_supervisor_honors_terminal_record_retention(tmp_path: Path) -> None:
    root = tmp_path / "durable"
    store = DurableProcessStore(root, max_terminal_records=1)
    for _ in range(2):
        created = store.create(command="exit 0", shell=_SHELL, cwd=tmp_path)
        store.launch(created.spec.process_id, environment=dict(os.environ))
        store.wait(created.spec.process_id, timeout_seconds=5)

    deadline = time.monotonic() + 2
    while len(store.discover()) > 1 and time.monotonic() < deadline:
        time.sleep(0.02)

    assert len(store.discover()) == 1


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_durable_process_records_are_private_and_spec_is_versioned(tmp_path: Path) -> None:
    root = tmp_path / "durable"
    store = DurableProcessStore(root)
    created = store.create(command="exit 0", shell=_SHELL, cwd=tmp_path)
    directory = root / created.spec.process_id

    spec = json.loads((directory / "spec.json").read_text(encoding="utf-8"))

    assert spec["version"] == 1
    assert spec["output_byte_limit"] > 0
    assert "environment" not in spec
    assert _mode(root) == 0o700
    assert _mode(directory) == 0o700
    assert _mode(directory / "spec.json") == 0o400
    assert _mode(directory / "status.json") == 0o600
    assert _mode(directory / "stdout.log") == 0o600
    assert _mode(directory / "stderr.log") == 0o600
    assert _mode(directory / "output.log") == 0o600
    assert store.request_stop(created.spec.process_id)
    assert _mode(directory / "control") == 0o700
    assert _mode(directory / "control" / "stop.json") == 0o600


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_legacy_spec_and_missing_capture_remain_readable(tmp_path: Path) -> None:
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(command="exit 0", shell=_SHELL, cwd=tmp_path)
    spec_path = store.directory(created.spec.process_id) / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec.pop("output_retention_byte_limit")
    spec_path.chmod(0o600)
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    snapshot = DurableProcessStore(store.root).get(created.spec.process_id)

    assert (
        snapshot.spec.output_retention_byte_limit == DEFAULT_DURABLE_PROCESS_OUTPUT_RETENTION_BYTES
    )
    assert snapshot.output_total_bytes == snapshot.output_bytes == 0
    assert snapshot.output_dropped_bytes == 0


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_durable_process_capacity_is_atomic(tmp_path: Path) -> None:
    store = DurableProcessStore(tmp_path / "durable")

    def create(index: int) -> bool:
        try:
            store.create(
                command=f"sleep {index + 1}",
                shell=_SHELL,
                cwd=tmp_path,
                max_active_processes=1,
            )
        except DurableProcessError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        created = list(executor.map(create, range(2)))

    assert created.count(True) == 1


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_supervisor_publishes_small_output_before_process_exit(tmp_path: Path) -> None:
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(
        command="printf ready; sleep 30",
        shell=_SHELL,
        cwd=tmp_path,
    )
    store.launch(created.spec.process_id, environment=dict(os.environ))
    running = store.wait_for_launch(created.spec.process_id, timeout_seconds=5)

    try:
        deadline = time.monotonic() + 1
        while running.output_bytes == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
            running = store.get(created.spec.process_id)
        output = store.read_output(
            created.spec.process_id,
            stream=DurableProcessStream.COMBINED,
            offset=0,
            limit=1024,
        )
        assert running.status.state == "running"
        assert output.text == "ready"
    finally:
        store.request_stop(created.spec.process_id)
        store.wait(created.spec.process_id, timeout_seconds=5)


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_supervisor_import_is_isolated_from_command_working_directory(tmp_path: Path) -> None:
    working_directory = tmp_path / "command"
    shadow_package = working_directory / "fast_agent" / "tools"
    shadow_package.mkdir(parents=True)
    marker = tmp_path / "shadow-imported"
    (shadow_package.parent / "__init__.py").write_text("", encoding="utf-8")
    (shadow_package / "__init__.py").write_text("", encoding="utf-8")
    (shadow_package / "durable_process_supervisor.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('shadow')\n",
        encoding="utf-8",
    )
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(
        command="printf child-ran",
        shell=_SHELL,
        cwd=working_directory,
    )

    store.launch(created.spec.process_id, environment=dict(os.environ))
    completed = store.wait(created.spec.process_id, timeout_seconds=5)
    output = store.read_output(
        created.spec.process_id,
        stream=DurableProcessStream.COMBINED,
        offset=0,
        limit=1024,
    )

    assert completed.status.state == "exited"
    assert output.text == "child-ran"
    assert not marker.exists()


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_drain_thread_failure_is_propagated() -> None:
    read_descriptor, write_descriptor = os.pipe()
    os.write(write_descriptor, b"output")
    os.close(write_descriptor)
    failures: SimpleQueue[BaseException] = SimpleQueue()
    with os.fdopen(read_descriptor, "rb") as source:
        thread = threading.Thread(
            target=_drain_output,
            args=(
                source,
                _FailingWriter(),
                io.BytesIO(),
                threading.Lock(),
                failures,
            ),
        )
        thread.start()
        thread.join(timeout=1)

    assert thread.is_alive() is False
    with pytest.raises(OSError, match="Could not drain durable process output"):
        _raise_drain_failure(failures)


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_durable_launch_is_claimed_once_and_combined_output_has_one_cursor(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "launches"
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(
        command=f"printf x >> '{marker}'; printf 12345; printf abcdefgh >&2; sleep 0.2",
        shell=_SHELL,
        cwd=tmp_path,
    )

    def launch() -> bool:
        try:
            store.launch(created.spec.process_id, environment=dict(os.environ))
        except DurableProcessError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        launched = list(executor.map(lambda _: launch(), range(2)))

    completed = store.wait(created.spec.process_id, timeout_seconds=5)
    chunks: list[str] = []
    offset = 0
    while True:
        output = store.read_output(
            created.spec.process_id,
            stream=DurableProcessStream.COMBINED,
            offset=offset,
            limit=3,
        )
        chunks.append(output.text)
        offset = output.next_offset
        if output.at_end:
            break

    assert launched.count(True) == 1
    assert marker.read_text(encoding="utf-8") == "x"
    assert completed.status.state == "exited"
    assert sorted("".join(chunks)) == sorted("12345abcdefgh")


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_durable_output_query_searches_beyond_response_limit(tmp_path: Path) -> None:
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(command="exit 0", shell=_SHELL, cwd=tmp_path)
    output_path = store.directory(created.spec.process_id) / "output.log"
    output_path.write_text(
        ("ordinary output\n" * 20) + "MATCH one\nMATCH two\n",
        encoding="utf-8",
    )

    limited = store.read_output(
        created.spec.process_id,
        stream=DurableProcessStream.COMBINED,
        offset=0,
        limit=len("MATCH one\n"),
        query="MATCH",
    )
    complete = store.read_output(
        created.spec.process_id,
        stream=DurableProcessStream.COMBINED,
        offset=0,
        limit=1024,
        query="MATCH",
    )

    assert limited.text == "MATCH one\n"
    assert limited.match_count == 2
    assert limited.next_offset == len(("ordinary output\n" * 20 + "MATCH one\n").encode())
    assert limited.at_end is False
    assert complete.text == "MATCH one\nMATCH two\n"
    assert complete.match_count == 2
    assert complete.at_end is True

    continued = store.read_output(
        created.spec.process_id,
        stream=DurableProcessStream.COMBINED,
        offset=limited.next_offset,
        limit=1024,
        query="MATCH",
    )
    assert continued.text == "MATCH two\n"
    assert continued.match_count == 1
    assert continued.at_end is True


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_durable_output_clamps_offset_beyond_current_eof(tmp_path: Path) -> None:
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(command="exit 0", shell=_SHELL, cwd=tmp_path)
    output_path = store.directory(created.spec.process_id) / "output.log"
    output_path.write_text("first", encoding="utf-8")

    at_eof = store.read_output(
        created.spec.process_id,
        stream=DurableProcessStream.COMBINED,
        offset=100,
        limit=1024,
    )
    output_path.write_text("first second", encoding="utf-8")
    continued = store.read_output(
        created.spec.process_id,
        stream=DurableProcessStream.COMBINED,
        offset=at_eof.next_offset,
        limit=1024,
    )

    assert at_eof.offset == len(b"first")
    assert at_eof.next_offset == len(b"first")
    assert at_eof.at_end is True
    assert continued.text == " second"
    assert continued.next_offset == len(b"first second")


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
def test_durable_output_query_scans_bounded_chunks_across_boundary(tmp_path: Path) -> None:
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(command="exit 0", shell=_SHELL, cwd=tmp_path)
    output_path = store.directory(created.spec.process_id) / "output.log"
    output_path.write_bytes(
        b"x" * (_OUTPUT_SEARCH_CHUNK_BYTES - 2) + b"MATCH" + b"y" * (_OUTPUT_SEARCH_CHUNK_BYTES * 2)
    )

    output = store.read_output(
        created.spec.process_id,
        stream=DurableProcessStream.COMBINED,
        offset=0,
        limit=32,
        query="MATCH",
    )

    assert output.text == "x" * 32
    assert output.returned_bytes == 32
    assert output.match_count == 1
    assert output.next_offset == output_path.stat().st_size
    assert output.at_end is True

    with pytest.raises(ValueError, match="query must be at most 512 characters"):
        store.read_output(
            created.spec.process_id,
            stream=DurableProcessStream.COMBINED,
            offset=0,
            limit=32,
            query="x" * 513,
        )


@pytest.mark.unit
@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="process-group descendant assertion uses /proc",
)
def test_supervisor_cleans_up_descendants_after_process_leader_exits(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    script = tmp_path / "fork_child.py"
    script.write_text(
        "import os, pathlib, time\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    time.sleep(30)\n"
        "    os._exit(0)\n"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child))\n"
        "os._exit(0)\n",
        encoding="utf-8",
    )
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(
        command=f'exec "{sys.executable}" "{script}"',
        shell=_SHELL,
        cwd=tmp_path,
    )
    store.launch(created.spec.process_id, environment=dict(os.environ))

    completed = store.wait(created.spec.process_id, timeout_seconds=5)
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))

    assert completed.status.state == "exited"
    assert not _linux_process_is_running(child_pid)


@pytest.mark.unit
@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="orphan assertion uses /proc",
)
def test_supervisor_failure_cleans_up_child_process_group(tmp_path: Path) -> None:
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(command="sleep 30", shell=_SHELL, cwd=tmp_path)
    store.launch(created.spec.process_id, environment=dict(os.environ))
    running = store.wait_for_launch(created.spec.process_id, timeout_seconds=5)
    assert running.status.child_pid is not None

    control = store.directory(created.spec.process_id) / "control"
    control.mkdir(mode=0o700)
    (control / "stop.json").write_text("{}", encoding="utf-8")

    unavailable = store.wait(created.spec.process_id, timeout_seconds=5)

    assert unavailable.status.state == "unavailable"
    assert not _linux_process_is_running(running.status.child_pid)


@pytest.mark.unit
@pytest.mark.parametrize("termination_signal", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="orphan assertion uses /proc",
)
def test_supervisor_signal_stops_child_process_group(
    tmp_path: Path,
    termination_signal: signal.Signals,
) -> None:
    store = DurableProcessStore(tmp_path / "durable")
    created = store.create(command="sleep 30", shell=_SHELL, cwd=tmp_path)
    store.launch(created.spec.process_id, environment=dict(os.environ))
    running = store.wait_for_launch(created.spec.process_id, timeout_seconds=5)
    assert running.status.supervisor_pid is not None
    assert running.status.child_pid is not None

    os.kill(running.status.supervisor_pid, termination_signal)
    stopped = store.wait(created.spec.process_id, timeout_seconds=5)

    assert stopped.status.state == "stopped"
    assert stopped.status.exit_code is not None
    assert not _linux_process_is_running(running.status.child_pid)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_shell_runtime_can_attach_and_stop_process_after_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processes"
    command = (
        f'exec "{sys.executable}" -c "import time; print(\'ready\', flush=True); time.sleep(30)"'
    )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
        session_id_provider=lambda: "session-one",
    )
    result = await runtime.execute({"command": command, "background": True})
    metadata = process_result_metadata(result)
    assert metadata is not None
    process_id = metadata["process_id"]
    await runtime.close()

    replacement = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )
    try:
        discovered = await replacement.discover_durable_processes()
        assert [snapshot.spec.process_id for snapshot in discovered] == [process_id]
        assert discovered[0].session_ids == ("session-one",)

        attached = await replacement.attach_durable_process(
            process_id,
            session_id="session-two",
        )
        assert attached.session_ids == ("session-one", "session-two")

        status = await replacement.poll_process(
            {"process_id": process_id, "wait_sec": 1, "wake_on_output": True}
        )
        status_metadata = process_result_metadata(status)
        assert status_metadata is not None
        assert status_metadata["process_status"] == "running"

        stopped = await replacement.terminate_process({"process_id": process_id})
        stopped_metadata = process_result_metadata(stopped)
        assert stopped_metadata is not None
        assert stopped_metadata["process_status"] == "stopping"
        terminal = await asyncio.to_thread(
            DurableProcessStore(root).wait,
            process_id,
            timeout_seconds=5,
        )
        assert terminal.status.state == "stopped"
    finally:
        await replacement.close()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_cancelled_durable_launch_is_registered_before_cancellation_propagates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processes"
    started = threading.Event()
    release = threading.Event()
    executor = _GatedLocalShellExecutor(
        started=started,
        release=release,
        working_directory=tmp_path,
    )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
        shell_environment=executor,
    )
    launch_task = asyncio.create_task(runtime.execute({"command": "sleep 30", "background": True}))

    assert await asyncio.to_thread(started.wait, 5)
    launch_task.cancel()
    await asyncio.sleep(0)
    launch_task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await launch_task

    discovered = await runtime.discover_durable_processes()
    snapshots = await runtime.process_snapshots()
    assert len(discovered) == 1
    assert [snapshot.process_id for snapshot in snapshots] == [discovered[0].spec.process_id]

    await runtime.terminate_process({"process_id": discovered[0].spec.process_id})
    await asyncio.to_thread(
        DurableProcessStore(root).wait,
        discovered[0].spec.process_id,
        timeout_seconds=5,
    )
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_shell_runtime_does_not_claim_stale_process_was_stopped(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processes"
    store = DurableProcessStore(root, heartbeat_timeout_seconds=0.01)
    created = store.create(command="exit 0", shell=_SHELL, cwd=tmp_path)
    status_path = root / created.spec.process_id / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "version": 1,
                "state": "running",
                "exit_code": None,
                "updated_at": 0.0,
                "heartbeat_at": 0.0,
                "supervisor_pid": 1,
                "child_pid": 2,
                "started_at": 0.0,
            }
        ),
        encoding="utf-8",
    )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )
    try:
        await runtime.attach_durable_process(created.spec.process_id)
        result = await runtime.terminate_process({"process_id": created.spec.process_id})
    finally:
        await runtime.close()

    metadata = process_result_metadata(result)
    assert result.is_error is True
    assert metadata is not None
    assert metadata["process_status"] == "unavailable"
    assert not (root / created.spec.process_id / "control" / "stop.json").exists()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_shell_runtime_counts_unattached_durable_processes_for_capacity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processes"
    store = DurableProcessStore(root)
    for index in range(MAX_MANAGED_SHELL_PROCESSES):
        store.create(
            command=f"sleep {index + 1}",
            shell=_SHELL,
            cwd=tmp_path,
        )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )

    durable_result = await runtime.execute({"command": "sleep 30", "background": True})
    session_result = await runtime.execute(
        {
            "command": "sleep 30",
            "background": True,
            "lifecycle": "session",
        }
    )

    for result in (durable_result, session_result):
        assert result.is_error is True
        assert isinstance(result.content[0], TextContent)
        assert f"at most {MAX_MANAGED_SHELL_PROCESSES}" in result.content[0].text
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_shell_runtime_disables_durability_when_store_is_unavailable(
    tmp_path: Path,
) -> None:
    durable_root = tmp_path / "processes"
    durable_root.write_text("not a directory", encoding="utf-8")

    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=durable_root,
    )
    try:
        result = await runtime.execute({"command": "printf foreground"})
    finally:
        await runtime.close()

    assert runtime.durable_process_root is None
    assert result.is_error is False
    assert isinstance(result.content[0], TextContent)
    assert "foreground" in result.content[0].text


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_durable_poll_waits_for_unread_output_to_settle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(shell_runtime_module, "_PROCESS_OUTPUT_DEBOUNCE_SECONDS", 0.1)
    root = tmp_path / "processes"
    script = tmp_path / "burst.py"
    script.write_text(
        "import time\n"
        "print('first', flush=True)\n"
        "time.sleep(0.03)\n"
        "print('second', flush=True)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )
    result = await runtime.execute(
        {
            "command": f'exec "{sys.executable}" "{script}"',
            "background": True,
        }
    )
    metadata = process_result_metadata(result)
    assert metadata is not None
    process_id = metadata["process_id"]
    store = DurableProcessStore(root)
    deadline = time.monotonic() + 2
    while store.get(process_id).output_bytes == 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert store.get(process_id).output_bytes > 0

    try:
        started = time.monotonic()
        poll_result = await runtime.poll_process(
            {
                "process_id": process_id,
                "wait_sec": 4,
                "wake_on_output": True,
            }
        )
        elapsed = time.monotonic() - started
    finally:
        await runtime.terminate_process({"process_id": process_id})
        await asyncio.to_thread(store.wait, process_id, timeout_seconds=5)
        await runtime.close()

    poll_metadata = process_result_metadata(poll_result)
    assert poll_metadata is not None
    assert poll_metadata["process_yield_reason"] == "output"
    assert 0.08 <= elapsed < 1.0
    assert poll_metadata["output_bytes_since_last_poll"] > 0
    assert poll_metadata["seconds_since_last_output"] >= 0.08
    assert poll_result.content
    assert isinstance(poll_result.content[0], TextContent)
    assert "first" in poll_result.content[0].text
    assert "second" in poll_result.content[0].text


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_cancelled_durable_poll_preserves_output_debounce(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(shell_runtime_module, "_PROCESS_OUTPUT_DEBOUNCE_SECONDS", 0.2)
    root = tmp_path / "processes"
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )
    result = await runtime.execute(
        {
            "command": "printf ready; sleep 30",
            "background": True,
        }
    )
    metadata = process_result_metadata(result)
    assert metadata is not None
    process_id = metadata["process_id"]
    store = DurableProcessStore(root)
    deadline = time.monotonic() + 2
    while store.get(process_id).output_bytes == 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert store.get(process_id).output_bytes > 0

    try:
        first_poll = asyncio.create_task(
            runtime.poll_process(
                {
                    "process_id": process_id,
                    "wait_sec": 4,
                    "wake_on_output": True,
                }
            )
        )
        await asyncio.sleep(0.05)
        first_poll.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_poll

        started = time.monotonic()
        poll_result = await runtime.poll_process(
            {
                "process_id": process_id,
                "wait_sec": 3,
                "wake_on_output": True,
            }
        )
        elapsed = time.monotonic() - started
    finally:
        await runtime.terminate_process({"process_id": process_id})
        await asyncio.to_thread(store.wait, process_id, timeout_seconds=5)
        await runtime.close()

    poll_metadata = process_result_metadata(poll_result)
    assert poll_metadata is not None
    assert poll_metadata["process_yield_reason"] == "output"
    assert 0.08 <= elapsed < 0.5


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_continuous_durable_output_waits_until_poll_deadline(tmp_path: Path) -> None:
    root = tmp_path / "processes"
    script = tmp_path / "continuous.py"
    script.write_text(
        "import time\n"
        "for index in range(30):\n"
        "    print(index, flush=True)\n"
        "    time.sleep(0.1)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )
    result = await runtime.execute(
        {
            "command": f'exec "{sys.executable}" "{script}"',
            "background": True,
        }
    )
    metadata = process_result_metadata(result)
    assert metadata is not None
    process_id = metadata["process_id"]
    store = DurableProcessStore(root)

    try:
        started = time.monotonic()
        poll_result = await runtime.poll_process(
            {
                "process_id": process_id,
                "wait_sec": 1,
                "wake_on_output": True,
            }
        )
        elapsed = time.monotonic() - started
    finally:
        await runtime.terminate_process({"process_id": process_id})
        await asyncio.to_thread(store.wait, process_id, timeout_seconds=5)
        await runtime.close()

    poll_metadata = process_result_metadata(poll_result)
    assert poll_metadata is not None
    assert poll_metadata["process_yield_reason"] == "deadline"
    assert elapsed >= 0.8
    assert poll_metadata["output_bytes_since_last_poll"] > 0
    assert poll_metadata["poll_deadline_overshoot_seconds"] >= 0


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
@pytest.mark.parametrize("action", ["wait", "status"])
@pytest.mark.parametrize("limit", [1, 999, 1000])
async def test_unified_durable_preview_limit(tmp_path: Path, action: str, limit: int) -> None:
    root = tmp_path / "processes"
    gate = tmp_path / "finish"
    exit_gate = tmp_path / "exit"
    script = tmp_path / "preview.py"
    script.write_text(
        "import pathlib,sys,time\n"
        f"gate = pathlib.Path({str(gate)!r})\n"
        "print('é' * 1000, flush=True)\n"
        "while not gate.exists(): time.sleep(0.01)\n"
        "print('z' * 1500, flush=True)\n"
        f"while not pathlib.Path({str(exit_gate)!r}).exists(): time.sleep(0.01)\n"
        "print('y' * 1500, flush=True)\n"
        "sys.exit(7)\n"
    )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
        output_byte_limit=4096,
        config=Settings(shell_execution=ShellSettings(tool_profile="minimal_process")),
    )
    launch = await runtime.call_tool(
        "bash", {"command": f'"{sys.executable}" "{script}"', "run_in_background": True}
    )
    metadata = process_result_metadata(launch)
    assert metadata is not None
    process_id = metadata["process_id"]
    store = DurableProcessStore(root)
    try:
        async with asyncio.timeout(10):
            while store.get(process_id).output_bytes < 2001:
                await asyncio.sleep(0.02)
        status = await runtime.call_tool(
            "process", {"process_id": process_id, "action": "status", "limit": limit}
        )
        status_metadata = process_result_metadata(status)
        assert status_metadata is not None
        assert status_metadata["process_status"] == "running"
        assert isinstance(status.content[0], TextContent)
        preview = status.content[0].text.split("[Output truncated:", 1)[0].rstrip("\n")
        assert preview == "é" * (limit // 2)
        assert "[Output truncated:" in status.content[0].text

        retained = await runtime.call_tool(
            "process", {"process_id": process_id, "action": "read_output", "limit": 3000}
        )
        assert isinstance(retained.content[0], TextContent)
        assert json.loads(retained.content[0].text)["content"] == "é" * 1000 + "\n"

        gate.touch()
        async with asyncio.timeout(10):
            while store.get(process_id).output_bytes < 3502:
                await asyncio.sleep(0.02)
        if action == "wait":
            exit_gate.touch()
            await asyncio.to_thread(store.wait, process_id, timeout_seconds=10)
        # Concurrent polls consume each batch once, without sharing a mutable limit.
        async with asyncio.TaskGroup() as group:
            limited = group.create_task(
                runtime.call_tool(
                    "process", {"process_id": process_id, "action": action, "limit": limit}
                )
            )
            default = group.create_task(
                runtime.call_tool("process", {"process_id": process_id, "action": "status"})
            )
        limited_result = limited.result()
        assert isinstance(limited_result.content[0], TextContent)
        assert "z" * limit + "\n[Output truncated:" in limited_result.content[0].text
        for result in (limited_result, default.result()):
            result_metadata = process_result_metadata(result)
            assert result_metadata is not None
            if action == "wait":
                assert result_metadata["exit_code"] == 7
            else:
                assert result_metadata["process_status"] == "running"
        assert store.get(process_id).spec.output_byte_limit == 4096
        exit_gate.touch()
        await asyncio.to_thread(store.wait, process_id, timeout_seconds=10)
        default_output = await runtime.call_tool(
            "process", {"process_id": process_id, "action": "status"}
        )
        assert isinstance(default_output.content[0], TextContent)
        if action == "status":
            assert "y" * 1500 in default_output.content[0].text
        assert "[Output truncated:" not in default_output.content[0].text
        final_metadata = process_result_metadata(default_output)
        assert final_metadata is not None
        assert final_metadata["exit_code"] == 7
    finally:
        gate.touch()
        exit_gate.touch()
        await runtime.terminate_process({"process_id": process_id})
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_durable_poll_preserves_per_command_output_limit(tmp_path: Path) -> None:
    root = tmp_path / "processes"
    script = tmp_path / "output_limit.py"
    script.write_text(
        "import sys, time\nsys.stdout.write('x' * 1000)\nsys.stdout.flush()\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
        output_byte_limit=1000,
    )
    launch = await runtime.execute(
        {
            "command": f'exec "{sys.executable}" "{script}"',
            "background": True,
            "output_byte_limit": 80,
        }
    )
    launch_metadata = process_result_metadata(launch)
    assert launch_metadata is not None
    process_id = launch_metadata["process_id"]
    store = DurableProcessStore(root)

    try:
        deadline = time.monotonic() + 2
        snapshot = store.get(process_id)
        while snapshot.output_bytes < 1000 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
            snapshot = store.get(process_id)

        first = await runtime.poll_process({"process_id": process_id, "wait_sec": 0})
        first_metadata = process_result_metadata(first)
        assert first_metadata is not None
        assert first_metadata["output_bytes_since_last_poll"] == 1000
        assert isinstance(first.content[0], TextContent)
        assert "showing first 40 bytes and last 40 bytes" in first.content[0].text

        second = await runtime.poll_process({"process_id": process_id, "wait_sec": 0})
        assert isinstance(second.content[0], TextContent)
        assert "xxx" not in second.content[0].text
    finally:
        await runtime.terminate_process({"process_id": process_id})
        await asyncio.to_thread(store.wait, process_id, timeout_seconds=5)
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_durable_processes_use_independent_poll_locks(tmp_path: Path) -> None:
    root = tmp_path / "processes"
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )
    process_ids: list[str] = []
    for _ in range(2):
        launch = await runtime.execute({"command": "sleep 30", "background": True})
        metadata = process_result_metadata(launch)
        assert metadata is not None
        process_ids.append(metadata["process_id"])
    store = DurableProcessStore(root)

    try:
        waiting_poll = asyncio.create_task(
            runtime.poll_process({"process_id": process_ids[0], "wait_sec": 1})
        )
        await asyncio.sleep(0.1)

        started_at = time.monotonic()
        independent_poll = await asyncio.wait_for(
            runtime.poll_process({"process_id": process_ids[1], "wait_sec": 0}),
            timeout=0.5,
        )

        assert time.monotonic() - started_at < 0.5
        independent_metadata = process_result_metadata(independent_poll)
        assert independent_metadata is not None
        assert independent_metadata["process_status"] == "running"
        assert not waiting_poll.done()
    finally:
        if not waiting_poll.done():
            waiting_poll.cancel()
            await asyncio.gather(waiting_poll, return_exceptions=True)
        for process_id in process_ids:
            await runtime.terminate_process({"process_id": process_id})
            await asyncio.to_thread(store.wait, process_id, timeout_seconds=5)
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_terminal_durable_launch_result_does_not_claim_process_is_running(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processes"
    store = DurableProcessStore(root)
    created = store.create(command="exit 0", shell=_SHELL, cwd=tmp_path)
    store.launch(created.spec.process_id, environment=dict(os.environ))
    completed = store.wait(created.spec.process_id, timeout_seconds=5)
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )

    result = runtime._durable_launch_result(completed)

    metadata = process_result_metadata(result)
    assert metadata is not None
    assert metadata["process_status"] == "completed"
    assert isinstance(result.content[0], TextContent)
    assert "completed before launch acknowledgement" in result.content[0].text
    assert "still running" not in result.content[0].text
    await runtime.close()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _linux_process_is_running(process_id: int) -> bool:
    try:
        stat_fields = Path(f"/proc/{process_id}/stat").read_text(encoding="utf-8").split()
    except FileNotFoundError:
        return False
    return len(stat_fields) > 2 and stat_fields[2] != "Z"
