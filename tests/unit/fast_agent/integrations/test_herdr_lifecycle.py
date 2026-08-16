from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
from contextlib import closing
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.integrations import herdr_lifecycle

if TYPE_CHECKING:
    from pathlib import Path


def _configure_herdr(
    monkeypatch: pytest.MonkeyPatch,
    socket_path: Path,
    *,
    bin_path: str = "/usr/bin/herdr",
) -> None:
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.setenv("HERDR_SOCKET_PATH", str(socket_path))
    monkeypatch.setenv("HERDR_BIN_PATH", bin_path)
    monkeypatch.setenv("HERDR_PANE_ID", "w1:p2")


def _capture_requests(
    socket_path: Path, count: int
) -> tuple[list[dict[str, object]], threading.Thread]:
    requests: list[dict[str, object]] = []
    ready = threading.Event()

    def serve() -> None:
        with closing(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)) as server:
            server.bind(str(socket_path))
            server.listen()
            ready.set()
            for _ in range(count):
                connection, _ = server.accept()
                with connection:
                    payload = b""
                    while not payload.endswith(b"\n"):
                        payload += connection.recv(65536)
                    decoded = json.loads(payload)
                    assert isinstance(decoded, dict)
                    response = json.dumps({"id": decoded["id"], "result": {"type": "ok"}}).encode()
                    connection.sendall(response + b"\n")
                requests.append(decoded)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(1)
    return requests, thread


def _wait_for_requests(requests: list[dict[str, object]], count: int) -> None:
    deadline = time.monotonic() + 1
    while len(requests) < count and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(requests) == count


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets are unavailable")
def test_reports_prompt_lifecycle_in_order_and_releases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "herdr.sock"
    requests, server = _capture_requests(socket_path, 4)
    _configure_herdr(monkeypatch, socket_path)

    herdr_lifecycle.report_prompt_mark("A")
    herdr_lifecycle.report_prompt_mark("C")
    herdr_lifecycle.report_agent_state(cast("herdr_lifecycle.HerdrBaseState", "blocked"))
    herdr_lifecycle.report_prompt_mark("D;0")
    _wait_for_requests(requests, 3)
    herdr_lifecycle.release_agent()

    server.join(1)
    assert not server.is_alive()
    assert [request["method"] for request in requests] == [
        "pane.report_agent",
        "pane.report_agent",
        "pane.report_agent",
        "pane.release_agent",
    ]

    params: list[dict[str, object]] = []
    for request in requests:
        item = request["params"]
        assert isinstance(item, dict)
        assert all(isinstance(key, str) for key in item)
        params.append({key: value for key, value in item.items() if isinstance(key, str)})
    assert [item.get("state") for item in params] == ["idle", "working", "idle", None]
    sequences: list[int] = []
    for item in params:
        seq = item["seq"]
        assert isinstance(seq, int)
        sequences.append(seq)
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == 4
    assert sequences[0] > 1_000_000_000_000
    assert all(item["pane_id"] == "w1:p2" for item in params)
    assert all(item["agent"] == "fast-agent" for item in params)
    assert {item["source"] for item in params} == {"herdr:fast-agent"}


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets are unavailable")
def test_nested_blocked_scope_restores_latest_base_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "herdr.sock"
    requests, server = _capture_requests(socket_path, 4)
    _configure_herdr(monkeypatch, socket_path)

    herdr_lifecycle.report_agent_state("working")
    with herdr_lifecycle.herdr_blocked():
        with herdr_lifecycle.herdr_blocked():
            herdr_lifecycle.report_agent_state("idle")
    _wait_for_requests(requests, 3)
    herdr_lifecycle.release_agent()

    server.join(1)
    assert not server.is_alive()
    states = [
        params.get("state")
        for request in requests
        if request["method"] == "pane.report_agent"
        and isinstance((params := request["params"]), dict)
    ]
    assert states == ["working", "blocked", "idle"]


def test_incomplete_environment_is_a_silent_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.setenv("HERDR_PANE_ID", "w1:p2")
    monkeypatch.delenv("HERDR_SOCKET_PATH", raising=False)

    herdr_lifecycle.report_agent_state("working")
    herdr_lifecycle.release_agent()


def test_release_is_idempotent_and_late_reports_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "herdr.sock"
    requests, server = _capture_requests(socket_path, 2)
    _configure_herdr(monkeypatch, socket_path)

    herdr_lifecycle.report_agent_state("idle")
    _wait_for_requests(requests, 1)
    herdr_lifecycle.release_agent()
    herdr_lifecycle.release_agent()

    herdr_lifecycle.report_agent_state("working")

    server.join(1)
    assert not server.is_alive()
    assert [request["method"] for request in requests] == [
        "pane.report_agent",
        "pane.release_agent",
    ]
    assert herdr_lifecycle._reporter is None


def test_blocked_scope_is_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_to_create_reporter():
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(herdr_lifecycle, "_active_reporter", fail_to_create_reporter)

    entered = False
    with herdr_lifecycle.herdr_blocked():
        entered = True

    assert entered is True


def test_after_fork_child_replaces_inherited_lock_and_disables_reporting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "herdr.sock"
    _configure_herdr(monkeypatch, socket_path)
    inherited_lock = herdr_lifecycle._reporter_lock
    inherited_lock.acquire()
    try:
        herdr_lifecycle._after_fork_child()
        herdr_lifecycle.report_agent_state("working")
    finally:
        inherited_lock.release()

    assert herdr_lifecycle._reporter is None


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets are unavailable")
def test_socket_delivery_retries_same_request_after_missing_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "herdr.sock"
    requests: list[dict[str, object]] = []
    ready = threading.Event()

    def serve() -> None:
        with closing(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)) as server:
            server.bind(str(socket_path))
            server.listen()
            ready.set()
            for attempt in range(2):
                connection, _ = server.accept()
                with connection:
                    payload = b""
                    while not payload.endswith(b"\n"):
                        payload += connection.recv(65536)
                    decoded = json.loads(payload)
                    assert isinstance(decoded, dict)
                    requests.append(decoded)
                    if attempt == 1:
                        response = json.dumps(
                            {"id": decoded["id"], "result": {"type": "ok"}}
                        ).encode()
                        connection.sendall(response + b"\n")

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    assert ready.wait(1)
    _configure_herdr(monkeypatch, socket_path)

    herdr_lifecycle.report_agent_state("working")
    _wait_for_requests(requests, 2)
    herdr_lifecycle.release_agent()

    server.join(1)
    assert not server.is_alive()
    assert requests[0] == requests[1]


def test_windows_transport_uses_injected_herdr_binary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(herdr_lifecycle, "_uses_windows_cli_transport", lambda: True)
    monkeypatch.setattr(herdr_lifecycle.subprocess, "run", run)
    _configure_herdr(monkeypatch, tmp_path / "unused.sock", bin_path=r"C:\Herdr\herdr.exe")
    monkeypatch.delenv("HERDR_SOCKET_PATH")

    herdr_lifecycle.report_agent_state("idle")
    deadline = time.monotonic() + 1
    while len(calls) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(calls) == 1
    herdr_lifecycle.release_agent()

    assert len(calls) == 2
    assert calls[0][:5] == [
        r"C:\Herdr\herdr.exe",
        "pane",
        "report-agent",
        "w1:p2",
        "--source",
    ]
    assert calls[0][5:7] == ["herdr:fast-agent", "--agent"]
    assert calls[0][-2:] == ["--state", "idle"]
    assert calls[1][1:4] == ["pane", "release-agent", "w1:p2"]


def test_delivery_retries_after_transport_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = herdr_lifecycle._HerdrEnvironment(
        pane_id="w1:p2",
        socket_path=str(tmp_path / "unused.sock"),
        bin_path=None,
    )
    reporter = herdr_lifecycle._HerdrLifecycleReporter(environment)
    attempts: list[float] = []

    def send_attempt(_request, timeout_seconds: float) -> bool:
        attempts.append(timeout_seconds)
        if len(attempts) == 1:
            raise OSError("socket unavailable")
        return True

    monkeypatch.setattr(reporter, "_send_request_attempt", send_attempt)
    request = reporter._build_request("pane.report_agent", state="working")

    reporter._send_request(request)
    reporter.release()

    assert attempts[:2] == [0.5, 1.5]
