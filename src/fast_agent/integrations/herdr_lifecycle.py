"""Native Herdr lifecycle reporting for interactive fast-agent sessions."""

from __future__ import annotations

import atexit
import json
import os
import queue
import socket
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

type HerdrAgentState = Literal["idle", "working", "blocked", "unknown"]
type HerdrBaseState = Literal["idle", "working", "unknown"]
type _HerdrMethod = Literal[
    "pane.report_agent",
    "pane.report_metadata",
    "pane.release_agent",
]
type _SessionMetadata = tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    bool,
    str | None,
    str | None,
    str | None,
    str | None,
]

_AGENT_LABEL = "fast-agent"
_SOURCE = "herdr:fast-agent"
_DELIVERY_TIMEOUT_SECONDS = (0.5, 1.5)
# A release may wait behind one already in-flight state request.
_RELEASE_TIMEOUT_SECONDS = (sum(_DELIVERY_TIMEOUT_SECONDS) * 2) + 0.25
_MAX_RESPONSE_BYTES = 64 * 1024
_WINDOWS_CREATE_NO_WINDOW = 0x08000000


@dataclass(frozen=True, slots=True)
class _HerdrEnvironment:
    pane_id: str
    socket_path: str | None
    bin_path: str | None


@dataclass(frozen=True, slots=True)
class _HerdrRequest:
    request_id: str
    method: _HerdrMethod
    params: dict[str, object]
    completion: threading.Event | None = None
    stop_after: bool = False
    session_metadata: _SessionMetadata | None = None


def _herdr_environment() -> _HerdrEnvironment | None:
    if os.environ.get("HERDR_ENV") != "1":
        return None

    socket_path = os.environ.get("HERDR_SOCKET_PATH", "").strip()
    bin_path = os.environ.get("HERDR_BIN_PATH", "").strip()
    pane_id = os.environ.get("HERDR_PANE_ID", "").strip()
    if not pane_id:
        return None
    if _uses_windows_cli_transport():
        if not bin_path:
            return None
    elif not socket_path or not hasattr(socket, "AF_UNIX"):
        return None
    return _HerdrEnvironment(
        pane_id=pane_id,
        socket_path=socket_path or None,
        bin_path=bin_path or None,
    )


def _uses_windows_cli_transport() -> bool:
    return os.name == "nt"


class _HerdrLifecycleReporter:
    def __init__(self, environment: _HerdrEnvironment) -> None:
        self._environment = environment
        self._creator_pid = os.getpid()
        self._request_nonce = uuid.uuid4().hex[:12]
        self._requests: queue.Queue[_HerdrRequest] = queue.Queue()
        self._lock = threading.Lock()
        self._sequence = time.time_ns()
        self._base_state: HerdrBaseState = "unknown"
        self._blocked_depth = 0
        self._session_metadata: _SessionMetadata | None = None
        self._pending_session_metadata: _SessionMetadata | None = None
        self._latest_metadata_request_id: str | None = None
        self._closing = False
        self._worker = threading.Thread(
            target=self._run_worker,
            name="fast-agent-herdr",
            daemon=True,
        )
        self._worker.start()

    @property
    def creator_pid(self) -> int:
        return self._creator_pid

    def report_state(self, state: HerdrBaseState) -> None:
        if state not in ("idle", "working", "unknown"):
            return
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing:
                return
            self._base_state = state
            if self._blocked_depth == 0:
                self._enqueue_locked("pane.report_agent", state=state)

    def enter_blocked(self) -> None:
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing:
                return
            self._enter_blocked_locked()

    def exit_blocked(self) -> None:
        if os.getpid() != self._creator_pid:
            return
        with self._lock:
            if self._closing or self._blocked_depth == 0:
                return
            self._blocked_depth -= 1
            if self._blocked_depth == 0:
                self._enqueue_locked("pane.report_agent", state=self._base_state)

    def report_session_metadata(
        self,
        *,
        session_id: str | None,
        title: str | None,
        model: str | None,
        agent_name: str | None,
        pinned: bool,
        forked_from: str | None,
        context_usage: str | None,
        token_usage: str | None,
        prompt: str | None = None,
    ) -> None:
        if os.getpid() != self._creator_pid:
            return
        metadata = (
            session_id,
            title,
            model,
            agent_name,
            pinned,
            forked_from,
            context_usage,
            token_usage,
            prompt,
        )
        with self._lock:
            if self._closing or metadata == self._pending_session_metadata:
                return
            if self._pending_session_metadata is None and metadata == self._session_metadata:
                return
            request = self._build_request(
                "pane.report_metadata",
                params={
                    "applies_to_source": _SOURCE,
                    "title": title,
                    "display_agent": title,
                    "tokens": {
                        "session": session_id,
                        "model": model,
                        "agent_name": agent_name,
                        "pinned": "pinned" if pinned else None,
                        "forked_from": forked_from,
                        "context": context_usage,
                        "tokens": token_usage,
                        "prompt": prompt,
                    },
                    "clear_title": title is None,
                    "clear_display_agent": title is None,
                },
                session_metadata=metadata,
            )
            self._pending_session_metadata = metadata
            self._latest_metadata_request_id = request.request_id
            self._requests.put(request)

    def report_session_usage(self, usage: str) -> None:
        if os.getpid() != self._creator_pid or not usage:
            return
        with self._lock:
            if self._closing:
                return
            self._enqueue_locked(
                "pane.report_metadata",
                params={
                    "applies_to_source": _SOURCE,
                    "tokens": {"tokens": usage},
                },
            )

    def release(self) -> None:
        if os.getpid() != self._creator_pid:
            return

        completion = threading.Event()
        with self._lock:
            if self._closing:
                return
            self._closing = True
            self._discard_queued_requests_locked()
            request = self._build_request(
                "pane.release_agent",
                completion=completion,
                stop_after=True,
            )
            self._requests.put(request)
        completion.wait(_RELEASE_TIMEOUT_SECONDS)

    def _enter_blocked_locked(self) -> None:
        self._blocked_depth += 1
        if self._blocked_depth == 1:
            self._enqueue_locked("pane.report_agent", state="blocked")

    def _enqueue_locked(
        self,
        method: _HerdrMethod,
        *,
        state: HerdrAgentState | None = None,
        params: dict[str, object] | None = None,
    ) -> None:
        self._requests.put(self._build_request(method, state=state, params=params))

    def _build_request(
        self,
        method: _HerdrMethod,
        *,
        state: HerdrAgentState | None = None,
        params: dict[str, object] | None = None,
        completion: threading.Event | None = None,
        stop_after: bool = False,
        session_metadata: _SessionMetadata | None = None,
    ) -> _HerdrRequest:
        self._sequence += 1
        request_params: dict[str, object] = {
            "pane_id": self._environment.pane_id,
            "source": _SOURCE,
            "agent": _AGENT_LABEL,
            "seq": self._sequence,
        }
        if state is not None:
            request_params["state"] = state
        if params is not None:
            request_params.update(params)
        return _HerdrRequest(
            request_id=f"{_SOURCE}:{self._request_nonce}:{self._sequence}",
            method=method,
            params=request_params,
            completion=completion,
            stop_after=stop_after,
            session_metadata=session_metadata,
        )

    def _discard_queued_requests_locked(self) -> None:
        while True:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                return

    def _run_worker(self) -> None:
        while True:
            request = self._requests.get()
            delivered = False
            try:
                delivered = self._send_request(request)
            except Exception:
                pass
            finally:
                if request.session_metadata is not None:
                    self._complete_metadata_request(request, delivered=delivered)
                if request.completion is not None:
                    request.completion.set()
            if request.stop_after:
                return

    def _complete_metadata_request(
        self,
        request: _HerdrRequest,
        *,
        delivered: bool,
    ) -> None:
        with self._lock:
            if delivered:
                self._session_metadata = request.session_metadata
            if self._latest_metadata_request_id == request.request_id:
                self._pending_session_metadata = None
                self._latest_metadata_request_id = None

    def _send_request(self, request: _HerdrRequest) -> bool:
        for timeout_seconds in _DELIVERY_TIMEOUT_SECONDS:
            try:
                delivered = self._send_request_attempt(request, timeout_seconds)
            except Exception:
                delivered = False
            if delivered:
                return True
        return False

    def _send_request_attempt(
        self,
        request: _HerdrRequest,
        timeout_seconds: float,
    ) -> bool:
        if _uses_windows_cli_transport():
            return self._send_request_with_cli(request, timeout_seconds)
        return self._send_request_with_socket(request, timeout_seconds)

    def _send_request_with_socket(
        self,
        request: _HerdrRequest,
        timeout_seconds: float,
    ) -> bool:
        socket_path = self._environment.socket_path
        if socket_path is None:
            return False
        payload = json.dumps(
            {
                "id": request.request_id,
                "method": request.method,
                "params": request.params,
            },
            separators=(",", ":"),
        )
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout_seconds)
            client.connect(socket_path)
            client.sendall(f"{payload}\n".encode())
            response = self._read_response_line(client)
        try:
            decoded = json.loads(response)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        return (
            isinstance(decoded, dict)
            and decoded.get("id") == request.request_id
            and "result" in decoded
            and "error" not in decoded
        )

    @staticmethod
    def _read_response_line(client: socket.socket) -> bytes:
        response = bytearray()
        while len(response) <= _MAX_RESPONSE_BYTES:
            chunk = client.recv(min(4096, _MAX_RESPONSE_BYTES + 1 - len(response)))
            if not chunk:
                break
            response.extend(chunk)
            newline = response.find(b"\n")
            if newline >= 0:
                return bytes(response[:newline])
        return b""

    def _send_request_with_cli(
        self,
        request: _HerdrRequest,
        timeout_seconds: float,
    ) -> bool:
        command = self._cli_command(request)
        if command is None:
            return False
        result = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            creationflags=_WINDOWS_CREATE_NO_WINDOW,
        )
        return result.returncode == 0

    def _cli_command(self, request: _HerdrRequest) -> list[str] | None:
        bin_path = self._environment.bin_path
        seq = request.params.get("seq")
        if bin_path is None or not isinstance(seq, int):
            return None
        subcommand = {
            "pane.release_agent": "release-agent",
            "pane.report_agent": "report-agent",
            "pane.report_metadata": "report-metadata",
        }[request.method]
        command = [
            bin_path,
            "pane",
            subcommand,
            self._environment.pane_id,
            "--source",
            _SOURCE,
        ]
        if request.method != "pane.report_metadata":
            command.extend(["--agent", _AGENT_LABEL])
        command.extend(["--seq", str(seq)])
        if request.method == "pane.report_agent":
            state = request.params.get("state")
            if not isinstance(state, str):
                return None
            command.extend(["--state", state])
        elif request.method == "pane.report_metadata":
            command.extend(["--agent", _AGENT_LABEL, "--applies-to-source", _SOURCE])
            title = request.params.get("title")
            if isinstance(title, str):
                command.extend(["--title", title, "--display-agent", title])
            else:
                if request.params.get("clear_title") is True:
                    command.append("--clear-title")
                if request.params.get("clear_display_agent") is True:
                    command.append("--clear-display-agent")
            tokens = request.params.get("tokens")
            if isinstance(tokens, dict):
                for name, value in tokens.items():
                    if isinstance(value, str):
                        command.extend(["--token", f"{name}={value}"])
                    elif value is None:
                        command.extend(["--clear-token", str(name)])
        return command


_reporter_lock = threading.Lock()
_reporter: _HerdrLifecycleReporter | None = None
_reporting_closed = False


def _after_fork_child() -> None:
    global _reporter_lock, _reporter, _reporting_closed

    _reporter_lock = threading.Lock()
    _reporter = None
    _reporting_closed = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def _active_reporter() -> _HerdrLifecycleReporter | None:
    global _reporter

    environment = _herdr_environment()
    if environment is None:
        return None

    current_pid = os.getpid()
    with _reporter_lock:
        if _reporting_closed:
            return None
        if _reporter is None or _reporter.creator_pid != current_pid:
            _reporter = _HerdrLifecycleReporter(environment)
        return _reporter


def report_agent_state(state: HerdrBaseState) -> None:
    """Report the pane-level lifecycle state when running inside Herdr."""
    if state not in ("idle", "working", "unknown"):
        return
    try:
        reporter = _active_reporter()
        if reporter is not None:
            reporter.report_state(state)
    except Exception:
        pass


def report_session_metadata(
    *,
    session_id: str | None,
    title: str | None,
    model: str | None,
    agent_name: str | None,
    pinned: bool,
    forked_from: str | None,
    context_usage: str | None = None,
    token_usage: str | None = None,
    prompt: str | None = None,
) -> None:
    """Report display-only persisted session metadata when running inside Herdr."""
    try:
        reporter = _active_reporter()
        if reporter is not None:
            reporter.report_session_metadata(
                session_id=session_id,
                title=title,
                model=model,
                agent_name=agent_name,
                pinned=pinned,
                forked_from=forked_from,
                context_usage=context_usage,
                token_usage=token_usage,
                prompt=prompt,
            )
    except Exception:
        pass


def report_session_usage(usage: str) -> None:
    """Report a plugin-projected session usage value when running inside Herdr."""
    try:
        reporter = _active_reporter()
        if reporter is not None:
            reporter.report_session_usage(usage)
    except Exception:
        pass


def report_prompt_mark(code: str) -> None:
    """Map OSC 133 semantic prompt marks to Herdr lifecycle states."""
    command = code.partition(";")[0]
    if command == "A":
        report_agent_state("idle")
    elif command == "C":
        report_agent_state("working")
    elif command == "D":
        report_agent_state("idle")


@contextmanager
def herdr_blocked():
    """Report blocked while fast-agent is waiting for interactive human input."""
    reporter: _HerdrLifecycleReporter | None = None
    try:
        try:
            reporter = _active_reporter()
            if reporter is not None:
                reporter.enter_blocked()
        except Exception:
            reporter = None
        yield
    finally:
        if reporter is not None:
            try:
                reporter.exit_blocked()
            except Exception:
                pass


def release_agent() -> None:
    """Release lifecycle authority without creating a reporter solely for shutdown."""
    global _reporter, _reporting_closed

    try:
        with _reporter_lock:
            _reporting_closed = True
            reporter = _reporter
            _reporter = None
        if reporter is not None:
            reporter.release()
    except Exception:
        pass


def _reset_for_tests() -> None:
    global _reporting_closed

    release_agent()
    with _reporter_lock:
        _reporting_closed = False


atexit.register(release_agent)
