"""Opt-in end-to-end tests for nested ``hc`` worker sessions against a real isolated herdr-core.

Run: FAST_AGENT_TEST_HC=1 uv run pytest tests/integration/ui/test_hc_nested_sessions.py -v
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pytest

from fast_agent.session.session_manager import SessionManager
from fast_agent.session.subagent_runs import subagent_run_from_session

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("FAST_AGENT_TEST_HC") != "1" or shutil.which("hc") is None,
        reason="opt-in real herdr test (FAST_AGENT_TEST_HC=1 and hc required)",
    ),
]
ROOT = Path(__file__).resolve().parents[3]


def _wait_until(
    description: str, predicate, *, timeout: float = 25.0, interval: float = 0.15
) -> Any:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except Exception as exc:  # pragma: no cover - diagnostic capture
            last_error = exc
        time.sleep(interval)
    raise AssertionError(f"Timed out waiting for {description} (last_error={last_error!r})")


class _IsolatedHcHarness:
    def __init__(self, root: Path, session_name: str) -> None:
        self.root = root
        self.home = root / "fast-agent-home"
        self.home.mkdir()
        (root / "run").mkdir()
        (root / "config").mkdir()
        (root / "state").mkdir()

        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith("HC_"):
                self.env.pop(key)
        for key in (
            "HERDR_ENV",
            "HERDR_SOCKET_PATH",
            "HERDR_PANE_ID",
            "FAST_AGENT_DISABLE_PROGRAM_STATUS",
            "FAST_AGENT_DISABLE_TURN_INPUT",
        ):
            self.env.pop(key, None)
        self.env.update(
            {
                "HC_SESSION": session_name,
                "XDG_CONFIG_HOME": str(root / "config"),
                "XDG_STATE_HOME": str(root / "state"),
                "XDG_RUNTIME_DIR": str(root / "run"),
            }
        )

    def hc(
        self,
        *args: str,
        extra_env: dict[str, str] | None = None,
        check: bool = True,
    ) -> dict[str, Any]:
        env = dict(self.env)
        if extra_env:
            env.update(extra_env)
        result = subprocess.run(
            ["hc", *args],
            env=env,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if check:
            assert result.returncode == 0, f"hc {args} failed: {result.stderr or result.stdout}"
        if not result.stdout.strip():
            return {}
        value = json.loads(result.stdout)
        assert isinstance(value, dict), value
        return value

    def fast_agent_command(self, *, name: str) -> str:
        return " ".join(
            shlex.quote(part)
            for part in (
                sys.executable,
                "-m",
                "fast_agent.cli",
                "go",
                "--model",
                "passthrough",
                "--name",
                name,
                "--home",
                str(self.home),
                "--no-subagents",
                "--quiet",
            )
        )

    def send(self, address: str, text: str) -> None:
        result = subprocess.run(
            ["hc", "send", address, text],
            env=self.env,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, f"hc send failed: {result.stderr or result.stdout}"

    def wait_ready(self, address: str) -> None:
        self.hc("wait", address, "--timeout", "20", "--json")

    def stop(self) -> None:
        subprocess.run(
            ["hc", "server", "stop"],
            env=self.env,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )


@pytest.fixture
def isolated_hc() -> Iterator[_IsolatedHcHarness]:
    with TemporaryDirectory(prefix="fa-hc-nested-") as tmp:
        root = Path(tmp)
        session_name = f"fa-nested-{os.getpid()}-{int(time.time() * 1000) % 100000}"
        harness = _IsolatedHcHarness(root, session_name)
        try:
            harness.hc("server", "start", "--json")
            yield harness
        finally:
            harness.stop()


def test_hc_child_worker_nests_under_parent_session_and_standalone_stays_top_level(
    isolated_hc: _IsolatedHcHarness,
) -> None:
    # 1. Start parent fast-agent in an isolated herdr-core terminal
    parent_run = isolated_hc.hc(
        "run",
        "--bg",
        "--name",
        "parent-agent",
        "--json",
        "--",
        "bash",
        "-lc",
        isolated_hc.fast_agent_command(name="parent-agent"),
    )
    parent_addr = str(parent_run["address"])
    isolated_hc.wait_ready(parent_addr)
    isolated_hc.send(parent_addr, "hello parent turn")

    def _parent_session_saved() -> list[Any]:
        mgr = SessionManager(cwd=ROOT, home_override=isolated_hc.home, respect_env_override=False)
        sessions = mgr.list_sessions()
        return sessions if len(sessions) == 1 and sessions[0].history_files else []

    parent_sessions = _wait_until("parent session persisted", _parent_session_saved)
    parent_session_id = parent_sessions[0].name

    # 2. Spawn a child fast-agent from the parent terminal (so herdr-core sets parent=parent_addr)
    child_run = isolated_hc.hc(
        "run",
        "--bg",
        "--name",
        "child-worker",
        "--json",
        "--",
        "bash",
        "-lc",
        isolated_hc.fast_agent_command(name="child-worker"),
        extra_env={"HC_TERMINAL_ID": parent_addr},
    )
    child_addr = str(child_run["address"])
    child_meta = isolated_hc.hc("get", child_addr, "--json")
    assert child_meta.get("parent") == parent_addr

    isolated_hc.wait_ready(child_addr)
    isolated_hc.send(child_addr, "hello nested worker turn")

    # 3. Verify child session landed under <parent>/children/<child_id>/ and NOT top-level!
    def _nested_child_saved() -> tuple[Any, Any] | None:
        mgr = SessionManager(cwd=ROOT, home_override=isolated_hc.home, respect_env_override=False)
        parent = mgr.get_session(parent_session_id)
        if parent is None:
            return None
        children = mgr.list_child_sessions(parent)
        if len(children) != 1:
            return None
        child = children[0]
        snap = child.load_snapshot_unlocked()
        if not snap.continuation.agents:
            return None
        return (parent, child)

    parent_sess, child_sess = _wait_until("nested hc child persisted", _nested_child_saved)
    inspect_mgr = SessionManager(
        cwd=ROOT, home_override=isolated_hc.home, respect_env_override=False
    )
    assert [s.name for s in inspect_mgr.list_sessions()] == [parent_session_id]

    run = subagent_run_from_session(parent_sess, child_sess)
    assert run is not None
    assert run.tool_name == "hc"
    assert run.hc_terminal_id == child_addr
    assert run.hc_name == "child-worker"
    assert run.alias == "01_child_worker"
    assert run.status == "running"

    # 4. Deleting parent while child is live MUST be refused as busy
    assert inspect_mgr.delete_session_result(parent_session_id).status == "busy"

    # 5. Cleanly exit child via /exit and verify status becomes completed
    isolated_hc.send(child_addr, "EXIT")

    def _child_completed() -> bool:
        mgr = SessionManager(cwd=ROOT, home_override=isolated_hc.home, respect_env_override=False)
        p = mgr.get_session(parent_session_id)
        if p is None:
            return False
        children = mgr.list_child_sessions(p)
        if not children:
            return False
        r = subagent_run_from_session(p, children[0])
        return r is not None and r.status == "completed"

    _wait_until("child status completed after /exit", _child_completed)

    # 6. Standalone fast-agent (no parent terminal) falls back cleanly to a top-level session
    standalone_run = isolated_hc.hc(
        "run",
        "--bg",
        "--name",
        "standalone-agent",
        "--json",
        "--",
        "bash",
        "-lc",
        isolated_hc.fast_agent_command(name="standalone-agent"),
    )
    standalone_addr = str(standalone_run["address"])
    isolated_hc.wait_ready(standalone_addr)
    isolated_hc.send(standalone_addr, "hello standalone turn")

    def _two_top_level_sessions() -> bool:
        mgr = SessionManager(cwd=ROOT, home_override=isolated_hc.home, respect_env_override=False)
        return len(mgr.list_sessions()) == 2

    _wait_until("standalone top-level session created", _two_top_level_sessions)
