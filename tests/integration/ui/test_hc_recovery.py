"""Opt-in end-to-end tests for hc reboot/restart recovery against a real isolated herdr-core.

Run: FAST_AGENT_TEST_HC=1 uv run pytest tests/integration/ui/test_hc_recovery.py -v

Each test owns a dedicated scratch HC_SESSION and scratch XDG_* / fast-agent home
directories. It never attaches to, stops, or restarts the user's live core.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pytest

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
    description: str, predicate, *, timeout: float = 20.0, interval: float = 0.1
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


class IsolatedHcRecoveryHarness:
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

    def hc(self, *args: str, check: bool = True) -> dict[str, Any]:
        result = subprocess.run(
            ["hc", *args],
            env=self.env,
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

    def ls_terminals(self) -> list[dict[str, Any]]:
        payload = self.hc("ls", "--all", "--json")
        terminals = payload.get("terminals")
        assert isinstance(terminals, list), payload
        return terminals

    def fast_agent_command(self) -> str:
        return " ".join(
            shlex.quote(part)
            for part in (
                sys.executable,
                "-m",
                "fast_agent.cli",
                "go",
                "--model",
                "passthrough",
                "--home",
                str(self.home),
                "--no-subagents",
                "--quiet",
            )
        )

    def send_keys(self, address: str, text: str) -> None:
        # Use `hc send --raw` (or plain `hc send` when terminal is in raw/shell mode)
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

    def read_tail(self, address: str, lines: int = 80) -> str:
        payload = self.hc("read", address, "--lines", str(lines), "--json")
        return str(payload.get("text") or payload.get("output") or payload)

    def simulate_reboot(self) -> None:
        status = self.hc("server", "status", "--json")
        core_pid = status.get("pid")
        holder = status.get("holder")
        holder_pid = holder.get("pid") if isinstance(holder, dict) else None
        assert isinstance(core_pid, int), status
        assert isinstance(holder_pid, int), status
        for pid in (core_pid, holder_pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        # Wait until the core auto-restarts on next hc query.
        def _restarted() -> bool:
            st = self.hc("server", "status", "--json", check=False)
            new_pid = st.get("pid")
            return isinstance(new_pid, int) and new_pid != core_pid

        _wait_until("isolated core restart", _restarted, timeout=20.0)

    def load_session_histories(self) -> dict[str, list[dict[str, Any]]]:
        sessions_dir = self.home / "sessions"
        result: dict[str, list[dict[str, Any]]] = {}
        if not sessions_dir.exists():
            return result
        for session_dir in sorted(sessions_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            histories = sorted(
                p
                for p in session_dir.glob("history_*.json")
                if not p.name.endswith("_previous.json")
            )
            if not histories:
                continue
            data = json.loads(histories[0].read_text(encoding="utf-8"))
            messages = data.get("messages") if isinstance(data, dict) else []
            if isinstance(messages, list):
                result[session_dir.name] = messages
        return result

    def setup_interrupted_session_with_stale_child(self) -> tuple[str, str, str]:
        """Start a --keep shell terminal, run fast-agent for 1 turn, spawn a child, then kill core+holder.

        Returns (parent_address, stale_child_address, original_session_id).
        """
        created = self.hc(
            "run",
            "--bg",
            "--keep",
            "--no-parent",
            "--no-notify",
            "--name",
            "host-tab",
            "--cwd",
            str(ROOT),
            "--",
            "/bin/bash",
            "--norc",
            "--noprofile",
        )
        parent = str(created["address"])

        # Launch fast-agent inside the --keep shell tab.
        self.send_keys(parent, self.fast_agent_command())
        self.hc("wait", parent)

        # Complete one turn so the session is persisted on disk and bound to (core_id, parent).
        self.send_keys(parent, "marker-turn-1")
        _wait_until(
            "marker-turn-1 persisted",
            lambda: any(
                "marker-turn-1" in json.dumps(msgs)
                for msgs in self.load_session_histories().values()
            ),
            timeout=20.0,
        )
        self.hc("wait", parent)
        histories = self.load_session_histories()
        assert len(histories) == 1, histories
        original_session_id = next(iter(histories))

        # Spawn a child under parent so reboot leaves it finished(stopped_by_restart).
        child_rec = self.hc(
            "run",
            "--bg",
            "--parent",
            parent,
            "--name",
            "stale-worker",
            "--",
            "sleep",
            "120",
        )
        stale_child = str(child_rec["address"])

        # Simulate machine reboot (kill scratch core + holder).
        self.simulate_reboot()

        def _after_reboot() -> bool:
            by_addr = {t["address"]: t for t in self.ls_terminals()}
            p = by_addr.get(parent)
            c = by_addr.get(stale_child)
            if not (
                p is not None
                and p.get("state") == "live"
                and not p.get("lost")
                and c is not None
                and c.get("state") == "finished"
                and c.get("reason") == "stopped_by_restart"
            ):
                return False
            probe = subprocess.run(
                ["hc", "send", parent, "true"],
                env=self.env,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return probe.returncode == 0

        _wait_until("keep parent restarted & child stopped_by_restart", _after_reboot, timeout=20.0)
        return parent, stale_child, original_session_id

    def close(self) -> None:
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
def hc_harness(request: pytest.FixtureRequest) -> Iterator[IsolatedHcRecoveryHarness]:
    with TemporaryDirectory(prefix="fa-hc-rec-") as tmp:
        session_name = f"rec-{abs(hash(request.node.name)) % 100000}"
        harness = IsolatedHcRecoveryHarness(Path(tmp), session_name)
        try:
            yield harness
        finally:
            harness.close()


def test_real_hc_reboot_fresh_and_clear_suppresses_stale_notices_and_preserves_live_child(
    hc_harness: IsolatedHcRecoveryHarness,
) -> None:
    parent, stale_child, original_session_id = (
        hc_harness.setup_interrupted_session_with_stale_child()
    )

    # Also spawn a NEW live child under parent to verify [c] only removes stopped_by_restart.
    live_rec = hc_harness.hc(
        "run",
        "--bg",
        "--parent",
        parent,
        "--name",
        "fresh-live",
        "--",
        "sleep",
        "120",
    )
    live_child = str(live_rec["address"])

    # Relaunch fast-agent in the same --keep terminal.
    hc_harness.send_keys(parent, hc_harness.fast_agent_command())

    # Wait for the recovery prompt to appear; verify NO <hc> notice is injected while held.
    _wait_until(
        "recovery prompt visible",
        lambda: "Choice [" in hc_harness.read_tail(parent),
        timeout=20.0,
    )
    time.sleep(0.5)
    prompt_screen = hc_harness.read_tail(parent)
    assert "stopped_by_restart" not in prompt_screen or "stale-worker" in prompt_screen
    assert "<hc>" not in prompt_screen, f"hc notice leaked during startup hold: {prompt_screen}"

    # Choose [c] Fresh session & clear stopped_by_restart notices.
    hc_harness.send_keys(parent, "c")
    hc_harness.hc("wait", parent)

    # Verify stale_child was removed while live_child remains live.
    by_addr = {t["address"]: t for t in hc_harness.ls_terminals()}
    assert stale_child not in by_addr, f"stale child should have been removed: {by_addr}"
    assert live_child in by_addr and by_addr[live_child]["state"] == "live"

    # Send a turn in the fresh session and verify ZERO stale notices arrived.
    hc_harness.send_keys(parent, "fresh-turn-after-clear")
    _wait_until(
        "fresh turn persisted",
        lambda: any(
            sid != original_session_id and "fresh-turn-after-clear" in json.dumps(msgs)
            for sid, msgs in hc_harness.load_session_histories().items()
        ),
        timeout=20.0,
    )
    histories = hc_harness.load_session_histories()
    fresh_histories = {sid: msgs for sid, msgs in histories.items() if sid != original_session_id}
    assert len(fresh_histories) == 1, fresh_histories
    fresh_dump = json.dumps(next(iter(fresh_histories.values())))
    assert "<hc>" not in fresh_dump, f"stale notice leaked into fresh session: {fresh_dump}"
    assert "stopped_by_restart" not in fresh_dump


def test_real_hc_reboot_resume_restores_session_and_delivers_notices(
    hc_harness: IsolatedHcRecoveryHarness,
) -> None:
    parent, stale_child, original_session_id = (
        hc_harness.setup_interrupted_session_with_stale_child()
    )

    hc_harness.send_keys(parent, hc_harness.fast_agent_command())
    _wait_until(
        "recovery prompt visible",
        lambda: "Choice [" in hc_harness.read_tail(parent),
        timeout=20.0,
    )
    time.sleep(0.5)
    assert "<hc>" not in hc_harness.read_tail(parent)

    # Choose [r] Resume previous session.
    hc_harness.send_keys(parent, "r")

    # Wait for the resumed session to receive the real <hc>...stopped_by_restart notice.
    _wait_until(
        "stopped_by_restart notice delivered to resumed session",
        lambda: (
            "stopped_by_restart"
            in json.dumps(hc_harness.load_session_histories().get(original_session_id, []))
        ),
        timeout=25.0,
    )
    histories = hc_harness.load_session_histories()
    assert list(histories.keys()) == [original_session_id]
    resumed_dump = json.dumps(histories[original_session_id])
    assert "marker-turn-1" in resumed_dump
    assert stale_child in resumed_dump
    assert "stopped_by_restart" in resumed_dump


def test_real_hc_reboot_fresh_as_is_keeps_children_and_delivers_notices(
    hc_harness: IsolatedHcRecoveryHarness,
) -> None:
    parent, stale_child, original_session_id = (
        hc_harness.setup_interrupted_session_with_stale_child()
    )

    hc_harness.send_keys(parent, hc_harness.fast_agent_command())
    _wait_until(
        "recovery prompt visible",
        lambda: "Choice [" in hc_harness.read_tail(parent),
        timeout=20.0,
    )

    # Choose [n] Fresh session as-is.
    hc_harness.send_keys(parent, "n")

    # Wait for the fresh session to receive the real <hc>...stopped_by_restart notice.
    _wait_until(
        "stopped_by_restart notice delivered to fresh session",
        lambda: any(
            sid != original_session_id and "stopped_by_restart" in json.dumps(msgs)
            for sid, msgs in hc_harness.load_session_histories().items()
        ),
        timeout=25.0,
    )
    by_addr = {t["address"]: t for t in hc_harness.ls_terminals()}
    assert stale_child in by_addr
