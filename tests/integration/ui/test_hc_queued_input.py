"""Opt-in tests with real hc terminals; no API keys or model inference required.

Run: FAST_AGENT_TEST_HC=1 uv run pytest tests/integration/ui/test_hc_queued_input.py -v
Each test owns a separate core/session and scratch XDG paths. Never attaches to,
stops, or restarts the user's core. A socket gates batch/late-arrival boundaries,
not sleeps; the natural-delivery case has no gate after hc wait.

Input-contract tests use quiet mode. A separate normal-progress test records the
known hc 0.1.17 startup-readiness failure only when its exact rejection is observed,
then verifies that a subsequent send succeeds with progress still enabled.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
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


class HerdrScenario:
    def __init__(self, root: Path) -> None:
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
                "HC_SESSION": "input-test",
                "XDG_CONFIG_HOME": str(root / "config"),
                "XDG_STATE_HOME": str(root / "state"),
                "XDG_RUNTIME_DIR": str(root / "run"),
            }
        )
        (root / "run").mkdir()
        self.terminals: list[str] = []
        self.audit: list[dict[str, Any]] = []
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.listener.settimeout(30)
        self.connection: socket.socket | None = None

    def hc(self, *args: str) -> dict[str, Any]:
        # CLI JSON is an external boundary; validate its container here.
        result = subprocess.run(
            ["hc", *args],
            env=self.env,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        value = json.loads(result.stdout)
        assert isinstance(value, dict), value
        return value

    def start(self, *, quiet: bool = True) -> str:
        result = self.hc(
            "run",
            "--bg",
            "--no-parent",
            "--no-notify",
            "--name",
            "coordinator",
            "--cwd",
            str(ROOT),
            "--",
            sys.executable,
            str(Path(__file__).with_name("hc_input_agent.py")),
            str(self.listener.getsockname()[1]),
            "quiet" if quiet else "progress",
        )
        address = result["address"]
        self.terminals.append(address)
        self.connection, _ = self.listener.accept()
        self.connection.settimeout(30)
        self.events = self.connection.makefile("r")
        assert self.receive("connected")
        self.hc("wait", address)
        self.hc("send", address, "warmup")
        self.receive("next_turn")
        self.hc("wait", address)
        return address

    def receive(self, kind: str) -> dict[str, Any]:
        event = json.loads(self.events.readline())
        assert isinstance(event, dict)
        self.audit.append(event)
        assert event["kind"] == kind, event
        return event

    def release(self) -> None:
        assert self.connection is not None
        self.connection.sendall(b"continue\n")

    def reports(self, parent: str, *messages: str) -> None:
        # A real child terminal sends real hc messages, including hc's envelope.
        result = self.hc(
            "run",
            "--bg",
            "--no-notify",
            "--parent",
            parent,
            "--",
            sys.executable,
            "-c",
            "import subprocess,sys; "
            "print('\\x1b]7501;state=working:app=test-worker\\x1b\\\\',end='',flush=True); "
            "[subprocess.run(['hc','send',sys.argv[1],m],check=True) for m in sys.argv[2:]]",
            parent,
            *messages,
        )
        address = result["address"]
        self.terminals.append(address)
        result = self.hc("wait", address)
        assert result["exit_code"] == 0, result

    def close(self) -> None:
        for address in reversed(self.terminals):
            subprocess.run(
                ["hc", "kill", address], env=self.env, capture_output=True, timeout=10, check=False
            )
            subprocess.run(
                ["hc", "rm", address], env=self.env, capture_output=True, timeout=10, check=False
            )
        subprocess.run(
            ["hc", "server", "stop"], env=self.env, capture_output=True, timeout=10, check=False
        )
        if self.connection is not None:
            self.events.close()
            self.connection.close()
        self.listener.close()


@pytest.fixture
def scenario(tmp_path: Path) -> Iterator[HerdrScenario]:
    # Unix-domain socket paths have a small fixed limit; pytest node paths
    # can exceed it even when hc itself uses short filenames.
    with TemporaryDirectory(prefix="fa-hc-") as directory:
        scenario = HerdrScenario(Path(directory))
        try:
            yield scenario
        finally:
            (tmp_path / "events.json").write_text(json.dumps(scenario.audit, indent=2))
            for address in scenario.terminals:
                result = subprocess.run(
                    ["hc", "read", address, "--lines", "150"],
                    env=scenario.env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                (tmp_path / f"{address}.txt").write_text(result.stdout + result.stderr)
            scenario.close()


def test_reports_are_seen_before_final_response(scenario: HerdrScenario) -> None:
    agent = scenario.start()
    scenario.hc("send", agent, "scenario batch")
    scenario.receive("initial")
    scenario.receive("tool_waiting")
    scenario.audit.append({"kind": "working_status", **scenario.hc("get", agent)})
    scenario.reports(agent, "REPORT_ONE: checked configuration", "REPORT_TWO: tests pass")
    event = scenario.receive("wait_returned")
    assert event["result"]["reason"] == "message"
    scenario.release()
    messages = scenario.receive("followup")["messages"]
    assert messages[0]["tools"] == ["wait"]
    assert len(messages) == 3
    assert "REPORT_ONE" in messages[1]["text"]
    assert "REPORT_TWO" in messages[2]["text"]
    scenario.hc("wait", agent)
    # A new input must be next, not a duplicate replay of either report.
    scenario.hc("send", agent, "probe")
    messages = scenario.receive("next_turn")["messages"]
    assert [message["text"] for message in messages] == ["probe"]


def test_commands_are_not_injected_or_overtaken(scenario: HerdrScenario) -> None:
    agent = scenario.start()
    scenario.hc("send", agent, "scenario command")
    scenario.receive("initial")
    scenario.receive("tool_waiting")
    scenario.hc("send", agent, "/help")
    scenario.reports(agent, "REPORT_AFTER_COMMAND")
    scenario.receive("wait_returned")
    scenario.release()
    messages = scenario.receive("followup")["messages"]
    assert len(messages) == 1 and messages[0]["tools"] == ["wait"]
    messages = scenario.receive("next_turn")["messages"]
    assert len(messages) == 1 and "REPORT_AFTER_COMMAND" in messages[0]["text"]
    assert "/help" not in messages[0]["text"]


def test_late_report_becomes_next_turn_once(scenario: HerdrScenario) -> None:
    agent = scenario.start()
    scenario.hc("send", agent, "scenario late")
    scenario.receive("initial")
    scenario.receive("tool_waiting")
    scenario.reports(agent, "EARLY_REPORT")
    scenario.receive("wait_returned")
    scenario.release()
    messages = scenario.receive("followup")["messages"]
    assert "EARLY_REPORT" in messages[-1]["text"]
    scenario.reports(agent, "LATE_REPORT")
    scenario.release()
    messages = scenario.receive("next_turn")["messages"]
    assert len(messages) == 1 and "LATE_REPORT" in messages[0]["text"]
    scenario.hc("wait", agent)
    scenario.hc("send", agent, "probe")
    messages = scenario.receive("next_turn")["messages"]
    assert [message["text"] for message in messages] == ["probe"]


@pytest.mark.parametrize("_attempt", range(3))
def test_wait_delivery_without_a_test_gate(scenario: HerdrScenario, _attempt: int) -> None:
    agent = scenario.start()
    scenario.hc("send", agent, "scenario natural")
    scenario.receive("initial")
    scenario.receive("tool_waiting")
    scenario.reports(agent, "NATURAL_REPORT")
    assert scenario.receive("wait_returned")["result"]["reason"] == "message"
    messages = scenario.receive("followup")["messages"]
    assert messages[0]["tools"] == ["wait"]
    # The provider boundary may combine a single continuation with tool results.
    assert sum(message["text"].count("NATURAL_REPORT") for message in messages) == 1
    scenario.hc("wait", agent)
    scenario.hc("send", agent, "probe")
    assert scenario.receive("next_turn")["messages"][-1]["text"] == "probe"


def test_send_with_progress_rendering(scenario: HerdrScenario) -> None:
    agent = scenario.start(quiet=False)
    scenario.hc("send", agent, "scenario progress")
    scenario.receive("initial")
    scenario.receive("tool_waiting")
    result = subprocess.run(
        ["hc", "send", agent, "PROGRESS_REPORT"],
        env=scenario.env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    startup_rejection = result.returncode == 5 and "output has not settled" in result.stderr
    if startup_rejection:
        assert "nothing was typed" in result.stderr
        assert scenario.hc("get", agent)["status"]["state"] == "working"
        # Read before retrying, as hc's manual requires. The failed startup
        # check happens before any paste; this is not a duplicate delivery.
        screen = subprocess.run(
            ["hc", "read", agent, "--lines", "0"],
            env=scenario.env,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        assert "PROGRESS_REPORT" not in screen.stdout
        # settle() waits at most five seconds, and stops applying at terminal
        # age ten seconds. A retry now crosses that age without any sleep.
        scenario.hc("send", agent, "PROGRESS_REPORT")
    else:
        assert result.returncode == 0, result.stderr
    scenario.receive("wait_returned")
    scenario.release()
    messages = scenario.receive("followup")["messages"]
    assert "PROGRESS_REPORT" in messages[-1]["text"]
    if startup_rejection:
        pytest.xfail(
            "hc 0.1.17's first-10-seconds startup gate rejected an already-working agent; "
            "retry succeeded with progress still enabled"
        )


def test_interrupt_preserves_queued_report(scenario: HerdrScenario) -> None:
    agent = scenario.start()
    scenario.hc("send", agent, "scenario cancel")
    scenario.receive("initial")
    scenario.receive("tool_waiting")
    scenario.reports(agent, "REPORT_BEFORE_CANCEL")
    scenario.receive("wait_returned")
    result = scenario.hc("interrupt", agent)
    assert result["outcome"] != "exited"
    messages = scenario.receive("next_turn")["messages"]
    assert len(messages) == 1 and "REPORT_BEFORE_CANCEL" in messages[0]["text"]
    scenario.hc("wait", agent)
    scenario.hc("send", agent, "probe")
    assert scenario.receive("next_turn")["messages"][-1]["text"] == "probe"


def test_read_retains_history_without_interrupting_the_agent(scenario: HerdrScenario) -> None:
    agent = scenario.start(quiet=False)
    scenario.hc("send", agent, "scenario read")
    scenario.receive("initial")
    scenario.receive("tool_waiting")

    screen = scenario.hc("read", "--json", agent, "--lines", "0")
    history = scenario.hc("read", "--json", agent, "--lines", "300")
    scenario.audit.extend(
        [
            {"kind": "working_screen", **screen},
            {"kind": "working_history", **history},
        ]
    )
    assert screen["how"] == "screen"
    assert "READ_HISTORY_000" not in screen["text"]
    assert "READ_HISTORY_179" in screen["text"]
    assert history["how"] == "history"  # normal scrollback, not scrolling the application
    for index in range(180):
        assert history["text"].count(f"READ_HISTORY_{index:03d}") == 1
    assert scenario.hc("get", agent)["status"]["state"] == "working"

    scenario.release()
    scenario.receive("followup")
    scenario.hc("wait", agent)
    completed = scenario.hc("read", "--json", agent, "--lines", "300")
    scenario.audit.append({"kind": "completed_history", **completed})
    assert completed["how"] == "history"
    assert "READ_FINAL_COMPLETE" in completed["text"]
    for index in range(180):
        assert completed["text"].count(f"READ_HISTORY_{index:03d}") == 1
    assert scenario.hc("get", agent)["status"]["state"] == "idle"


@pytest.mark.parametrize(
    ("shell_flag", "ptc"), [("--shell", False), ("--shell", True), ("--no-shell", False)]
)
def test_cli_system_includes_herdr_guidance_in_fresh_and_new_session(
    scenario: HerdrScenario, tmp_path: Path, shell_flag: str, ptc: bool
) -> None:
    home = tmp_path / "home"
    scenario.env["FAST_AGENT_HOME"] = str(home)
    scenario.env["FAST_AGENT_RUNTIME_HOME"] = str(home)
    config = tmp_path / "fast-agent.yaml"
    config.write_text(f"default_model: passthrough\nptc:\n  enabled: {str(ptc).lower()}\n")
    result = scenario.hc(
        "run",
        "--bg",
        "--no-parent",
        "--no-notify",
        "--cwd",
        str(tmp_path),
        "--",
        sys.executable,
        "-m",
        "fast_agent.cli",
        "--no-update-check",
        "--home",
        str(home),
        "go",
        "--config-path",
        str(config),
        "--model",
        "passthrough",
        shell_flag,
    )
    address = result["address"]
    scenario.terminals.append(address)
    scenario.hc("wait", address)
    for command in (None, "/session new"):
        if command:
            scenario.hc("send", address, command)
        scenario.hc("send", address, "/system")
        screen = scenario.hc("read", address, "--json", "--lines", "300")["text"]
        current_system = screen.rsplit("❯ /system", 1)[-1]
        assert ("hc --skill" in current_system) is (shell_flag == "--shell"), screen
        if command:
            assert "Created session:" in screen, screen
