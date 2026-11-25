"""Unit tests for hc terminal recovery, passive hc inspection, and startup hold."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Literal

import pytest

from fast_agent.cli.runtime.hc_recovery import (
    HcRecoveryOutcome,
    HcRecoveryPlan,
    resolve_hc_startup_recovery,
    should_check_hc_recovery,
)
from fast_agent.cli.runtime.run_request import AgentRunRequest
from fast_agent.integrations import hc_children, herdr_lifecycle
from fast_agent.integrations.hc_children import (
    HcChild,
    HcTerminalChildren,
    read_hc_core_id,
    read_hc_terminal_children,
    remove_hc_terminals,
)
from fast_agent.integrations.herdr_lifecycle import (
    herdr_blocked,
    herdr_startup_hold,
    report_prompt_mark,
)
from fast_agent.session.session_manager import SessionManager

if TYPE_CHECKING:
    from pathlib import Path


class _CaptureTty:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def isatty(self) -> bool:
        return True

    def write(self, data: str) -> int:
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        pass


_FAKE_HC = """#!{python}
import json, os, sys
state = json.load(open({state!r}))
with open({calls!r}, "a") as fh:
    fh.write(json.dumps({{
        "argv": sys.argv[1:],
        "caller": os.environ.get("HC_TERMINAL_ID"),
        "socket": os.environ.get("HC_SOCKET"),
        "session": os.environ.get("HC_SESSION"),
    }}) + chr(10))
cmd = sys.argv[1]
if cmd == "ls":
    print(state["ls"])
elif cmd == "server":
    print(state["server"])
elif cmd == "rm":
    target = sys.argv[-1]
    if target in state.get("fail_rm", []):
        sys.exit(1)
    print(json.dumps({{"removed": target}}))
elif cmd == "wait":
    print(state["wait"])
elif cmd == "get":
    print(json.dumps(state["get"][sys.argv[2]]))
"""


@pytest.fixture
def fake_hc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state = tmp_path / "state.json"
    calls = tmp_path / "calls.jsonl"
    calls.touch()
    executable = tmp_path / "bin" / "hc"
    executable.parent.mkdir()
    executable.write_text(
        _FAKE_HC.format(python=sys.executable, state=str(state), calls=str(calls))
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent))
    monkeypatch.setenv("HC_TERMINAL_ID", "t120")
    monkeypatch.setenv("HC_TERMINAL_SOCKET", "/run/hc.sock")

    def configure(
        *,
        ls: object = None,
        server: object = None,
        fail_rm: list[str] | None = None,
        wait: object = None,
        get: dict[str, dict[str, object]] | None = None,
    ) -> Path:
        payload = {
            "ls": ls
            if isinstance(ls, str)
            else json.dumps(ls if ls is not None else {"terminals": []}),
            "server": server
            if isinstance(server, str)
            else json.dumps(server if server is not None else {"core": {"id": "core-1"}}),
            "fail_rm": fail_rm or [],
            "wait": wait
            if isinstance(wait, str)
            else json.dumps(wait if wait is not None else {"children": []}),
            "get": get or {},
        }
        state.write_text(json.dumps(payload))
        return calls

    return configure


def _make_request(
    *,
    home: Path | None = None,
    resume: str | None = None,
    no_home: bool = False,
    message: str | None = None,
    mode: Literal["interactive", "serve"] = "interactive",
) -> AgentRunRequest:
    return AgentRunRequest(
        name="fast-agent",
        instruction=None,
        config_path=None,
        server_list=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message=message,
        prompt_file=None,
        result_file=None,
        resume=resume,
        startup_mcp_servers=None,
        mcp_startup_notices=(),
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        home=home,
        no_home=no_home,
        shell_runtime=True,
        no_shell=False,
        mode=mode,
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
    )


def _persist_dummy_history(session_dir: Path) -> None:
    (session_dir / "history_agent.json").write_text('{"messages": []}')


@pytest.mark.skipif(sys.platform == "win32", reason="simulated hc is a POSIX script")
def test_passive_discovery_strips_caller_and_filters_direct_local_children(fake_hc) -> None:
    calls = fake_hc(
        ls={
            "terminals": [
                {"address": "t120", "parent": None, "state": "live"},
                {
                    "address": "t1bb",
                    "name": "worker",
                    "parent": "t120",
                    "state": "finished",
                    "reason": "stopped_by_restart",
                },
                {
                    "address": "t1kk",
                    "name": "done",
                    "parent": "t120",
                    "state": "finished",
                    "reason": "exited",
                },
                {"address": "t1ng", "name": "dash", "parent": "t120", "state": "live"},
                {
                    "address": "box/t9",
                    "name": "remote",
                    "parent": "t120",
                    "state": "finished",
                    "reason": "stopped_by_restart",
                },
                {
                    "address": "t1jt",
                    "name": "other",
                    "parent": "t1hy",
                    "state": "finished",
                    "reason": "stopped_by_restart",
                },
            ]
        },
        server={"core": {"id": "core-abc"}},
    )

    assert read_hc_core_id() == "core-abc"
    children = read_hc_terminal_children("t120")
    assert children == HcTerminalChildren(
        live=(HcChild("t1ng", "dash"),),
        stopped_by_restart=(HcChild("t1bb", "worker"),),
    )

    logged = [json.loads(line) for line in calls.read_text().splitlines()]
    assert all(entry["caller"] is None for entry in logged)


@pytest.mark.skipif(sys.platform == "win32", reason="simulated hc is a POSIX script")
def test_read_hc_attachments_pins_socket_and_strips_caller(
    fake_hc, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HC_SOCKET", "/run/other-core.sock")
    monkeypatch.setenv("HC_SESSION", "other-core")
    calls = fake_hc(
        wait={"children": [{"terminal": "t2", "ended": False}]},
        get={"t2": {"name": "probe", "notify": True, "labels": {}}},
    )
    attachments = hc_children.read_hc_attachments()
    assert attachments.resuming == (HcChild("t2", "probe"),)
    logged = [json.loads(line) for line in calls.read_text().splitlines()]
    assert logged[0]["argv"][:4] == ["wait", "t120", "--children", "all"]
    assert all(entry["caller"] is None for entry in logged)
    assert all(entry["socket"] == "/run/hc.sock" for entry in logged)


@pytest.mark.skipif(sys.platform == "win32", reason="simulated hc is a POSIX script")
def test_remove_hc_terminals_reports_partial_failures(fake_hc) -> None:
    fake_hc(fail_rm=["t1bc"])
    removed, failed = remove_hc_terminals(["t1bb", "t1bc", "remote/t9"])
    assert removed == ["t1bb"]
    assert failed == ["t1bc", "remote/t9"]


def test_startup_hold_prevents_osc_7501_and_blocked_clear_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tty = _CaptureTty()
    monkeypatch.setattr(sys, "stdout", tty)
    monkeypatch.delenv("HERDR_ENV", raising=False)
    monkeypatch.delenv("FAST_AGENT_DISABLE_PROGRAM_STATUS", raising=False)
    herdr_lifecycle._reset_for_tests()
    try:
        with herdr_startup_hold():
            with herdr_blocked(kind="question", activate=True):
                pass
            report_prompt_mark("A")
        assert tty.writes == []

        report_prompt_mark("A")
        assert any("\x1b]7501;state=idle" in w for w in tty.writes)
    finally:
        herdr_lifecycle._reset_for_tests()


def test_session_terminal_binding_interrupted_clean_switch_and_fork(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    manager = SessionManager(cwd=tmp_path, home_override=home)
    manager.bind_terminal("t120", core_id="core-1")

    s1 = manager.create_session()
    _persist_dummy_history(s1.directory)
    # Simulate crash: release owner lease without calling close_terminal_binding_cleanly().
    manager.close()

    # Fresh manager sees s1 as interrupted on (core-1, t120), but not on core-2.
    probe = SessionManager(cwd=tmp_path, home_override=home)
    found = probe.find_interrupted_terminal_session("t120", core_id="core-1")
    assert found is not None and found.name == s1.info.name
    assert probe.find_interrupted_terminal_session("t120", core_id="core-2") is None

    # Forked session must not inherit terminal_attachment.
    loaded = probe.load_session(s1.info.name)
    assert loaded is not None
    forked = probe.fork_current_session()
    assert forked is not None
    assert forked.load_snapshot().execution.terminal_attachment is None
    probe.close()

    # Switch-then-clean-exit without another turn closes both old and new cleanly.
    m2 = SessionManager(cwd=tmp_path, home_override=home)
    m2.bind_terminal("t120", core_id="core-1")
    m2.load_session(s1.info.name)
    s2 = m2.create_session()
    _persist_dummy_history(s2.directory)
    m2.close_terminal_binding_cleanly()
    m2.close()

    check = SessionManager(cwd=tmp_path, home_override=home)
    assert check.find_interrupted_terminal_session("t120", core_id="core-1") is None


def test_locked_interrupted_session_is_skipped(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    live_manager = SessionManager(cwd=tmp_path, home_override=home)
    live_manager.bind_terminal("t120", core_id="core-1")
    session = live_manager.create_session()
    _persist_dummy_history(session.directory)

    other_manager = SessionManager(cwd=tmp_path, home_override=home)
    try:
        assert other_manager.find_interrupted_terminal_session("t120", core_id="core-1") is None
    finally:
        live_manager.close()


@pytest.mark.skipif(sys.platform == "win32", reason="simulated hc is a POSIX script")
def test_should_check_hc_recovery_guards(fake_hc, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_hc()
    assert should_check_hc_recovery(_make_request(), stdin_is_tty=True, stdout_is_tty=True)
    assert not should_check_hc_recovery(
        _make_request(resume="latest"), stdin_is_tty=True, stdout_is_tty=True
    )
    assert not should_check_hc_recovery(
        _make_request(no_home=True), stdin_is_tty=True, stdout_is_tty=True
    )
    assert not should_check_hc_recovery(
        _make_request(message="hi"), stdin_is_tty=True, stdout_is_tty=True
    )
    assert not should_check_hc_recovery(
        _make_request(mode="serve"), stdin_is_tty=True, stdout_is_tty=True
    )
    assert not should_check_hc_recovery(_make_request(), stdin_is_tty=False, stdout_is_tty=True)
    monkeypatch.delenv("HC_TERMINAL_ID")
    assert not should_check_hc_recovery(_make_request(), stdin_is_tty=True, stdout_is_tty=True)


@pytest.mark.skipif(sys.platform == "win32", reason="simulated hc is a POSIX script")
@pytest.mark.asyncio
async def test_resolve_hc_startup_recovery_resume_and_fresh_clear(
    tmp_path: Path, fake_hc, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".fast-agent"
    m = SessionManager(cwd=tmp_path, home_override=home)
    m.bind_terminal("t120", core_id="core-1")
    s1 = m.create_session()
    _persist_dummy_history(s1.directory)
    m.close()

    fake_hc(
        ls={
            "terminals": [
                {
                    "address": "t1bb",
                    "name": "worker",
                    "parent": "t120",
                    "state": "finished",
                    "reason": "stopped_by_restart",
                },
            ]
        },
        server={"core": {"id": "core-1"}},
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    req_resume = _make_request(home=home)
    seen_plans: list[HcRecoveryPlan] = []

    def pick_resume(plan: HcRecoveryPlan):
        seen_plans.append(plan)
        return "resume"

    outcome_resume = await resolve_hc_startup_recovery(req_resume, prompt_choice=pick_resume)
    assert outcome_resume == HcRecoveryOutcome(action="resume", resumed_session_id=s1.info.name)
    assert req_resume.resume == s1.info.name
    assert len(seen_plans) == 1
    assert seen_plans[0].stopped_by_restart == ("t1bb",)

    req_clear = _make_request(home=home)
    outcome_clear = await resolve_hc_startup_recovery(
        req_clear, prompt_choice=lambda _p: "fresh_clear"
    )
    assert outcome_clear == HcRecoveryOutcome(action="fresh_clear", removed_terminals=("t1bb",))
    assert req_clear.resume is None
    # Declining retired s1's binding so a 0-turn exit will not re-prompt for s1.
    assert (
        SessionManager(cwd=tmp_path, home_override=home).find_interrupted_terminal_session(
            "t120", core_id="core-1"
        )
        is None
    )


def test_resuming_older_session_then_crashing_finds_resumed_session(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    m = SessionManager(cwd=tmp_path, home_override=home)
    m.bind_terminal("t120", core_id="core-1")
    s_old = m.create_session()
    _persist_dummy_history(s_old.directory)
    s_new = m.create_session()
    _persist_dummy_history(s_new.directory)
    # Switch back to older session without another turn, then crash.
    m.load_session(s_old.info.name)
    m.close()

    found = SessionManager(cwd=tmp_path, home_override=home).find_interrupted_terminal_session(
        "t120", core_id="core-1"
    )
    assert found is not None and found.name == s_old.info.name
