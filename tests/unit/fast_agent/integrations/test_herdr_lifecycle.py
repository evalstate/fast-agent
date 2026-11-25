from __future__ import annotations

import io
import re
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.integrations import herdr_lifecycle, program_status

if TYPE_CHECKING:
    from collections.abc import Iterator

_OSC_7501 = re.compile(r"\x1b\]7501;([^\x1b]*)\x1b\\")


class _TtyStream(io.StringIO):
    def isatty(self) -> bool:
        return True


def _reports(stream: _TtyStream) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    for payload in _OSC_7501.findall(stream.getvalue()):
        fields: dict[str, str] = {}
        for part in payload.split(":"):
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value
        parsed.append(fields)
    return parsed


@pytest.fixture
def tty_terminal(monkeypatch: pytest.MonkeyPatch) -> Iterator[_TtyStream]:
    stream = _TtyStream()
    monkeypatch.delenv("FAST_AGENT_DISABLE_PROGRAM_STATUS", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(program_status, "_status_stream", lambda: stream)
    herdr_lifecycle._reset_for_tests()
    try:
        yield stream
    finally:
        herdr_lifecycle._reset_for_tests()


def test_reports_prompt_lifecycle_in_order_and_releases(tty_terminal: _TtyStream) -> None:
    herdr_lifecycle.report_prompt_mark("A")
    herdr_lifecycle.report_prompt_mark("C")
    herdr_lifecycle.report_agent_state(cast("herdr_lifecycle.HerdrBaseState", "blocked"))
    herdr_lifecycle.report_prompt_mark("D;0")
    herdr_lifecycle.release_agent()

    states = [r["state"] for r in _reports(tty_terminal)]
    assert states == ["idle", "working", "idle", "clear"]


def test_nested_blocked_scope_restores_latest_base_state(tty_terminal: _TtyStream) -> None:
    herdr_lifecycle.report_prompt_mark("A")
    herdr_lifecycle.report_agent_state("working")
    with herdr_lifecycle.herdr_blocked(kind="question"):
        herdr_lifecycle.report_agent_state("idle")
        with herdr_lifecycle.herdr_blocked(kind="permission"):
            pass

    states = [(r["state"], r.get("kind")) for r in _reports(tty_terminal)]
    assert states == [
        ("idle", None),
        ("working", None),
        ("blocked", "question"),
        ("idle", None),
    ]


def test_nested_work_keeps_pane_working_until_outer_scope_completes(
    tty_terminal: _TtyStream,
) -> None:
    herdr_lifecycle.report_prompt_mark("A")
    with herdr_lifecycle.herdr_working():
        with herdr_lifecycle.herdr_working():
            pass
    herdr_lifecycle.release_agent()

    states = [r["state"] for r in _reports(tty_terminal)]
    assert states == ["idle", "working", "idle", "clear"]


def test_release_is_idempotent_and_late_reports_are_ignored(tty_terminal: _TtyStream) -> None:
    herdr_lifecycle.report_prompt_mark("A")
    herdr_lifecycle.release_agent()
    herdr_lifecycle.release_agent()
    herdr_lifecycle.report_agent_state("working")

    states = [r["state"] for r in _reports(tty_terminal)]
    assert states == ["idle", "clear"]


def test_blocked_scope_is_fail_open(tty_terminal: _TtyStream) -> None:
    herdr_lifecycle.report_prompt_mark("A")
    with pytest.raises(RuntimeError, match="boom"):
        with herdr_lifecycle.herdr_blocked(kind="question"):
            raise RuntimeError("boom")

    states = [r["state"] for r in _reports(tty_terminal)]
    assert states == ["idle", "blocked", "idle"]


def test_after_fork_child_replaces_inherited_lock_and_disables_reporting(
    tty_terminal: _TtyStream,
) -> None:
    herdr_lifecycle.report_prompt_mark("A")
    before = len(_reports(tty_terminal))
    herdr_lifecycle._after_fork_child()
    herdr_lifecycle.report_agent_state("working")
    assert len(_reports(tty_terminal)) == before
