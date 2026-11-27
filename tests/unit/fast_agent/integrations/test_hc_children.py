"""hc attachment snapshots, read through a simulated ``hc`` executable."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from fast_agent.integrations import hc_children
from fast_agent.integrations.hc_children import (
    HcAttachments,
    HcChild,
    format_hc_elapsed,
    format_status_summary,
    format_turn_end_lines,
    hc_message_heading,
    read_hc_attachments,
    unwrap_hc_message,
)
from fast_agent.ui.interactive_prompt import _turn_end_line_text

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.core.agent_app import AgentApp

# A stand-in for the hc CLI: answers `wait <parent> --children` and `get <t>` from a JSON file.
_FAKE_HC = """#!{python}
import json, os, sys
state = json.load(open({state!r}))
open({calls!r}, "a").write(json.dumps({{
    "argv": sys.argv[1:],
    "caller": os.environ.get("HC_TERMINAL_ID"),
    "socket": os.environ.get("HC_SOCKET"),
    "session": os.environ.get("HC_SESSION"),
}}) + chr(10))
if sys.argv[1] == "wait":
    print(state["wait"])
else:
    print(json.dumps(state["get"][sys.argv[2]]))
"""


def _child(address: str, *, ended: bool = False) -> dict[str, object]:
    return {"terminal": address, "ended": ended, "reason": "running", "children": []}


@pytest.fixture
def hc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state = tmp_path / "state.json"
    calls = tmp_path / "calls"
    calls.touch()
    executable = tmp_path / "bin" / "hc"
    executable.parent.mkdir()
    executable.write_text(
        _FAKE_HC.format(python=sys.executable, state=str(state), calls=str(calls))
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent))
    monkeypatch.setenv("HC_TERMINAL_ID", "t1")
    monkeypatch.setenv("HC_TERMINAL_SOCKET", "/run/hc.sock")

    def configure(wait: object, get: dict[str, dict[str, object]] | None = None) -> Path:
        text = wait if isinstance(wait, str) else json.dumps(wait)
        state.write_text(json.dumps({"wait": text, "get": get or {}}))
        return calls

    return configure


@pytest.mark.skipif(sys.platform == "win32", reason="simulated hc is a POSIX script")
def test_unfinished_children_are_sorted_by_what_they_mean_for_the_session(hc) -> None:
    calls = hc(
        {
            "reason": "message",
            "children": [
                _child("t2"),
                _child("t3"),
                _child("t4"),
                _child("t5"),
                _child("t6", ended=True),
            ],
        },
        {
            "t2": {"name": "probe", "notify": True, "reported": False, "labels": {}},
            "t3": {"name": "dash", "notify": True, "labels": {"role": "service"}},
            "t4": {"name": "quiet-build", "notify": False, "reported": False},
            "t5": {"name": None, "notify": False, "reported": True},
        },
    )
    assert read_hc_attachments() == HcAttachments(
        resuming=(HcChild("t2", "probe"), HcChild("t5")), services=1
    )
    logged = [json.loads(line) for line in calls.read_text().splitlines()]
    assert logged[0]["argv"][:4] == ["wait", "t1", "--children", "all"]
    assert all(entry["caller"] is None for entry in logged)
    assert all(entry["socket"] == "/run/hc.sock" for entry in logged)


@pytest.mark.skipif(sys.platform == "win32", reason="simulated hc is a POSIX script")
@pytest.mark.parametrize("wait", ["not json", "[]", {"error": {"kind": "not_found"}}])
def test_unusable_hc_output_reads_as_nothing_attached(hc, wait: object) -> None:
    hc(wait)
    assert read_hc_attachments() == HcAttachments()


@pytest.mark.skipif(sys.platform == "win32", reason="simulated hc is a POSIX script")
@pytest.mark.asyncio
async def test_hc_is_not_run_outside_an_hc_terminal(hc, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = hc({"children": [_child("t2")]}, {"t2": {"notify": True}})
    monkeypatch.delenv("HC_TERMINAL_SOCKET")
    assert await hc_children.refresh_hc_attachments() == HcAttachments()
    assert hc_children.current_hc_attachments() == HcAttachments()
    assert calls.read_text() == ""


def test_turn_end_lines_say_what_resumes_the_session_and_what_does_not() -> None:
    attachments = HcAttachments(resuming=(HcChild("t134", "ptcoff"), HcChild("t135")), services=2)
    assert format_turn_end_lines(attachments, 1) == [
        (
            "⧗ hc · waiting on ptcoff (t134), t135 · resumes when they report or exit"
            " · 2 services running"
        ),
        "1 background process still running; nothing will resume this session when it ends",
    ]
    assert format_turn_end_lines(HcAttachments(resuming=(HcChild("t2", "probe"),)), 0) == [
        "⧗ hc · waiting on probe (t2) · resumes when it reports or exits"
    ]
    assert format_turn_end_lines(HcAttachments(services=1), 0) == ["hc · 1 service running"]
    assert format_turn_end_lines(HcAttachments(), 0) == []
    assert format_status_summary(attachments, 3) == (
        "waiting on 2: ptcoff, t135, 2 services, 3 bg (no resume)"
    )
    assert format_status_summary(HcAttachments(), 0) is None


@pytest.mark.parametrize(
    ("texts", "heading"),
    [
        (["<hc>t135 probe exited (exit 0). Read: hc read t135</hc>"], "hc · t135 exited"),
        (['<hc from="t7 review" reach="hc send t7">Review x</hc>'], "hc · t7 review"),
        (["<hc>something new</hc>"], "hc"),
        (["<hc>t1 a exited</hc>", '<hc from="t2">b</hc>'], "hc · 2 messages"),
        (["we <hc>t1 a exited</hc>"], None),
        (["<hcard>not hc</hcard>"], None),
        (["typed by a person"], None),
    ],
)
def test_only_input_delivered_by_hc_gets_an_hc_heading(
    texts: list[str], heading: str | None
) -> None:
    assert hc_message_heading(texts) == heading


def test_turn_end_lines_weight_the_state_nothing_will_resume() -> None:
    lines = format_turn_end_lines(
        HcAttachments(resuming=(HcChild("t2"),), services=1), background_processes=2
    )
    waiting, trap = (_turn_end_line_text(line) for line in lines)

    assert waiting.plain == lines[0]
    assert [(waiting.plain[span.start : span.end], span.style) for span in waiting.spans] == [
        ("⧗", "cyan"),
        (lines[0].removeprefix("⧗"), "dim"),
    ]
    assert trap.style == "yellow"
    assert _turn_end_line_text("hc · 1 service running").style == "dim"


def test_wait_clock_runs_while_children_will_resume_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(hc_children.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(hc_children, "_current", HcAttachments())
    monkeypatch.setattr(hc_children, "_waiting_since", None)
    waiting = HcAttachments(resuming=(HcChild("t2"),))

    hc_children._update_current(waiting)
    now[0] = 400.0
    # A later snapshot with the child still unfinished keeps the original start.
    hc_children._update_current(HcAttachments(resuming=(HcChild("t2"), HcChild("t3"))))
    assert hc_children.hc_waiting_since() == 100.0

    hc_children._update_current(HcAttachments(services=1))
    assert hc_children.hc_waiting_since() is None
    hc_children._update_current(waiting)
    assert hc_children.hc_waiting_since() == 400.0


@pytest.mark.parametrize(
    ("seconds", "elapsed"),
    [(0, None), (59.9, None), (60, "1m"), (3599, "59m"), (3600, "1h00m"), (7380, "2h03m")],
)
def test_wait_duration_is_coarse_and_silent_for_the_first_minute(
    seconds: float, elapsed: str | None
) -> None:
    assert format_hc_elapsed(seconds) == elapsed


@pytest.mark.parametrize(
    ("text", "body"),
    [
        ('<hc from="t7 review" reach="hc send t7">Review ok</hc>', "Review ok"),
        (
            "<hc>\nt1 probe exited (exit 0).\n\nRead: hc read t1\n</hc>\n",
            "t1 probe exited (exit 0).\n\nRead: hc read t1",
        ),
        ("<hc>no closing tag", "no closing tag"),
        ("we <hc>t1 a exited</hc>", "we <hc>t1 a exited</hc>"),
        ("<hcard>not hc</hcard>", "<hcard>not hc</hcard>"),
    ],
)
def test_unwrapping_shows_only_the_body_of_hc_messages(text: str, body: str) -> None:
    assert unwrap_hc_message(text) == body


def test_multi_notice_paste_unwraps_cleanly_and_counts_all_blocks() -> None:
    pasted = "<hc>t1 w1 exited (exit 0). Read: hc read t1</hc>\n<hc>t2 w2 hit an error: boom. See: hc read t2</hc>"
    assert unwrap_hc_message(pasted) == (
        "t1 w1 exited (exit 0). Read: hc read t1\n\nt2 w2 hit an error: boom. See: hc read t2"
    )
    assert hc_children.hc_message_heading([pasted]) == "hc · 2 messages"
    assert (
        hc_children.hc_message_heading(
            ["<hc>t7 worker is blocked (permission). See: hc read t7</hc>"]
        )
        == "hc · t7 blocked"
    )
    assert (
        hc_children.hc_message_heading(
            ["<hc>t7 worker hit an error: rate limited. See: hc read t7</hc>"]
        )
        == "hc · t7 error"
    )


def test_turn_end_lines_and_status_summary_include_event_waits() -> None:
    label1 = "jobs/job.status_changed (job_id=abc123)"
    label2 = "ci/build.finished"

    assert format_turn_end_lines(HcAttachments(event_waits=(label1,)), 0) == [
        "⧗ events · waiting on jobs/job.status_changed (job_id=abc123) · resumes when an event arrives"
    ]
    assert format_turn_end_lines(
        HcAttachments(event_waits=(label1, label2), event_wake_paused=True),
        0,
    ) == ["⧗ events · waiting on 2 events · wake paused"]
    assert format_turn_end_lines(
        HcAttachments(
            resuming=(HcChild("t7", "review"),),
            event_waits=(label1, label2),
        ),
        0,
    ) == ["⧗ hc · waiting on review (t7) · resumes when it reports or exits · 2 events"]

    assert (
        format_status_summary(HcAttachments(event_waits=(label1,)), 0)
        == "waiting on jobs/job.status_changed (job_id=abc123)"
    )
    assert (
        format_status_summary(
            HcAttachments(
                resuming=(HcChild("t7", "review"),),
                event_waits=(label1, label2),
            ),
            0,
        )
        == "waiting on 1: review, 2 events"
    )
    assert (
        format_status_summary(
            HcAttachments(event_waits=(label1, label2), event_wake_paused=True),
            0,
        )
        == "waiting on 2 events (wake paused)"
    )


@pytest.mark.asyncio
async def test_refresh_session_attachments_reports_event_waits_outside_hc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import cast

    from fast_agent.commands.context import StaticAgentProvider
    from fast_agent.mcp.events_subscriptions import EventSubscription
    from fast_agent.ui.prompt import input as prompt_input

    monkeypatch.delenv("HC_TERMINAL_ID", raising=False)
    monkeypatch.delenv("HC_TERMINAL_SOCKET", raising=False)
    reported: list[str | None] = []
    monkeypatch.setattr(prompt_input, "report_session_attached", reported.append)

    class _WakeAgent:
        def __init__(self, subs: list[EventSubscription], paused: bool = False) -> None:
            self._subs = subs
            self._paused = paused

        def event_wait_state(self) -> tuple[list[EventSubscription], bool]:
            return list(self._subs), self._paused

    sub = EventSubscription(
        server_name="jobs",
        name="job.status_changed",
        arguments={"job_id": "abc123"},
        wake=True,
    )
    provider = StaticAgentProvider({"main": _WakeAgent([sub])})
    lines = await prompt_input.refresh_session_attachments(
        agent_name="main",
        agent_provider=cast("AgentApp", provider),
    )

    assert lines == [
        "⧗ events · waiting on jobs/job.status_changed (job_id=abc123) · resumes when an event arrives"
    ]
    assert reported == ["waiting on jobs/job.status_changed (job_id=abc123)"]
