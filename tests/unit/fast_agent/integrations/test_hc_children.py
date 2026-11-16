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
    format_status_summary,
    format_turn_end_lines,
    hc_message_heading,
    read_hc_attachments,
)

if TYPE_CHECKING:
    from pathlib import Path

# A stand-in for the hc CLI: answers `wait --children` and `get <t>` from a JSON file.
_FAKE_HC = """#!{python}
import json, sys
state = json.load(open({state!r}))
open({calls!r}, "a").write(" ".join(sys.argv[1:]) + "\\n")
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
    hc(
        {
            "reason": "message",  # a pending message must not hide the children
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
        "hc: 2 running - ptcoff (t134), t135 · will resume when they report or exit · 2 services running",
        "1 background process still running; nothing will resume this session when it ends",
    ]
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
