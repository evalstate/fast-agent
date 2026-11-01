from __future__ import annotations

import base64
import errno
import io
import os
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from fast_agent.integrations import herdr_lifecycle, herdr_session, program_status
from fast_agent.integrations.program_status import ProgramStatus, ProgramStatusReporter
from fast_agent.ui import prompt_marks

if TYPE_CHECKING:
    from collections.abc import Iterator


class TtyStream(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> Iterator[TtyStream]:
    stream = TtyStream()
    output = SimpleNamespace(stdout=stream)
    monkeypatch.setattr(program_status, "sys", output)
    monkeypatch.setattr(prompt_marks, "sys", output)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("FAST_AGENT_DISABLE_PROGRAM_STATUS", raising=False)
    yield stream
    herdr_lifecycle.release_agent()


def reports(stream: io.StringIO) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for body in re.findall(r"\x1b\]7501;([^\x1b]*)\x1b\\", stream.getvalue()):
        record = dict(pair.split("=", 1) for pair in body.split(":"))
        for key in ("title", "msg"):
            if key in record:
                record[key] = base64.b64decode(record[key], validate=True).decode("utf-8")
        result.append(record)
    return result


def metadata(*, title: str | None = "Review auth", usage: str | None = "$0.0123") -> None:
    herdr_lifecycle.report_session_metadata(
        session_id="session-1",
        title=title,
        model="test-model",
        agent_name="agent",
        pinned=False,
        forked_from=None,
        context_usage="35% - test-model",
        token_usage=usage,
    )


def test_encoder_is_bounded_utf8_and_cannot_inject_terminal_controls() -> None:
    text = "review:\n\t\x00\x1b]52;\x7f\x85\x9b café 🔎" * 1000
    sequence = ProgramStatus("working", text, text).encode()
    stream = io.StringIO(sequence)
    report = reports(stream)[0]
    assert report["state"] == "working"
    assert report["app"] == "fast-agent"
    assert len(sequence.encode("utf-8")) <= 4096
    for key, limit in (("title", 192), ("msg", 2048)):
        assert len(report[key].encode("utf-8")) <= limit
        assert all(ord(c) >= 32 and not 0x7F <= ord(c) <= 0x9F for c in report[key])
        assert "café 🔎" in report[key]
    assert sequence.count("\x1b") == 2


def test_interactive_lifecycle_without_herdr_keeps_full_snapshots(terminal: TtyStream) -> None:
    metadata()
    assert terminal.getvalue() == ""  # Metadata alone cannot opt a server into OSC.
    herdr_lifecycle.report_prompt_mark("A")
    herdr_lifecycle.report_prompt_mark("C")
    with herdr_lifecycle.herdr_working(), herdr_lifecycle.herdr_working():
        with herdr_lifecycle.herdr_blocked(), herdr_lifecycle.herdr_blocked():
            pass
    herdr_lifecycle.report_prompt_mark("D;0")
    herdr_lifecycle.release_agent()
    herdr_lifecycle.release_agent()
    herdr_lifecycle.report_prompt_mark("A")

    records = reports(terminal)
    assert [r["state"] for r in records] == [
        "idle",
        "working",
        "blocked",
        "working",
        "idle",
        "clear",
    ]
    for record in records[:-1]:
        assert record["app"] == "fast-agent"
        assert record["title"] == "Review auth"
        assert record["msg"] == "35% - test-model · $0.0123"
    assert records[-1] == {"state": "clear"}


def test_metadata_updates_do_not_end_a_turn_and_plugin_usage_survives_state_changes(
    terminal: TtyStream,
) -> None:
    herdr_lifecycle.report_prompt_mark("C")
    metadata()
    herdr_lifecycle.report_session_usage("$0.0200 custom")
    metadata()  # Same base metadata must not erase the plugin projection.
    with herdr_lifecycle.herdr_blocked():
        pass
    herdr_lifecycle.report_prompt_mark("D")
    records = reports(terminal)
    assert [r["state"] for r in records] == [
        "working",
        "working",
        "working",
        "blocked",
        "working",
        "idle",
    ]
    assert all(r["msg"].endswith("$0.0200 custom") for r in records[2:])
    metadata(title=None, usage=None)
    assert "title" not in reports(terminal)[-1]
    assert reports(terminal)[-1]["msg"] == "35% - test-model"


def test_unchanged_snapshots_are_not_repeated_except_after_prompt_start(
    terminal: TtyStream,
) -> None:
    herdr_lifecycle.report_prompt_mark("C")
    herdr_lifecycle.report_prompt_mark("C")
    herdr_lifecycle.report_prompt_mark("A")
    herdr_lifecycle.report_prompt_mark("A")
    assert [r["state"] for r in reports(terminal)] == ["working", "idle", "idle"]


def test_prompt_start_restores_working_and_blocked_records(terminal: TtyStream) -> None:
    herdr_lifecycle.report_prompt_mark("A")
    with herdr_lifecycle.herdr_working():
        herdr_lifecycle.report_prompt_mark("A")
        with herdr_lifecycle.herdr_blocked():
            herdr_lifecycle.report_prompt_mark("A")
    assert [r["state"] for r in reports(terminal)] == [
        "idle",
        "working",
        "working",
        "blocked",
        "blocked",
        "working",
        "idle",
    ]


def test_program_status_follows_osc133_prompt_start(
    terminal: TtyStream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prompt_marks, "prompt_mark_sequence", lambda *args, **kwargs: "\x1b]133;A\x07"
    )
    prompt_marks.emit_prompt_mark("A")
    assert terminal.getvalue().startswith("\x1b]133;A\x07\x1b]7501;state=idle")


def test_program_status_does_not_depend_on_prompt_marks_setting(
    terminal: TtyStream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompt_marks, "prompt_mark_sequence", lambda *args, **kwargs: "")
    prompt_marks.emit_prompt_mark("A")
    assert reports(terminal) == [{"state": "idle", "app": "fast-agent"}]


@pytest.mark.parametrize("disabled", ["1", "true", "YES", "on"])
def test_environment_opt_out(
    terminal: TtyStream,
    monkeypatch: pytest.MonkeyPatch,
    disabled: str,
) -> None:
    monkeypatch.setenv("FAST_AGENT_DISABLE_PROGRAM_STATUS", disabled)
    herdr_lifecycle.report_prompt_mark("A")
    metadata()
    herdr_lifecycle.report_prompt_mark("C")
    herdr_lifecycle.release_agent()
    assert terminal.getvalue() == ""


def test_false_opt_out_value_does_not_disable(
    terminal: TtyStream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAST_AGENT_DISABLE_PROGRAM_STATUS", "0")
    herdr_lifecycle.report_prompt_mark("A")
    assert reports(terminal)[0]["state"] == "idle"


def test_dumb_terminal_is_silent(
    terminal: TtyStream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM", "dumb")
    herdr_lifecycle.report_prompt_mark("A")
    herdr_lifecycle.release_agent()
    assert terminal.getvalue() == ""


def test_redirected_output_is_silent_even_inside_core(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(program_status, "sys", SimpleNamespace(stdout=stream))
    monkeypatch.setenv("HC_TERMINAL_ID", "t2")
    monkeypatch.delenv("FAST_AGENT_DISABLE_PROGRAM_STATUS", raising=False)
    herdr_lifecycle.report_prompt_mark("A")
    metadata()
    herdr_lifecycle.release_agent()
    assert stream.getvalue() == ""


def test_output_is_rechecked_after_activation(
    terminal: TtyStream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    herdr_lifecycle.report_prompt_mark("A")
    redirected = io.StringIO()
    with monkeypatch.context() as patch:
        patch.setattr(program_status.sys, "stdout", redirected)
        with herdr_lifecycle.herdr_blocked():
            pass
        metadata()
    assert redirected.getvalue() == ""
    herdr_lifecycle.report_prompt_mark("A")
    assert [r["state"] for r in reports(terminal)] == ["idle", "idle"]


def test_protocol_working_scopes_and_metadata_never_activate_osc(terminal: TtyStream) -> None:
    # ACP uses these scopes even if someone launches it directly from a TTY.
    with herdr_lifecycle.herdr_working(), herdr_lifecycle.herdr_blocked():
        metadata()
        herdr_lifecycle.report_session_usage("$1.0000")
    herdr_lifecycle.release_agent()
    assert terminal.getvalue() == ""


def test_after_fork_child_cannot_write_or_clear_parent_status(terminal: TtyStream) -> None:
    herdr_lifecycle.report_prompt_mark("A")
    before = terminal.getvalue()
    herdr_lifecycle._after_fork_child()
    herdr_lifecycle.report_prompt_mark("C")
    herdr_lifecycle.release_agent()
    assert terminal.getvalue() == before


def test_session_presentation_is_no_longer_gated_on_herdr(
    terminal: TtyStream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        herdr_session, "_usage_metadata", lambda agent, *, model: ("42% - example", "$0.1000")
    )
    herdr_session.report_session(None)
    herdr_lifecycle.report_prompt_mark("A")
    assert reports(terminal)[0]["msg"] == "42% - example · $0.1000"


def test_output_failures_are_best_effort_and_can_retry(
    terminal: TtyStream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = ProgramStatusReporter()
    reporter.activate()
    for error in (OSError("broken pipe"), ValueError("closed")):
        with monkeypatch.context() as patch:

            def fail_write(_value: str) -> int:
                raise error

            patch.setattr(terminal, "write", fail_write)
            reporter.report_state("working")
    assert terminal.getvalue() == ""
    reporter.report_state("working")
    assert reports(terminal) == [{"state": "working", "app": "fast-agent"}]
    reporter.release()


def test_blocked_scope_restores_status_on_exception(terminal: TtyStream) -> None:
    herdr_lifecycle.report_prompt_mark("C")
    with pytest.raises(RuntimeError), herdr_lifecycle.herdr_blocked():
        raise RuntimeError("cancelled")
    herdr_lifecycle.report_prompt_mark("D;1")
    assert [r["state"] for r in reports(terminal)] == ["working", "blocked", "working", "idle"]


def test_rich_streaming_proxy_receives_no_control_text(
    terminal: TtyStream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rich.console import Console
    from rich.file_proxy import FileProxy

    rendered = io.StringIO()
    proxy = FileProxy(Console(file=rendered), terminal)
    monkeypatch.setattr(program_status.sys, "stdout", proxy)
    herdr_lifecycle.report_prompt_mark("C")
    metadata()
    with herdr_lifecycle.herdr_blocked():
        pass
    herdr_lifecycle.release_agent()
    assert [r["state"] for r in reports(terminal)] == [
        "working",
        "working",
        "blocked",
        "working",
        "clear",
    ]
    assert rendered.getvalue() == ""


def test_rich_proxy_does_not_make_redirected_output_eligible(
    terminal: TtyStream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rich.console import Console
    from rich.file_proxy import FileProxy

    redirected = io.StringIO()
    proxy = FileProxy(Console(file=terminal), redirected)
    monkeypatch.setattr(program_status.sys, "stdout", proxy)
    herdr_lifecycle.report_prompt_mark("A")
    herdr_lifecycle.release_agent()
    assert terminal.getvalue() == redirected.getvalue() == ""


@pytest.mark.skipif(sys.platform == "win32", reason="Requires a POSIX PTY")
def test_real_pty_lifecycle_and_atexit_cleanup() -> None:
    code = """
from fast_agent.integrations.herdr_lifecycle import report_prompt_mark, herdr_blocked
report_prompt_mark("A")
report_prompt_mark("C")
with herdr_blocked():
    pass
report_prompt_mark("D")
"""
    env = dict(os.environ)
    env.pop("FAST_AGENT_DISABLE_PROGRAM_STATUS", None)
    env["TERM"] = "xterm-256color"
    master, slave = os.openpty()
    try:
        try:
            child = subprocess.run(
                [sys.executable, "-c", code],
                stdout=slave,
                stderr=subprocess.PIPE,
                env=env,
                timeout=30,
                check=True,
            )
        finally:
            os.close(slave)
        assert child.stderr == b""
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(master, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:  # Linux's PTY EOF.
                    break
                raise
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(master)
    assert [r["state"] for r in reports(io.StringIO(b"".join(chunks).decode()))] == [
        "idle",
        "working",
        "blocked",
        "working",
        "idle",
        "clear",
    ]

    # Identical code with piped stdout must emit no bytes at all.
    piped = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=env,
        timeout=30,
        check=True,
    )
    assert piped.stdout == piped.stderr == b""
