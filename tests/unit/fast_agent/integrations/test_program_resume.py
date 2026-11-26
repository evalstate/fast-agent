from __future__ import annotations

import base64
import hashlib
import io
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast

import pytest

from fast_agent.cli.runtime.program_resume import build_cli_resume_command
from fast_agent.cli.runtime.run_request import AgentRunRequest
from fast_agent.integrations import herdr_lifecycle, herdr_session, program_resume
from fast_agent.integrations.program_resume import (
    PROGRAM_RESUME_TOKEN_ENV,
    ProgramResumeCommand,
    ProgramResumeReporter,
    format_program_resume,
    format_program_resume_clear,
)
from fast_agent.session.session_manager import Session, SessionInfo

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class TtyStream(io.StringIO):
    def isatty(self) -> bool:
        return True


def _parse_7577(output: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for body in re.findall(r"\x1b\]7577;([^\x1b]*)\x1b\\", output):
        records.append(dict(pair.split("=", 1) for pair in body.split(":")))
    return records


def _decode_argv(encoded: str) -> list[str]:
    return [base64.b64decode(part, validate=True).decode("utf-8") for part in encoded.split(",")]


def _make_request(
    *,
    name: str = "fast-agent",
    server_list: list[str] | None = None,
    model: str | None = "sonnet",
    message: str | None = None,
    home: Path | None = None,
    no_home: bool = False,
    shell_runtime: bool = False,
    mode: Literal["interactive", "serve"] = "interactive",
    workspace: Path | None = None,
    subagents: bool | None = None,
    model_base_url: str | None = None,
    isolated: bool = False,
    ptc: bool | None = None,
) -> AgentRunRequest:
    return AgentRunRequest(
        name=name,
        instruction=None,
        config_path=None,
        server_list=server_list,
        agent_cards=None,
        card_tools=None,
        model=model,
        message=message,
        prompt_file=None,
        result_file=None,
        resume=None,
        startup_mcp_servers=None,
        mcp_startup_notices=(),
        agent_name=None,
        target_agent_name=None,
        skills_directory=None,
        home=home,
        no_home=no_home,
        shell_runtime=shell_runtime,
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
        workspace=workspace,
        subagents=subagents,
        model_base_url=model_base_url,
        isolated=isolated,
        ptc=ptc,
    )


@pytest.fixture
def resume_terminal(monkeypatch: pytest.MonkeyPatch) -> Iterator[TtyStream]:
    stream = TtyStream()
    monkeypatch.setattr(program_resume, "sys", SimpleNamespace(stdout=stream))
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("FAST_AGENT_DISABLE_PROGRAM_RESUME", raising=False)
    program_resume._reset_for_tests()
    yield stream
    program_resume._reset_for_tests()
    herdr_lifecycle._reset_for_tests()


def test_format_program_resume_and_clear_match_spec_mac() -> None:
    token = "dummy-token-replace-me"
    clear_seq = format_program_resume_clear(token=token, pid=74877, now=1795543200)
    assert clear_seq is not None
    expected_clear_mac = hashlib.sha256(
        b"pid=74877:time=1795543200:clear=1:dummy-token-replace-me"
    ).hexdigest()
    assert clear_seq == (
        f"\x1b]7577;pid=74877:time=1795543200:clear=1:mac={expected_clear_mac}\x1b\\"
    )

    cmd = ProgramResumeCommand(
        argv=("fast-agent", "go", "--resume", "2611260946-kthqY9"),
        cwd="/tmp/workspace",
    )
    seq = format_program_resume(cmd, token=token, pid=74877, now=1795543200)
    assert seq is not None
    records = _parse_7577(seq)
    assert len(records) == 1
    rec = records[0]
    assert list(rec.keys())[-1] == "mac"
    assert _decode_argv(rec["argv"]) == ["fast-agent", "go", "--resume", "2611260946-kthqY9"]
    assert base64.b64decode(rec["cwd"], validate=True).decode("utf-8") == "/tmp/workspace"
    body_before_mac = seq[len("\x1b]7577;") : seq.rfind(":mac=")]
    assert rec["mac"] == hashlib.sha256(f"{body_before_mac}:{token}".encode("utf-8")).hexdigest()


def test_format_rejects_invalid_paths_controls_and_oversized_sequences() -> None:
    token = "a" * 32
    assert format_program_resume(ProgramResumeCommand(argv=()), token=token, pid=1, now=10) is None
    assert (
        format_program_resume(
            ProgramResumeCommand(argv=("./fast-agent",)), token=token, pid=1, now=10
        )
        is None
    )
    assert (
        format_program_resume(ProgramResumeCommand(argv=("-bad",)), token=token, pid=1, now=10)
        is None
    )
    assert (
        format_program_resume(
            ProgramResumeCommand(argv=("fast-agent", "bad\narg")),
            token=token,
            pid=1,
            now=10,
        )
        is None
    )
    assert (
        format_program_resume(
            ProgramResumeCommand(argv=("fast-agent",), cwd="relative/path"),
            token=token,
            pid=1,
            now=10,
        )
        is None
    )
    assert (
        format_program_resume(
            ProgramResumeCommand(argv=("fast-agent",) + ("a",) * 64),
            token=token,
            pid=1,
            now=10,
        )
        is None
    )
    assert (
        format_program_resume(
            ProgramResumeCommand(argv=("fast-agent", "x" * 2000)),
            token=token,
            pid=1,
            now=10,
        )
        is None
    )


def test_reporter_clears_stale_record_when_next_command_is_unreportable(
    resume_terminal: TtyStream,
) -> None:
    reporter = ProgramResumeReporter("t" * 32, pid=42, clock=lambda: 1795543200)
    assert reporter.report(ProgramResumeCommand(argv=("fast-agent", "go", "--resume", "s1")))
    assert reporter.report(ProgramResumeCommand(argv=("fast-agent", "x" * 2000)))
    records = _parse_7577(resume_terminal.getvalue())
    assert len(records) == 2
    assert "argv" in records[0]
    assert records[1].get("clear") == "1"
    assert reporter.reported_command is None


def test_capture_strips_token_and_lifecycle_reports_refreshes_and_preserves_on_deactivate(
    monkeypatch: pytest.MonkeyPatch,
    resume_terminal: TtyStream,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(PROGRAM_RESUME_TOKEN_ENV, "secret-token-0123456789abcdef")
    assert program_resume.capture_program_resume_token() is True
    assert PROGRAM_RESUME_TOKEN_ENV not in program_resume.os.environ

    request = _make_request(shell_runtime=True, workspace=tmp_path)
    program_resume.configure_program_resume(
        lambda sid: build_cli_resume_command(request, sid, cwd=tmp_path)
    )

    now = datetime.now(timezone.utc)
    empty_session = cast(
        "Session",
        SimpleNamespace(
            info=SessionInfo(name="empty-1", created_at=now, last_activity=now, history_files=[])
        ),
    )
    saved_session = cast(
        "Session",
        SimpleNamespace(
            info=SessionInfo(
                name="2611260946-abc123",
                created_at=now,
                last_activity=now,
                history_files=["history_agent.json"],
            )
        ),
    )

    # Unsaved session emits nothing.
    herdr_session.report_session(empty_session)
    assert _parse_7577(resume_terminal.getvalue()) == []

    # First persisted turn emits resume command; duplicate call is deduplicated.
    herdr_session.report_session(saved_session)
    herdr_session.report_session(saved_session)
    records = _parse_7577(resume_terminal.getvalue())
    assert len(records) == 1
    decoded = _decode_argv(records[0]["argv"])
    assert "go" in decoded and "--resume" in decoded and "2611260946-abc123" in decoded
    assert "--shell" in decoded
    assert "--model" not in decoded

    # SIGCONT refresh re-emits the current command.
    program_resume.refresh_program_resume()
    assert len(_parse_7577(resume_terminal.getvalue())) == 2

    # Switching to an empty session withdraws the record.
    herdr_session.report_session(empty_session)
    records = _parse_7577(resume_terminal.getvalue())
    assert len(records) == 3
    assert records[2].get("clear") == "1"

    # Re-report, then deactivate (terminal close / SIGHUP) must NOT send clear=1.
    herdr_session.report_session(saved_session)
    assert len(_parse_7577(resume_terminal.getvalue())) == 4
    program_resume.deactivate_program_resume()
    assert len(_parse_7577(resume_terminal.getvalue())) == 4

    # Intentional user exit DOES send clear=1.
    program_resume.clear_program_resume()
    records = _parse_7577(resume_terminal.getvalue())
    assert len(records) == 5
    assert records[4].get("clear") == "1"


def test_disable_env_suppresses_output_while_still_stripping_token(
    monkeypatch: pytest.MonkeyPatch,
    resume_terminal: TtyStream,
) -> None:
    monkeypatch.setenv(PROGRAM_RESUME_TOKEN_ENV, "secret-token-0123456789abcdef")
    monkeypatch.setenv("FAST_AGENT_DISABLE_PROGRAM_RESUME", "1")
    assert program_resume.capture_program_resume_token() is True
    assert PROGRAM_RESUME_TOKEN_ENV not in program_resume.os.environ

    program_resume.configure_program_resume(
        lambda sid: (
            ProgramResumeCommand(argv=("fast-agent", "go", "--resume", sid)) if sid else None
        )
    )
    program_resume.report_program_resume_session("sess-1")
    assert resume_terminal.getvalue() == ""


def test_build_cli_resume_command_respects_mode_and_flags(tmp_path: Path) -> None:
    req = _make_request(
        name="worker-1",
        home=tmp_path / "home",
        shell_runtime=True,
        subagents=False,
        ptc=True,
        server_list=["srv1", "srv2"],
        model="opus",
    )
    cmd = build_cli_resume_command(req, "sess-42", cwd=tmp_path)
    assert cmd is not None
    assert cmd.cwd == str(tmp_path.resolve())
    assert "--resume" in cmd.argv and "sess-42" in cmd.argv
    assert "--name" in cmd.argv and "worker-1" in cmd.argv
    assert "--shell" in cmd.argv
    assert "--no-subagents" in cmd.argv
    assert "--ptc" in cmd.argv
    assert "--servers" in cmd.argv and "srv1,srv2" in cmd.argv
    assert "--model" not in cmd.argv

    # Non-resumable modes return None.
    assert build_cli_resume_command(_make_request(no_home=True), "sess-42", cwd=tmp_path) is None
    assert build_cli_resume_command(_make_request(isolated=True), "sess-42", cwd=tmp_path) is None
    assert build_cli_resume_command(_make_request(message="hi"), "sess-42", cwd=tmp_path) is None
    assert build_cli_resume_command(_make_request(mode="serve"), "sess-42", cwd=tmp_path) is None
    assert (
        build_cli_resume_command(
            _make_request(model_base_url="http://localhost:8080"),
            "sess-42",
            cwd=tmp_path,
        )
        is None
    )
