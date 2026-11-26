from __future__ import annotations

import base64
import hashlib
import io
import re
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast

import pytest

from fast_agent.cli.runtime.program_resume import (
    _resolve_cli_executable,
    build_cli_resume_command,
    capture_cli_invocation,
)
from fast_agent.cli.runtime.request_builders import build_command_run_request
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

    from fast_agent.cli.runtime.run_request import AgentRunRequest


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
    model: str | None = "sonnet",
    message: str | None = None,
    no_home: bool = False,
    shell_enabled: bool = False,
    mode: Literal["interactive", "serve"] = "interactive",
    workspace: Path | None = None,
    model_base_url: str | None = None,
    isolated: bool = False,
) -> AgentRunRequest:
    return build_command_run_request(
        name=name,
        instruction_option=None,
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        agent_cards=None,
        card_tools=None,
        model=model,
        message=message,
        prompt_file=None,
        result_file=None,
        resume=None,
        npx=None,
        uvx=None,
        stdio=None,
        target_agent_name=None,
        skills_directory=None,
        home=None,
        shell_enabled=shell_enabled,
        mode=mode,
        no_home=no_home,
        isolated=isolated,
        workspace=workspace,
        model_base_url=model_base_url,
    )


@pytest.fixture
def resume_terminal(monkeypatch: pytest.MonkeyPatch) -> Iterator[TtyStream]:
    stream = TtyStream()
    monkeypatch.setattr(
        program_resume,
        "sys",
        SimpleNamespace(stdout=stream, platform=sys.platform),
    )
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


def test_plain_build_command_run_request_formats_valid_osc_7577(tmp_path: Path) -> None:
    """Smoke test with real build_command_run_request (multiline DEFAULT_AGENT_INSTRUCTION)."""
    request = _make_request(workspace=tmp_path)
    cmd = build_cli_resume_command(
        request,
        "2611260946-abc123",
        raw_argv=("fast-agent", "go"),
        cwd=tmp_path,
    )
    assert cmd is not None
    seq = format_program_resume(cmd, token="t" * 32, pid=1234, now=1795543200)
    assert seq is not None
    rec = _parse_7577(seq)[0]
    decoded = _decode_argv(rec["argv"])
    assert decoded[-3:] == ["go", "--resume", "2611260946-abc123"]
    assert "--instruction" not in decoded
    assert "--agent" not in decoded


def test_replay_argv_preserves_flags_strips_resume_and_single_model_and_skips_unsafe(
    tmp_path: Path,
) -> None:
    req = _make_request(workspace=tmp_path)

    # Auto-routed invocation without "go": preserves --card/--agent/--url/--shell,
    # strips old --resume and single --model, inserts "go", appends new --resume.
    cmd = build_cli_resume_command(
        req,
        "sess-42",
        raw_argv=(
            "fast-agent",
            "--card",
            "cards/reviewer.md",
            "--agent",
            "reviewer",
            "--url",
            "https://example.com/mcp",
            "--model",
            "sonnet",
            "--resume",
            "old-sess",
            "--shell",
        ),
        cwd=tmp_path,
    )
    assert cmd is not None
    assert cmd.cwd == str(tmp_path.resolve())
    assert cmd.argv[1:] == (
        "go",
        "--card",
        "cards/reviewer.md",
        "--agent",
        "reviewer",
        "--url",
        "https://example.com/mcp",
        "--shell",
        "--resume",
        "sess-42",
    )

    # Multi-model fan-out is preserved; bare -r is stripped.
    multi = build_cli_resume_command(
        req,
        "sess-42",
        raw_argv=("fast-agent", "go", "--model", "sonnet,opus", "-r"),
        cwd=tmp_path,
    )
    assert multi is not None
    assert multi.argv[1:] == ("go", "--model", "sonnet,opus", "--resume", "sess-42")

    # python -m fallback targets fast_agent.cli (fast_agent/__main__.py does not exist).
    exe = _resolve_cli_executable("-m")
    assert exe[1:] == ("-m", "fast_agent.cli")

    # Unsafe / non-resumable invocations return None.
    for bad_argv in (
        ("fast-agent", "local"),
        ("fast-agent", "go", "--local"),
        ("fast-agent", "go", "--model-url", "http://localhost:8080"),
        ("fast-agent", "go", "--model-auth", "secret"),
        ("fast-agent", "go", "--auth", "bearer-secret"),
    ):
        assert build_cli_resume_command(req, "sess-42", raw_argv=bad_argv, cwd=tmp_path) is None

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


def test_capture_strips_token_and_lifecycle_reports_refreshes_and_preserves_on_deactivate(
    monkeypatch: pytest.MonkeyPatch,
    resume_terminal: TtyStream,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(PROGRAM_RESUME_TOKEN_ENV, "secret-token-0123456789abcdef")
    program_resume.capture_program_resume_token()
    assert PROGRAM_RESUME_TOKEN_ENV not in program_resume.os.environ

    capture_cli_invocation(("fast-agent", "go", "--shell", "--model", "opus"), cwd=tmp_path)
    request = _make_request(shell_enabled=True, workspace=tmp_path)
    program_resume.configure_program_resume(lambda sid: build_cli_resume_command(request, sid))

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
    assert decoded[1:] == ["go", "--shell", "--resume", "2611260946-abc123"]
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
    program_resume.capture_program_resume_token()
    assert PROGRAM_RESUME_TOKEN_ENV not in program_resume.os.environ

    program_resume.configure_program_resume(
        lambda sid: (
            ProgramResumeCommand(argv=("fast-agent", "go", "--resume", sid)) if sid else None
        )
    )
    program_resume.report_program_resume_session("sess-1")
    assert resume_terminal.getvalue() == ""
