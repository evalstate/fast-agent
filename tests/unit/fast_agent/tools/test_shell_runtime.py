import asyncio
import logging
import platform
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp.types import TextContent

from fast_agent.config import Settings, ShellSettings
from fast_agent.event_progress import ProgressAction
from fast_agent.tools.shell_runtime import ShellRuntime
from fast_agent.ui import console
from fast_agent.ui.progress_display import progress_display


class DummyStream:
    def __init__(self, lines: list[bytes] | None = None) -> None:
        self._lines = list(lines or [])

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class DummyProcess:
    def __init__(self) -> None:
        self.stdout = DummyStream()
        self.stderr = DummyStream()
        self.returncode: int | None = None
        self.pid = 1234
        self.sent_signals: list[Any] = []
        self.terminated = False
        self.killed = False

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def send_signal(self, sig: Any) -> None:
        self.sent_signals.append(sig)

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 1 if self.returncode is None else self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = 1 if self.returncode is None else self.returncode


class RecordingFastLogger:
    def __init__(self) -> None:
        self.info_calls: list[tuple[str, dict[str, Any]]] = []
        self.debug_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.error_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def info(self, message: str, **kwargs: Any) -> None:
        self.info_calls.append((message, kwargs))

    def debug(self, *args: Any, **kwargs: Any) -> None:
        self.debug_calls.append((args, kwargs))

    def error(self, *args: Any, **kwargs: Any) -> None:
        self.error_calls.append((args, kwargs))


@contextmanager
def _no_progress():
    yield


def _setup_runtime(
    monkeypatch: pytest.MonkeyPatch, runtime_info: dict[str, str]
) -> tuple[ShellRuntime, DummyProcess, dict[str, Any]]:
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(activation_reason="test", logger=logger)
    runtime.runtime_info = lambda: runtime_info  # type: ignore[assignment]
    runtime.working_directory = lambda: Path(".")  # type: ignore[assignment]

    dummy_process = DummyProcess()
    captured: dict[str, Any] = {}

    async def fake_exec(*args, **kwargs):
        captured["exec_args"] = args
        captured["exec_kwargs"] = kwargs
        return dummy_process

    async def fail_shell(*args, **kwargs):
        pytest.fail("create_subprocess_shell should not be used for this test")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", fail_shell)
    monkeypatch.setattr(console.console, "print", lambda *a, **k: None)
    monkeypatch.setattr(progress_display, "paused", _no_progress)
    if not hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        monkeypatch.setattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0x00000200,
            raising=False,
        )
    if not hasattr(signal, "CTRL_BREAK_EVENT"):
        monkeypatch.setattr(signal, "CTRL_BREAK_EVENT", object(), raising=False)

    return runtime, dummy_process, captured


def _extract_progress_payloads(logger: RecordingFastLogger) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for _, kwargs in logger.info_calls:
        payload = kwargs.get("data")
        if not isinstance(payload, dict):
            continue
        action = payload.get("progress_action")
        if action in {ProgressAction.CALLING_TOOL, ProgressAction.TOOL_PROGRESS}:
            payloads.append(payload)
    return payloads


@pytest.mark.asyncio
async def test_execute_simple_command() -> None:
    """Test that shell runtime can execute a simple cross-platform command."""
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(activation_reason="test", logger=logger, timeout_seconds=10)

    # Use 'echo' which works on Windows, Linux, macOS
    result = await runtime.execute({"command": "echo hello"})

    assert result.isError is False
    assert result.content is not None
    assert result.content[0].type == "text"
    assert isinstance(result.content[0], TextContent)
    assert "hello" in result.content[0].text
    assert "exit code" in result.content[0].text


@pytest.mark.asyncio
async def test_execute_command_with_exit_code() -> None:
    """Test that shell runtime captures non-zero exit codes."""
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(activation_reason="test", logger=logger, timeout_seconds=10)

    # Use different exit commands based on platform
    if platform.system() == "Windows":
        # Windows cmd.exe
        result = await runtime.execute({"command": "exit 1"})
    else:
        # Unix shells
        result = await runtime.execute({"command": "false"})

    assert result.isError is True
    assert result.content is not None
    assert result.content[0].type == "text"
    assert isinstance(result.content[0], TextContent)
    assert "exit code" in result.content[0].text


@pytest.mark.asyncio
async def test_execute_reports_informative_truncation_summary() -> None:
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        timeout_seconds=10,
        output_byte_limit=120,
    )

    long_echo = "echo " + ("x" * 2000)
    result = await runtime.execute({"command": long_echo})

    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert "[Output truncated: retained" in text
    assert "Increase shell_execution.output_byte_limit to retain more." in text
    assert "omitted" in text


@pytest.mark.asyncio
async def test_execute_with_missing_working_directory_returns_actionable_error(
    tmp_path: Path,
) -> None:
    logger = logging.getLogger("shell-runtime-test")
    missing_dir = tmp_path / "missing-dir"
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        timeout_seconds=10,
        working_directory=missing_dir,
    )

    result = await runtime.execute({"command": "pwd"})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "Shell working directory does not exist" in result.content[0].text
    assert str(missing_dir.resolve()) in result.content[0].text


@pytest.mark.asyncio
async def test_execute_with_file_working_directory_returns_actionable_error(
    tmp_path: Path,
) -> None:
    logger = logging.getLogger("shell-runtime-test")
    file_path = tmp_path / "not-a-directory.txt"
    file_path.write_text("x", encoding="utf-8")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        timeout_seconds=10,
        working_directory=file_path,
    )

    result = await runtime.execute({"command": "pwd"})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "Shell working directory is not a directory" in result.content[0].text
    assert str(file_path.resolve()) in result.content[0].text


@pytest.mark.asyncio
async def test_timeout_sends_ctrl_break_for_pwsh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    runtime, process, captured = _setup_runtime(
        monkeypatch, {"name": "pwsh", "path": r"C:\Program Files\PowerShell\7\pwsh.exe"}
    )
    runtime._timeout_seconds = 0
    runtime._warning_interval_seconds = 0

    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    result = await runtime.execute({"command": "Start-Sleep -Seconds 5"})

    ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
    assert ctrl_break is not None
    assert ctrl_break in process.sent_signals
    assert process.terminated is True
    assert captured["exec_args"][0].endswith("pwsh.exe")
    assert result.isError is True
    assert result.content is not None
    assert result.content[0].type == "text"
    assert isinstance(result.content[0], TextContent)
    assert "(timeout after 0s" in result.content[0].text


@pytest.mark.asyncio
async def test_execute_no_output_shows_compact_exit_banner_detail() -> None:
    """No-output commands should include compact '(no output)' + id detail."""
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(activation_reason="test", logger=logger, timeout_seconds=10)

    if platform.system() == "Windows":
        command = "exit 0"
    else:
        command = "true"

    with console.console.capture() as capture:
        result = await runtime.execute(
            {"command": command},
            tool_use_id="call_abcdef0123456789",
            show_tool_call_id=True,
        )

    assert result.isError is False
    rendered = capture.get()
    assert "exit code 0" in rendered
    assert "(no output)" in rendered
    assert "id: call_" in rendered


@pytest.mark.asyncio
async def test_execute_live_display_truncates_with_head_and_tail_windows() -> None:
    """Live shell display should show head + marker + tail when line-limited."""
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        timeout_seconds=10,
        config=Settings(shell_execution=ShellSettings(output_display_lines=6, show_bash=True)),
    )

    command = (
        f'"{sys.executable}" -c "for i in range(1, 11): '
        "print('out-{0:02d}'.format(i))\""
    )

    with console.console.capture() as capture:
        result = await runtime.execute({"command": command})

    assert result.isError is False
    rendered = capture.get()
    assert "out-01" in rendered
    assert "out-02" in rendered
    assert "out-03" in rendered
    assert "out-08" in rendered
    assert "out-09" in rendered
    assert "out-10" in rendered
    assert "out-04" not in rendered
    assert "out-05" not in rendered
    assert "out-06" not in rendered
    assert "out-07" not in rendered
    assert "..." in rendered


@pytest.mark.asyncio
async def test_execute_deferred_display_suppresses_live_console_output() -> None:
    """When display is deferred, shell runtime should not stream output directly."""
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(activation_reason="test", logger=logger, timeout_seconds=10)

    with console.console.capture() as capture:
        result = await runtime.execute(
            {"command": "echo hello"},
            tool_use_id="call_abcdef0123456789",
            show_tool_call_id=True,
            defer_display_to_tool_result=True,
        )

    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "hello" in result.content[0].text
    assert "process exit code was 0" in result.content[0].text
    assert getattr(result, "_suppress_display", True) is False
    rendered = capture.get()
    assert "hello" not in rendered
    assert "exit code" not in rendered


@pytest.mark.asyncio
async def test_execute_emits_shell_lifecycle_progress_events(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = RecordingFastLogger()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        timeout_seconds=10,
        agent_name="assistant",
    )
    runtime.runtime_info = lambda: {"name": "bash", "path": "/bin/bash"}  # type: ignore[assignment]
    runtime.working_directory = lambda: Path(".")  # type: ignore[assignment]

    process = DummyProcess()
    process.stdout = DummyStream([b"hello\n"])
    process.stderr = DummyStream([])

    async def fake_shell(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_shell)
    monkeypatch.setattr(console.console, "print", lambda *a, **k: None)
    monkeypatch.setattr(progress_display, "paused", _no_progress)

    result = await runtime.execute({"command": "echo hello"}, tool_use_id="call-123")
    assert result.isError is False

    progress_payloads = _extract_progress_payloads(logger)
    assert len(progress_payloads) == 2

    start_payload = progress_payloads[0]
    assert start_payload == {
        "progress_action": ProgressAction.CALLING_TOOL,
        "tool_name": "execute",
        "server_name": "local",
        "agent_name": "assistant",
        "tool_use_id": "call-123",
        "tool_call_id": "call-123",
        "tool_event": "start",
    }

    end_payload = progress_payloads[1]
    assert end_payload["progress_action"] == ProgressAction.TOOL_PROGRESS
    assert end_payload["tool_name"] == "execute"
    assert end_payload["server_name"] == "local"
    assert end_payload["agent_name"] == "assistant"
    assert end_payload["tool_use_id"] == "call-123"
    assert end_payload["tool_call_id"] == "call-123"
    assert end_payload["progress"] == 1.0
    assert end_payload["total"] == 1.0
    assert end_payload["details"] == "completed (exit 0)"


@pytest.mark.asyncio
async def test_execute_emits_terminal_failed_progress_when_subprocess_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = RecordingFastLogger()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        timeout_seconds=10,
        agent_name="assistant",
    )

    async def fail_shell(*args, **kwargs):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fail_shell)
    monkeypatch.setattr(progress_display, "paused", _no_progress)

    result = await runtime.execute({"command": "echo hello"}, tool_use_id="call-456")

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "Command failed to start" in result.content[0].text

    progress_payloads = _extract_progress_payloads(logger)
    assert len(progress_payloads) == 2
    assert progress_payloads[0]["progress_action"] == ProgressAction.CALLING_TOOL
    assert progress_payloads[0]["tool_event"] == "start"
    assert progress_payloads[1]["progress_action"] == ProgressAction.TOOL_PROGRESS
    assert progress_payloads[1]["details"] == "failed: spawn failed"
