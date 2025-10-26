import asyncio
import logging
import platform
import signal
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

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


@contextmanager
def _no_progress():
    yield


def _setup_runtime(
    monkeypatch: pytest.MonkeyPatch, runtime_info: Dict[str, str]
) -> Tuple[ShellRuntime, DummyProcess, Dict[str, Any]]:
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(activation_reason="test", logger=logger)
    runtime.runtime_info = lambda: runtime_info  # type: ignore[assignment]
    runtime.working_directory = lambda: Path(".")  # type: ignore[assignment]

    dummy_process = DummyProcess()
    captured: Dict[str, Any] = {}

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


@pytest.mark.asyncio
async def test_execute_simple_command() -> None:
    """Test that shell runtime can execute a simple cross-platform command."""
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(activation_reason="test", logger=logger, timeout_seconds=10)

    # Use 'echo' which works on Windows, Linux, macOS
    result = await runtime.execute({"command": "echo hello"})

    assert result.isError is False
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
    assert "exit code" in result.content[0].text


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

    assert signal.CTRL_BREAK_EVENT in process.sent_signals
    assert process.terminated is True
    assert captured["exec_args"][0].endswith("pwsh.exe")
    assert result.isError is True
    assert "(timeout after 0s" in result.content[0].text
