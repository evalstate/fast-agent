import asyncio
import logging
import os
import platform
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest
from mcp.types import TextContent

import fast_agent.tools.local_shell_executor as local_shell_executor
import fast_agent.tools.shell_runtime as shell_runtime_module
from fast_agent.config import Settings, ShellSettings
from fast_agent.constants import (
    DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
    MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
)
from fast_agent.event_progress import ProgressAction
from fast_agent.tools.execution_environment import (
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionOptions,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellRuntimeInfo,
)
from fast_agent.tools.local_shell_executor import LocalShellExecutor
from fast_agent.tools.shell_runtime import ShellRuntime
from fast_agent.ui import console
from fast_agent.ui.display_suppression import suppress_interactive_display
from fast_agent.ui.progress_display import progress_display
from fast_agent.ui.shell_output_truncation import SHELL_OUTPUT_TRUNCATION_MARKER


class DummyStream:
    def __init__(self, lines: list[bytes] | None = None) -> None:
        self._lines = list(lines or [])

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""

    async def read(self, n: int = -1) -> bytes:
        if not self._lines:
            return b""
        if n < 0:
            data = b"".join(self._lines)
            self._lines.clear()
            return data

        chunks: list[bytes] = []
        remaining = n
        while self._lines and remaining > 0:
            current = self._lines[0]
            if len(current) <= remaining:
                chunks.append(self._lines.pop(0))
                remaining -= len(current)
                continue
            chunks.append(current[:remaining])
            self._lines[0] = current[remaining:]
            remaining = 0
        return b"".join(chunks)


class DummyProcess:
    def __init__(self) -> None:
        self.stdout = DummyStream()
        self.stderr = DummyStream()
        self.returncode: int | None = None
        self.pid = 1234
        self.sent_signals: list[Any] = []
        self.terminated = False

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


class _TestLocalShellExecutor(LocalShellExecutor):
    def __init__(
        self,
        *,
        runtime_info: Mapping[str, str | None],
        **kwargs: Any,
    ) -> None:
        self._test_runtime_info = dict(runtime_info)
        super().__init__(**kwargs)

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(
            name=self._test_runtime_info.get("name") or "shell",
            path=self._test_runtime_info.get("path"),
        )


class _RecordingShellEnvironment:
    def __init__(self, cwd: str = "/workspace") -> None:
        self._cwd = cwd
        self.requests: list[ShellExecutionRequest] = []
        self.resolved_paths: list[str] = []

    async def open(self) -> None:
        return None

    @property
    def cwd(self) -> str:
        return self._cwd

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(name="bash", kind="docker", provider="test")

    def resolve_path(self, path: str) -> str:
        self.resolved_paths.append(path)
        return path if path.startswith("/") else f"{self._cwd}/{path}"

    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        del callbacks
        self.requests.append(request)
        return ShellExecution(
            result=ShellExecutionResult(stdout="", stderr="", exit_code=0),
            options=ShellExecutionOptions(),
        )

    async def close(self) -> None:
        return None


class _DirectShellEnvironment:
    def __init__(
        self,
        *,
        stream_output: bool,
        timed_out: bool = False,
        stdout: str | None = None,
    ) -> None:
        self.stream_output = stream_output
        self.timed_out = timed_out
        self.stdout = stdout
        self.requests: list[ShellExecutionRequest] = []

    async def open(self) -> None:
        return None

    @property
    def cwd(self) -> str:
        return "/workspace"

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(name="bash", kind="remote", provider="test")

    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        self.requests.append(request)
        stdout = self.stdout or ("streamed\n" if self.stream_output else "buffered\n")
        if self.stream_output and callbacks is not None:
            for line in stdout.splitlines(keepends=True):
                await callbacks.on_stdout(line)
            if self.timed_out:
                await callbacks.on_timeout()
            return ShellExecution(
                result=ShellExecutionResult(stdout=stdout, stderr="", exit_code=0),
                options=ShellExecutionOptions(timeout_seconds=request.timeout),
                timed_out=self.timed_out,
            )
        return ShellExecution(
            result=ShellExecutionResult(stdout=stdout, stderr="", exit_code=0),
            options=ShellExecutionOptions(timeout_seconds=request.timeout),
            timed_out=self.timed_out,
        )

    async def close(self) -> None:
        return None


class _ManagedShellEnvironment:
    def __init__(self) -> None:
        self._cwd = "/workspace"
        self.requests: list[ShellExecutionRequest] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.stdout = "managed complete\n"
        self.exit_code = 0

    async def open(self) -> None:
        return None

    @property
    def cwd(self) -> str:
        return self._cwd

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(name="bash", kind="remote", provider="managed-test")

    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        self.requests.append(request)
        if callbacks is not None:
            await callbacks.on_started(4321)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = request.terminate_on_cancel
            raise
        if callbacks is not None and self.stdout:
            await callbacks.on_stdout(self.stdout)
        return ShellExecution(
            result=ShellExecutionResult(
                stdout=self.stdout if request.retain_output else "",
                stderr="",
                exit_code=self.exit_code,
            ),
            options=ShellExecutionOptions(timeout_seconds=request.timeout),
        )

    async def close(self) -> None:
        return None


class _ActiveManagedShellEnvironment(_ManagedShellEnvironment):
    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        self.requests.append(request)
        if callbacks is not None:
            await callbacks.on_started(4321)
        self.started.set()
        try:
            while not self.release.is_set():
                if callbacks is not None:
                    await callbacks.on_stdout("still working\n")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            self.cancelled = request.terminate_on_cancel
            raise
        return ShellExecution(
            result=ShellExecutionResult(stdout="", stderr="", exit_code=0),
            options=ShellExecutionOptions(timeout_seconds=request.timeout),
        )


class _FailedCancellationShellEnvironment(_ManagedShellEnvironment):
    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        self.requests.append(request)
        if callbacks is not None:
            await callbacks.on_started(4321)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError as exc:
            raise RuntimeError("remote termination failed") from exc
        raise AssertionError("unreachable")


class _CancellableLocalShellExecutor(LocalShellExecutor):
    def __init__(self, *, logger: logging.Logger) -> None:
        super().__init__(logger=logger)
        self.processes: list[DummyProcess] = []
        self.terminated_pids: list[int] = []
        self.waiting_count = 0
        self.waiting = asyncio.Event()

    async def _start_shell_process(
        self,
        command: str,
        plan: Any,
    ) -> asyncio.subprocess.Process:
        process = DummyProcess()
        process.pid = 10_000 + len(self.processes)
        self.processes.append(process)
        return cast("asyncio.subprocess.Process", process)

    async def _wait_for_process_exit(self, process: Any) -> int:
        self.waiting_count += 1
        self.waiting.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def _terminate_cancelled_process(
        self,
        process: Any,
        *,
        is_windows: bool,
    ) -> None:
        self.terminated_pids.append(process.pid)
        process.returncode = -signal.SIGTERM


@contextmanager
def _no_progress():
    yield


def _setup_runtime(
    monkeypatch: pytest.MonkeyPatch,
    runtime_info: dict[str, str],
    **runtime_kwargs: Any,
) -> tuple[ShellRuntime, DummyProcess, dict[str, Any]]:
    logger = logging.getLogger("shell-runtime-test")
    shell_environment = _TestLocalShellExecutor(
        logger=logger,
        runtime_info=runtime_info,
        timeout_seconds=runtime_kwargs.get("timeout_seconds", 90),
        warning_interval_seconds=runtime_kwargs.get("warning_interval_seconds", 30),
        config=runtime_kwargs.get("config"),
    )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        shell_environment=shell_environment,
        **runtime_kwargs,
    )

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


def test_shell_output_byte_limit_coerces_invalid_values() -> None:
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(activation_reason="test", logger=logger)

    for value in (None, 0, -1, True):
        runtime.set_output_byte_limit(value)  # type: ignore[arg-type]
        assert runtime.output_byte_limit == DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT

    runtime.set_output_byte_limit(MAX_TERMINAL_OUTPUT_BYTE_LIMIT + 1)
    assert runtime.output_byte_limit == MAX_TERMINAL_OUTPUT_BYTE_LIMIT

    runtime.set_output_byte_limit(1024)
    assert runtime.output_byte_limit == 1024


def test_shell_runtime_reads_typed_shell_settings() -> None:
    settings = Settings(
        shell_execution=ShellSettings(
            output_display_lines=7,
            show_bash=False,
            prefer_local_shell=True,
        )
    )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        config=settings,
    )

    assert runtime._output_display_lines == 7
    assert runtime._show_bash_output is False
    assert runtime.prefer_local_shell is True


def test_execute_tool_schema_declares_per_call_options() -> None:
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
    )

    assert runtime.tool is not None
    assert runtime.tool.description is not None
    assert "keeps running and returns a process ID" in runtime.tool.description
    assert "Do not append '&'" in runtime.tool.description
    assert "lifecycle='persistent'" in runtime.tool.description
    assert set(runtime.tool.inputSchema["properties"]) == {
        "command",
        "cwd",
        "background",
        "lifecycle",
        "yield_after_idle_sec",
        "output_byte_limit",
    }
    lifecycle_schema = runtime.tool.inputSchema["properties"]["lifecycle"]
    assert lifecycle_schema["enum"] == ["session", "persistent"]
    assert lifecycle_schema["default"] == "session"
    assert runtime.tool.inputSchema["required"] == ["command"]
    assert runtime.tool.inputSchema["additionalProperties"] is False
    assert {tool.name for tool in runtime.tools} == {
        "execute",
        "poll_process",
        "terminate_process",
    }


def test_shell_metadata_uses_effective_per_call_options() -> None:
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        working_directory=Path("/default"),
        timeout_seconds=90,
        output_byte_limit=1000,
    )

    metadata = runtime.metadata(
        {
            "command": "pwd",
            "cwd": "/per-call",
            "yield_after_idle_sec": 15,
            "output_byte_limit": 80,
        }
    )

    assert metadata["working_dir"] == "/per-call"
    assert metadata["idle_yield_seconds"] == 15
    assert metadata["foreground_yield_seconds"] == 30
    assert metadata["output_byte_limit"] == 80
    assert metadata["lifecycle"] == "session"


@pytest.mark.asyncio
async def test_shell_environment_exports_runtime_home(tmp_path: Path) -> None:
    settings = Settings()
    settings._fast_agent_home = str(tmp_path / ".fast-agent")
    executor = LocalShellExecutor(logger=logging.getLogger(__name__), config=settings)

    result = await executor.execute_shell(
        f"{sys.executable} -c \"import os; print(os.environ['FAST_AGENT_HOME'])\"",
        cwd=tmp_path,
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == str((tmp_path / ".fast-agent").resolve())


@pytest.mark.asyncio
async def test_shell_environment_strips_runtime_home_in_no_home(tmp_path: Path) -> None:
    settings = Settings()
    settings._fast_agent_home = str(tmp_path / ".fast-agent")
    settings._fast_agent_no_home = True
    executor = LocalShellExecutor(logger=logging.getLogger(__name__), config=settings)

    result = await executor.execute_shell(
        (f"{sys.executable} -c \"import os; print(os.environ.get('FAST_AGENT_HOME', 'missing'))\""),
        cwd=tmp_path,
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "missing"


@pytest.mark.asyncio
async def test_shell_environment_terminates_process_when_cancelled() -> None:
    executor = _CancellableLocalShellExecutor(logger=logging.getLogger(__name__))

    task = asyncio.create_task(executor.execute(ShellExecutionRequest(command="long-running")))
    await executor.waiting.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert executor.terminated_pids == [10_000]


@pytest.mark.asyncio
async def test_shell_environment_terminates_parallel_processes_when_cancelled() -> None:
    executor = _CancellableLocalShellExecutor(logger=logging.getLogger(__name__))

    tasks = [
        asyncio.create_task(executor.execute(ShellExecutionRequest(command="long-running-a"))),
        asyncio.create_task(executor.execute(ShellExecutionRequest(command="long-running-b"))),
    ]
    while executor.waiting_count < len(tasks):
        await executor.waiting.wait()
        executor.waiting.clear()

    for task in tasks:
        task.cancel()

    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert executor.terminated_pids == [10_000, 10_001]


def _terminate_pid(pid_path: Path) -> None:
    if not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        except OSError:
            return
        time.sleep(0.1)


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
async def test_execute_shell_returns_structured_output(tmp_path: Path) -> None:
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        timeout_seconds=10,
        config=Settings(shell_execution=ShellSettings(show_bash=False)),
    )

    script = (
        "import os, pathlib, sys; "
        "print(pathlib.Path.cwd().name); "
        "print(os.environ['FAST_AGENT_TEST_ENV']); "
        "print('problem', file=sys.stderr)"
    )
    result = await runtime.execute_shell(
        f"{sys.executable} -c {script!r}",
        cwd=tmp_path,
        env={"FAST_AGENT_TEST_ENV": "present"},
    )

    assert isinstance(result, ShellExecutionResult)
    assert result.exit_code == 0
    assert result.stdout.splitlines() == [tmp_path.name, "present"]
    assert result.stderr == "problem\n"


@pytest.mark.asyncio
async def test_set_working_directory_updates_execute_shell_cwd(tmp_path: Path) -> None:
    initial_dir = tmp_path / "initial"
    updated_dir = tmp_path / "updated"
    initial_dir.mkdir()
    updated_dir.mkdir()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        timeout_seconds=10,
        working_directory=initial_dir,
        config=Settings(shell_execution=ShellSettings(show_bash=False)),
    )

    runtime.set_working_directory(updated_dir)
    result = await runtime.execute_shell(
        f'{sys.executable} -c "import pathlib; print(pathlib.Path.cwd())"'
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == str(updated_dir)


@pytest.mark.asyncio
async def test_shared_shell_environment_preserves_runtime_working_directory() -> None:
    environment = _RecordingShellEnvironment(cwd="/workspace")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        working_directory=Path("/agent-cwd"),
        shell_environment=environment,
    )

    await runtime.execute_shell("pwd")

    assert environment.cwd == "/workspace"
    assert [(request.command, request.cwd) for request in environment.requests] == [("pwd", "/agent-cwd")]
    assert [request.timeout for request in environment.requests] == [90]


@pytest.mark.asyncio
async def test_execute_tool_uses_runtime_working_directory_with_shared_environment() -> None:
    environment = _RecordingShellEnvironment(cwd="/workspace")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        working_directory=Path("/agent-cwd"),
        shell_environment=environment,
    )

    result = await runtime.execute({"command": "pwd"})

    assert result.isError is False
    assert environment.cwd == "/workspace"
    assert [request.cwd for request in environment.requests] == ["/agent-cwd"]
    assert [request.timeout for request in environment.requests] == [None]
    assert [request.terminate_after_idle for request in environment.requests] == [False]
    assert [request.retain_output for request in environment.requests] == [False]


@pytest.mark.asyncio
async def test_execute_honors_per_call_cwd_and_yield_options() -> None:
    environment = _RecordingShellEnvironment(cwd="/workspace")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        working_directory=Path("/agent-cwd"),
        shell_environment=environment,
    )

    result = await runtime.execute(
        {
            "command": "pwd",
            "cwd": "/per-call-cwd",
            "yield_after_idle_sec": 20,
        }
    )

    assert result.isError is False
    assert [
        (request.cwd, request.timeout, request.terminate_after_idle)
        for request in environment.requests
    ] == [
        ("/per-call-cwd", None, False)
    ]


@pytest.mark.asyncio
async def test_execute_resolves_relative_per_call_cwd_against_active_working_directory() -> None:
    environment = _RecordingShellEnvironment(cwd="/workspace")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        working_directory=Path("/agent-cwd"),
        shell_environment=environment,
    )

    result = await runtime.execute(
        {
            "command": "pwd",
            "cwd": "subdir",
        }
    )

    assert result.isError is False
    assert environment.resolved_paths[-1] == "/agent-cwd/subdir"
    assert environment.requests[0].cwd == "/agent-cwd/subdir"


@pytest.mark.asyncio
async def test_execute_rejects_unknown_arguments_without_running() -> None:
    environment = _RecordingShellEnvironment()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        shell_environment=environment,
    )

    timeout_result = await runtime.execute({"command": "touch /tmp/nope", "timeout": 120000})
    unknown_result = await runtime.execute({"command": "touch /tmp/nope", "stream": True})

    assert timeout_result.isError is True
    assert unknown_result.isError is True
    assert environment.requests == []
    assert timeout_result.content is not None
    assert isinstance(timeout_result.content[0], TextContent)
    assert "use 'yield_after_idle_sec'" in timeout_result.content[0].text


@pytest.mark.asyncio
async def test_execute_rejects_idle_yield_over_thirty_seconds() -> None:
    environment = _RecordingShellEnvironment()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        shell_environment=environment,
    )

    result = await runtime.execute(
        {"command": "sleep 3600", "yield_after_idle_sec": 31}
    )

    assert result.isError is True
    assert environment.requests == []
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "'yield_after_idle_sec' argument must be at most 30" in result.content[0].text


@pytest.mark.asyncio
async def test_silent_command_yields_alive_then_poll_reports_completion() -> None:
    environment = _ManagedShellEnvironment()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        shell_environment=environment,
        idle_yield_seconds=0.05,
        foreground_yield_seconds=0.5,
    )

    result = await runtime.execute({"command": "slow-build"})

    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "Process is still running" in result.content[0].text
    assert "process_id: process-1" in result.content[0].text
    assert environment.requests[0].terminate_after_idle is False
    assert environment.requests[0].retain_output is False

    running_poll = await runtime.poll_process({"process_id": "process-1"})
    assert running_poll.content is not None
    assert isinstance(running_poll.content[0], TextContent)
    assert "Process is still running." in running_poll.content[0].text
    assert "because it is still running" not in running_poll.content[0].text

    environment.release.set()
    poll_result = await runtime.poll_process(
        {"process_id": "process-1", "wait_sec": 1}
    )

    assert poll_result.isError is False
    assert poll_result.content is not None
    assert isinstance(poll_result.content[0], TextContent)
    assert "managed complete" in poll_result.content[0].text
    assert "process exit code was 0" in poll_result.content[0].text
    assert getattr(poll_result, "output_line_count", None) == 1


@pytest.mark.asyncio
async def test_continuous_output_still_yields_at_foreground_ceiling() -> None:
    environment = _ActiveManagedShellEnvironment()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        shell_environment=environment,
        idle_yield_seconds=0.2,
        foreground_yield_seconds=0.05,
    )

    result = await runtime.execute({"command": "chatty-build"})

    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert "still working" in text
    assert "reached the foreground yield threshold" in text
    assert "process_id: process-1" in text
    assert environment.cancelled is False

    await runtime.terminate_process({"process_id": "process-1"})
    assert environment.cancelled is True


@pytest.mark.asyncio
async def test_background_command_returns_handle_and_terminate_cancels_job() -> None:
    environment = _ManagedShellEnvironment()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        shell_environment=environment,
    )

    with console.console.capture() as capture:
        result = await runtime.execute({"command": "server", "background": True})
    rendered = capture.get()
    assert "process-1" in rendered
    assert "running • background" in rendered
    assert "pid 4321" in rendered
    assert runtime.active_process_count == 1
    snapshots = await runtime.process_snapshots()
    assert len(snapshots) == 1
    assert snapshots[0].status == "running"
    assert snapshots[0].os_process_id == 4321

    terminate_result = await runtime.terminate_process({"process_id": "process-1"})

    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "os_pid: 4321" in result.content[0].text
    assert terminate_result.isError is False
    assert environment.cancelled is True
    assert terminate_result.content is not None
    assert isinstance(terminate_result.content[0], TextContent)
    assert "outcome: terminated" in terminate_result.content[0].text
    assert runtime.active_process_count == 0
    snapshots = await runtime.process_snapshots()
    assert snapshots[0].status == "terminated"


@pytest.mark.asyncio
async def test_completed_process_snapshot_elapsed_time_stops_advancing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _ManagedShellEnvironment()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        shell_environment=environment,
    )

    await runtime.execute({"command": "quick task", "background": True})
    environment.release.set()
    for _ in range(100):
        if runtime.active_process_count == 0:
            break
        await asyncio.sleep(0)

    first = (await runtime.process_snapshots())[0]
    assert first.status == "completed"
    monkeypatch.setattr(
        shell_runtime_module.time,
        "monotonic",
        lambda: first.elapsed_seconds + 10_000,
    )

    later = (await runtime.process_snapshots())[0]

    assert later.elapsed_seconds == first.elapsed_seconds


@pytest.mark.asyncio
async def test_background_deferred_display_exposes_ordered_result() -> None:
    environment = _ManagedShellEnvironment()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        shell_environment=environment,
    )

    result = await runtime.execute(
        {"command": "server", "background": True},
        defer_display_to_tool_result=True,
    )

    assert getattr(result, "_suppress_display", True) is False
    await runtime.close()


@pytest.mark.asyncio
async def test_terminate_process_reports_environment_cancellation_failure() -> None:
    environment = _FailedCancellationShellEnvironment()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        shell_environment=environment,
    )
    await runtime.execute({"command": "server", "background": True})

    result = await runtime.terminate_process({"process_id": "process-1"})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "outcome: termination_failed" in result.content[0].text
    assert "remote termination failed" in result.content[0].text
    assert getattr(result, "process_status", None) == "termination_failed"
    await runtime.close()


@pytest.mark.asyncio
async def test_lifecycle_tool_calls_emit_correlated_progress() -> None:
    environment = _ManagedShellEnvironment()
    logger = RecordingFastLogger()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        shell_environment=environment,
        agent_name="assistant",
    )
    await runtime.execute({"command": "server", "background": True})
    logger.info_calls.clear()

    result = await runtime.call_tool(
        "poll_process",
        {"process_id": "process-1", "wait_sec": 0},
        tool_use_id="call-poll",
    )

    assert result.isError is False
    progress_payloads = _extract_progress_payloads(logger)
    assert [payload["tool_name"] for payload in progress_payloads] == [
        "poll_process",
        "poll_process",
    ]
    assert progress_payloads[0]["tool_event"] == "start"
    assert progress_payloads[1]["tool_terminal"] is True
    assert progress_payloads[1]["details"] == "process-1: running"
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_close_terminates_all_managed_processes() -> None:
    environment = _ManagedShellEnvironment()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        shell_environment=environment,
    )

    await runtime.execute({"command": "server", "background": True})
    await runtime.close()

    assert environment.cancelled is True


@pytest.mark.asyncio
async def test_runtime_close_detaches_persistent_process() -> None:
    environment = _ManagedShellEnvironment()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        shell_environment=environment,
    )

    await runtime.execute(
        {
            "command": "server",
            "background": True,
            "lifecycle": "persistent",
        }
    )
    await runtime.close()

    assert environment.cancelled is False
    assert environment.requests[0].terminate_on_cancel is False


@pytest.mark.asyncio
async def test_terminate_process_overrides_persistent_lifecycle() -> None:
    environment = _ManagedShellEnvironment()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        shell_environment=environment,
    )

    await runtime.execute(
        {
            "command": "server",
            "background": True,
            "lifecycle": "persistent",
        }
    )
    await runtime.terminate_process({"process_id": "process-1"})

    assert environment.cancelled is True
    assert environment.requests[0].terminate_on_cancel is True


@pytest.mark.asyncio
@pytest.mark.skipif(platform.system() == "Windows", reason="Unix process persistence")
async def test_local_persistent_process_survives_runtime_close(tmp_path: Path) -> None:
    pid_path = tmp_path / "server.pid"
    script_path = tmp_path / "server.py"
    script_path.write_text(
        "import os, pathlib, time\n"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        working_directory=tmp_path,
    )

    try:
        await runtime.execute(
            {
                "command": f'"{sys.executable}" "{script_path}"',
                "background": True,
                "lifecycle": "persistent",
            }
        )
        for _ in range(100):
            if pid_path.exists():
                break
            await asyncio.sleep(0.01)
        await runtime.close()

        assert pid_path.exists()
        pid = int(pid_path.read_text(encoding="utf-8"))
        os.kill(pid, 0)
    finally:
        _terminate_pid(pid_path)


@pytest.mark.asyncio
async def test_execute_rejects_invalid_argument_payloads() -> None:
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
    )

    invalid_results = [
        await runtime.execute(None),
        await runtime.execute({}),
        await runtime.execute({"command": "   "}),
        await runtime.execute({"command": 123}),  # type: ignore[dict-item]
    ]

    assert [result.isError for result in invalid_results] == [True, True, True, True]
    messages: list[str] = []
    for result in invalid_results:
        assert result.content is not None
        assert isinstance(result.content[0], TextContent)
        messages.append(result.content[0].text)

    assert messages == [
        "Error: arguments must be a dict",
        "Error: 'command' argument is required and must be a string",
        "Error: 'command' argument is required and must be a string",
        "Error: 'command' argument is required and must be a string",
    ]

    invalid_lifecycle = await runtime.execute(
        {"command": "server", "background": True, "lifecycle": "forever"}
    )
    persistent_foreground = await runtime.execute(
        {"command": "server", "lifecycle": "persistent"}
    )
    assert invalid_lifecycle.content is not None
    assert persistent_foreground.content is not None
    assert isinstance(invalid_lifecycle.content[0], TextContent)
    assert isinstance(persistent_foreground.content[0], TextContent)
    assert invalid_lifecycle.content[0].text == (
        "Error: 'lifecycle' argument must be 'session' or 'persistent'"
    )
    assert persistent_foreground.content[0].text == (
        "Error: lifecycle='persistent' requires background=true"
    )


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
    assert "[Output truncated: showing first" in text
    assert "Increase shell_execution.output_byte_limit to retain more." in text
    assert "omitted" in text


@pytest.mark.asyncio
async def test_execute_truncated_result_includes_tail() -> None:
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        timeout_seconds=10,
        output_byte_limit=80,
        config=Settings(shell_execution=ShellSettings(show_bash=False)),
    )

    script = "for i in range(30): print(f'line-{i:02d}')"
    result = await runtime.execute({"command": f"{sys.executable} -c {script!r}"})

    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert "line-00" in text
    assert "line-29" in text
    assert "last" in text
    assert "omitted" in text
    assert "process exit code was 0" in text


@pytest.mark.asyncio
async def test_execute_honors_per_call_output_byte_limit() -> None:
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-runtime-test"),
        timeout_seconds=10,
        output_byte_limit=1000,
        config=Settings(shell_execution=ShellSettings(show_bash=False)),
    )

    result = await runtime.execute(
        {
            "command": f"{sys.executable} -c \"print('x' * 2000)\"",
            "output_byte_limit": 80,
        }
    )

    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert "[Output truncated:" in text
    assert "showing first 40 bytes and last 40 bytes" in text
    assert "process exit code was 0" in text


@pytest.mark.asyncio
async def test_execute_handles_overlong_output_lines_without_timeout() -> None:
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        timeout_seconds=5,
        output_byte_limit=256,
        config=Settings(shell_execution=ShellSettings(show_bash=False)),
    )

    command = f'"{sys.executable}" -c "print(\'x\' * 70000)"'
    result = await runtime.execute({"command": command})

    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert "timeout after" not in text
    assert "process exit code was 0" in text
    assert "[Output truncated: showing first" in text


@pytest.mark.asyncio
@pytest.mark.skipif(platform.system() == "Windows", reason="Unix inherited-pipe behavior")
async def test_execute_returns_when_descendant_keeps_pipe_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_shell_executor, "_IO_DRAIN_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(shell_runtime_module, "_IO_DRAIN_TIMEOUT_SECONDS", 0.1)
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        timeout_seconds=10,
        config=Settings(shell_execution=ShellSettings(show_bash=False)),
    )
    pid_path = tmp_path / "descendant.pid"
    script_path = tmp_path / "hold_pipe.py"
    script_path.write_text(
        "\n".join(
            [
                "import subprocess, sys",
                "child = subprocess.Popen(",
                "    [sys.executable, '-c', 'import time; time.sleep(30)'],",
                "    stdout=sys.stdout,",
                "    stderr=sys.stderr,",
                "    start_new_session=True,",
                ")",
                f"open({str(pid_path)!r}, 'w', encoding='utf-8').write(str(child.pid))",
                "print('parent exiting', flush=True)",
            ]
        ),
        encoding="utf-8",
    )

    started = time.monotonic()
    try:
        result = await runtime.execute({"command": f'"{sys.executable}" "{script_path}"'})
    finally:
        _terminate_pid(pid_path)
    elapsed = time.monotonic() - started

    assert elapsed < 1
    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert "parent exiting" in text
    assert "Output collection stopped after" in text
    assert "process exit code was 0" in text


@pytest.mark.asyncio
@pytest.mark.skipif(platform.system() == "Windows", reason="Unix inherited-pipe behavior")
async def test_direct_executor_timeout_with_inherited_pipe_does_not_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_shell_executor, "_IO_DRAIN_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(local_shell_executor, "_WATCHDOG_POLL_SECONDS", 0.05)
    monkeypatch.setattr(shell_runtime_module, "_IO_DRAIN_TIMEOUT_SECONDS", 0.1)

    async def terminate_unix_process(self: LocalShellExecutor, process: Any) -> None:
        os.killpg(process.pid, signal.SIGTERM)

    monkeypatch.setattr(
        LocalShellExecutor,
        "_terminate_unix_process",
        terminate_unix_process,
    )
    logger = logging.getLogger("shell-runtime-test")
    executor = LocalShellExecutor(
        logger=logger,
        timeout_seconds=0.1,
        warning_interval_seconds=10,
        working_directory=tmp_path,
        config=Settings(shell_execution=ShellSettings(show_bash=False)),
    )
    pid_path = tmp_path / "descendant.pid"
    script_path = tmp_path / "timeout_hold_pipe.py"
    script_path.write_text(
        "\n".join(
            [
                "import subprocess, sys, time",
                "print('before idle timeout', flush=True)",
                "child = subprocess.Popen(",
                "    [sys.executable, '-c', 'import time; time.sleep(30)'],",
                "    stdout=sys.stdout,",
                "    stderr=sys.stderr,",
                "    start_new_session=True,",
                ")",
                f"open({str(pid_path)!r}, 'w', encoding='utf-8').write(str(child.pid))",
                "time.sleep(30)",
            ]
        ),
        encoding="utf-8",
    )

    started = time.monotonic()
    try:
        execution = await executor.execute(
            ShellExecutionRequest(command=f'"{sys.executable}" "{script_path}"')
        )
    finally:
        _terminate_pid(pid_path)
    elapsed = time.monotonic() - started

    assert elapsed < 1
    assert execution.timed_out is True
    assert execution.io_drain_timed_out is True
    assert "before idle timeout" in execution.result.stdout


@pytest.mark.asyncio
async def test_execute_huge_output_exits_cleanly_with_low_byte_limit() -> None:
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        timeout_seconds=10,
        output_byte_limit=1024,
        config=Settings(shell_execution=ShellSettings(show_bash=False)),
    )

    command = f'"{sys.executable}" -c "import sys; sys.stdout.buffer.write(b\'x\' * 5_000_000)"'
    result = await runtime.execute({"command": command})

    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert "process exit code was 0" in text
    assert "[Output truncated: showing first" in text
    assert len(text.encode("utf-8")) < 5_000


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
        monkeypatch,
        {"name": "pwsh", "path": r"C:\Program Files\PowerShell\7\pwsh.exe"},
        timeout_seconds=0,
        warning_interval_seconds=0,
    )

    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    execution = await runtime._environment.execute(
        ShellExecutionRequest(command="Start-Sleep -Seconds 5", timeout=0)
    )

    ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
    assert ctrl_break is not None
    assert ctrl_break in process.sent_signals
    assert process.terminated is True
    assert captured["exec_args"][0].endswith("pwsh.exe")
    assert execution.timed_out is True


@pytest.mark.asyncio
async def test_execute_no_output_shows_compact_exit_banner_detail() -> None:
    """No-output commands should include compact '(no output)' + id detail."""
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(activation_reason="test", logger=logger, timeout_seconds=10)

    command = "exit 0" if platform.system() == "Windows" else "true"

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
async def test_execute_direct_shell_displays_streamed_output_once() -> None:
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        shell_environment=_DirectShellEnvironment(stream_output=True),
    )

    with console.console.capture() as capture:
        result = await runtime.execute_direct_shell("print-streamed")

    rendered = capture.get()
    assert result.stdout == "streamed\n"
    assert rendered.count("streamed") == 1
    assert "exit code" not in rendered


@pytest.mark.asyncio
async def test_execute_direct_shell_displays_buffered_output_when_adapter_does_not_stream() -> None:
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        shell_environment=_DirectShellEnvironment(stream_output=False),
    )

    with console.console.capture() as capture:
        result = await runtime.execute_direct_shell("print-buffered")

    rendered = capture.get()
    assert result.stdout == "buffered\n"
    assert rendered.count("buffered") == 1
    assert "exit code" not in rendered


@pytest.mark.asyncio
async def test_execute_direct_shell_ignores_configured_display_line_limit() -> None:
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        config=Settings(shell_execution=ShellSettings(output_display_lines=5, show_bash=True)),
        shell_environment=_DirectShellEnvironment(
            stream_output=True,
            stdout="0\n1\n2\n3\n4\n5\n6\n",
        ),
    )

    with console.console.capture() as capture:
        result = await runtime.execute_direct_shell("seven-lines")

    rendered = capture.get()
    assert result.stdout == "0\n1\n2\n3\n4\n5\n6\n"
    for line in ("0", "1", "2", "3", "4", "5", "6"):
        assert line in rendered
    assert SHELL_OUTPUT_TRUNCATION_MARKER not in rendered


@pytest.mark.asyncio
async def test_execute_direct_shell_displays_final_timeout_notice() -> None:
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        shell_environment=_DirectShellEnvironment(stream_output=False, timed_out=True),
    )

    with console.console.capture() as capture:
        result = await runtime.execute_direct_shell("sleep-too-long", timeout=5)

    rendered = capture.get()
    assert result.stdout == "buffered\n"
    assert "buffered" in rendered
    assert "Timeout after 5s - process terminated" in rendered


@pytest.mark.asyncio
async def test_execute_direct_shell_does_not_duplicate_callback_timeout_notice() -> None:
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        shell_environment=_DirectShellEnvironment(stream_output=True, timed_out=True),
    )

    with console.console.capture() as capture:
        result = await runtime.execute_direct_shell("sleep-too-long", timeout=5)

    rendered = capture.get()
    assert result.stdout == "streamed\n"
    assert rendered.count("Timeout") == 1


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

    command = f'"{sys.executable}" -c "for i in range(1, 11): print(\'out-{{0:02d}}\'.format(i))"'

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
    assert SHELL_OUTPUT_TRUNCATION_MARKER in rendered
    assert "10 lines" in rendered


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
    assert getattr(result, "output_line_count", None) == 1
    rendered = capture.get()
    assert "hello" not in rendered
    assert "exit code" not in rendered


@pytest.mark.asyncio
async def test_execute_progress_only_mode_suppresses_live_console_output() -> None:
    """Progress-only display mode should suppress streamed shell output."""
    logger = logging.getLogger("shell-runtime-test")
    runtime = ShellRuntime(activation_reason="test", logger=logger, timeout_seconds=10)

    with suppress_interactive_display():
        with console.console.capture() as capture:
            result = await runtime.execute({"command": "echo hello"})

    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "hello" in result.content[0].text
    assert "process exit code was 0" in result.content[0].text
    rendered = capture.get()
    assert "hello" not in rendered
    assert "exit code" not in rendered


@pytest.mark.asyncio
async def test_execute_emits_shell_lifecycle_progress_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = RecordingFastLogger()
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logger,
        timeout_seconds=10,
        agent_name="assistant",
        shell_environment=_TestLocalShellExecutor(
            logger=logger,
            runtime_info={"name": "bash", "path": "/bin/bash"},
            timeout_seconds=10,
            warning_interval_seconds=30,
        ),
    )

    process = DummyProcess()
    process.returncode = 0
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
    assert end_payload["details"] == "completed (exit 0)"
    assert end_payload["tool_state"] == "completed"
    assert end_payload["tool_terminal"] is True


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
    assert "Command execution failed" in result.content[0].text

    progress_payloads = _extract_progress_payloads(logger)
    assert len(progress_payloads) == 2
    assert progress_payloads[0]["progress_action"] == ProgressAction.CALLING_TOOL
    assert progress_payloads[0]["tool_event"] == "start"
    assert progress_payloads[1]["progress_action"] == ProgressAction.TOOL_PROGRESS
    assert progress_payloads[1]["details"] == "failed: spawn failed"
    assert progress_payloads[1]["tool_state"] == "failed"
    assert progress_payloads[1]["tool_terminal"] is True
