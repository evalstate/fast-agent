from __future__ import annotations

import asyncio
import base64
import logging
import re
import sys
import threading
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from fast_agent.core.exceptions import EnvironmentStartupError
from fast_agent.tools import huggingface_sandbox_environment as hf_sandbox_environment
from fast_agent.tools.execution_environment import ShellExecutionRequest
from fast_agent.tools.huggingface_sandbox_environment import (
    FAST_AGENT_HF_SANDBOX_LABEL,
    HuggingFaceSandboxEnvironment,
    HuggingFaceVolumeMount,
    _SandboxCommandResult,
    _SandboxFiles,
    _SandboxProcess,
)
from fast_agent.tools.huggingface_sandbox_environment import (
    _Sandbox as SandboxProtocol,
)
from fast_agent.tools.shell_runtime import ShellRuntime


class _CommandResult(_SandboxCommandResult):
    def __init__(
        self,
        *,
        stdout: str = "ok",
        stderr: str = "",
        exit_code: int | None = 0,
        timed_out: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.timed_out = timed_out


class _Files(_SandboxFiles):
    def __init__(self) -> None:
        self.created: list[str] = []

    def mkdir(self, path: str) -> None:
        self.created.append(path)

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return ""

    def write(self, path: str, data: str | bytes, mode: str | None = None) -> None:
        pass

    def exists(self, path: str) -> bool:
        return True

    def delete(self, path: str, recursive: bool = False) -> None:
        pass


class _Sandbox(SandboxProtocol):
    def __init__(self) -> None:
        self.id = "sandbox-job-123"
        self.test_files = _Files()
        self.files: _Files = self.test_files
        self.cwd: str | None = None
        self.commands: list[str | list[str]] = []

    def run(
        self,
        cmd: str | list[str],
        *,
        shell: bool | None = None,
        env: dict[str, Any] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
        check: bool = True,
        background: bool = False,
    ) -> _CommandResult | _SandboxProcess:
        del shell, env, timeout, on_stdout, on_stderr, check, background
        self.commands.append(cmd)
        self.cwd = cwd
        if isinstance(cmd, list) and cmd[:2] == ["python3", "-c"]:
            return _CommandResult(
                stdout=(
                    "["
                    '{"path": "/workspace/skills/alpha", "name": "alpha", "kind": "directory"},'
                    '{"path": "/workspace/skills/readme.txt", "name": "readme.txt", "kind": "file"}'
                    "]"
                )
            )
        return _CommandResult()

    def kill(self) -> None:
        pass

    def processes(self) -> list[Any]:
        return []

    def close(self) -> None:
        pass


class _StreamingSandbox(_Sandbox):
    def __init__(self, *, timed_out: bool = False) -> None:
        super().__init__()
        self.timed_out = timed_out

    def run(
        self,
        cmd: str | list[str],
        *,
        shell: bool | None = None,
        env: dict[str, Any] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
        check: bool = True,
        background: bool = False,
    ) -> _CommandResult:
        del shell, env, timeout, check, background
        self.commands.append(cmd)
        self.cwd = cwd
        if on_stdout is not None:
            on_stdout("a")
        if on_stderr is not None:
            on_stderr("b")
        if on_stdout is not None:
            on_stdout("c")
        return _CommandResult(stdout="ac", stderr="b", timed_out=self.timed_out)


class _ManagedFiles(_Files):
    def __init__(self) -> None:
        super().__init__()
        self.contents: dict[str, str] = {}
        self.deleted: list[tuple[str, bool]] = []

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        del encoding
        return self.contents[path]

    def exists(self, path: str) -> bool:
        return path in self.contents

    def delete(self, path: str, recursive: bool = False) -> None:
        self.deleted.append((path, recursive))
        prefix = f"{path.rstrip('/')}/"
        for candidate in list(self.contents):
            if candidate == path or candidate.startswith(prefix):
                self.contents.pop(candidate)


class _ManagedProcess:
    def __init__(self, pid: int = 9876) -> None:
        self.pid = pid
        self.running = True
        self.exit_code: int | None = None
        self.kill_count = 0

    def kill(self) -> None:
        self.kill_count += 1
        self.running = False
        self.exit_code = -15


class _ManagedSandbox(_Sandbox):
    def __init__(
        self,
        *,
        auto_complete: bool,
        block_spawn: bool = False,
        stdout_content: str = "managed stdout",
        stderr_content: str = "managed stderr",
    ) -> None:
        super().__init__()
        self.managed_files = _ManagedFiles()
        self.files = self.managed_files
        self.process = _ManagedProcess()
        self.auto_complete = auto_complete
        self.block_spawn = block_spawn
        self.spawn_entered = threading.Event()
        self.spawn_release = threading.Event()
        self.background_requested = False
        self.stdout_path: str | None = None
        self.stderr_path: str | None = None
        self.stdout_content = stdout_content
        self.stderr_content = stderr_content
        self.output_read_requests: list[tuple[str, int, int]] = []

    def run(
        self,
        cmd: str | list[str],
        *,
        shell: bool | None = None,
        env: dict[str, Any] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
        check: bool = True,
        background: bool = False,
    ) -> _CommandResult | _ManagedProcess:
        del shell, env, timeout, on_stdout, on_stderr, check
        self.commands.append(cmd)
        if background or cwd is not None:
            self.cwd = cwd
        assert isinstance(cmd, list)
        if not background:
            if cmd[:2] == ["python3", "-c"]:
                path = cmd[3]
                offset = int(cmd[4])
                length = int(cmd[5])
                payload = self.managed_files.contents.get(path, "").encode("utf-8")[
                    offset : offset + length
                ]
                self.output_read_requests.append((path, offset, length))
                return _CommandResult(stdout=base64.b64encode(payload).decode("ascii"))
            assert "kill -" in cmd[-1]
            self.process.kill()
            return _CommandResult()
        self.background_requested = True
        script = cmd[-1]
        match = re.search(r"exec >(\S+) 2>(\S+)", script)
        assert match is not None
        self.stdout_path, self.stderr_path = match.groups()
        self.spawn_entered.set()
        if self.block_spawn:
            self.spawn_release.wait(timeout=5)
        return self.process

    def processes(self) -> list[_SandboxProcess]:
        if self.auto_complete and self.process.running:
            assert self.stdout_path is not None
            assert self.stderr_path is not None
            self.managed_files.contents[self.stdout_path] = self.stdout_content
            self.managed_files.contents[self.stderr_path] = self.stderr_content
            self.process.running = False
            self.process.exit_code = 0
        return [self.process]


class _ManagedPollingFailureSandbox(_ManagedSandbox):
    def processes(self) -> list[_SandboxProcess]:
        raise RuntimeError("process listing failed")


class _RecordingCallbacks:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def on_started(self, process_id: int | None) -> None:
        self.events.append(("started", str(process_id)))

    async def on_stdout(self, text: str) -> None:
        await asyncio.sleep(0.01)
        self.events.append(("stdout", text))

    async def on_stderr(self, text: str) -> None:
        await asyncio.sleep(0.01)
        self.events.append(("stderr", text))

    async def on_idle_warning(self, elapsed: float, remaining: float) -> None:
        self.events.append(("idle", f"{elapsed}:{remaining}"))

    async def on_timeout(self) -> None:
        self.events.append(("timeout", ""))


@pytest.mark.asyncio
async def test_open_creates_configured_cwd() -> None:
    sandbox = _Sandbox()
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace/project")

    await environment.open()

    assert sandbox.test_files.created == ["/workspace/project"]


@pytest.mark.asyncio
async def test_open_existing_sandbox_emits_startup_stages() -> None:
    sandbox = _Sandbox()
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace/project")
    stages: list[str] = []
    environment.set_startup_progress_callback(stages.append)

    await environment.open()

    assert stages == [
        "using existing sandbox sandbox-job-123",
        "preparing cwd /workspace/project",
        "sandbox filesystem ready",
    ]


@pytest.mark.asyncio
async def test_execute_uses_created_default_cwd() -> None:
    sandbox = _Sandbox()
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    await environment.open()

    await environment.execute(ShellExecutionRequest(command="pwd"))

    assert sandbox.cwd == "/workspace"


@pytest.mark.asyncio
async def test_execute_streams_huggingface_output_chunks_without_duplicates() -> None:
    sandbox = _StreamingSandbox()
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    callbacks = _RecordingCallbacks()
    await environment.open()

    execution = await environment.execute(
        ShellExecutionRequest(command="printf output"),
        callbacks=callbacks,
    )

    assert callbacks.events == [
        ("started", "None"),
        ("stdout", "a"),
        ("stderr", "b"),
        ("stdout", "c"),
    ]
    assert execution.result.stdout == "ac"
    assert execution.result.stderr == "b"


@pytest.mark.asyncio
async def test_execute_reports_huggingface_timeout_after_streaming_output() -> None:
    sandbox = _StreamingSandbox(timed_out=True)
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    callbacks = _RecordingCallbacks()
    await environment.open()

    execution = await environment.execute(
        ShellExecutionRequest(command="sleep 10", timeout=1),
        callbacks=callbacks,
    )

    assert callbacks.events == [
        ("started", "None"),
        ("stdout", "a"),
        ("stderr", "b"),
        ("stdout", "c"),
        ("timeout", ""),
    ]
    assert execution.timed_out is True


@pytest.mark.asyncio
async def test_managed_execute_uses_cancellable_remote_process_and_streams_spooled_output() -> None:
    sandbox = _ManagedSandbox(auto_complete=True)
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    callbacks = _RecordingCallbacks()
    await environment.open()

    execution = await environment.execute(
        ShellExecutionRequest(
            command="printf managed",
            terminate_after_idle=False,
        ),
        callbacks=callbacks,
    )

    assert sandbox.background_requested is True
    assert sandbox.cwd == "/workspace"
    assert callbacks.events == [
        ("started", "9876"),
        ("stdout", "managed stdout"),
        ("stderr", "managed stderr"),
    ]
    assert execution.result.stdout == "managed stdout"
    assert execution.result.stderr == "managed stderr"
    assert execution.result.exit_code == 0
    assert sandbox.managed_files.deleted
    assert sandbox.managed_files.deleted[-1][1] is True


@pytest.mark.asyncio
async def test_managed_execute_reads_remote_output_once_by_advancing_byte_offsets() -> None:
    chunk_size = hf_sandbox_environment._MANAGED_OUTPUT_READ_CHUNK_BYTES
    stdout = "x" * (chunk_size + 17)
    sandbox = _ManagedSandbox(
        auto_complete=True,
        stdout_content=stdout,
        stderr_content="",
    )
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    callbacks = _RecordingCallbacks()
    await environment.open()

    execution = await environment.execute(
        ShellExecutionRequest(
            command="produce substantial output",
            terminate_after_idle=False,
            retain_output=False,
        ),
        callbacks=callbacks,
    )

    assert execution.result.stdout == ""
    assert execution.result.stderr == ""
    assert "".join(text for event, text in callbacks.events if event == "stdout") == stdout
    assert sandbox.stdout_path is not None
    stdout_offsets = [
        offset
        for path, offset, _ in sandbox.output_read_requests
        if path == sandbox.stdout_path
    ]
    assert stdout_offsets == [0, 0, chunk_size]


@pytest.mark.asyncio
async def test_managed_execute_preserves_utf8_characters_split_across_remote_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_sandbox_environment, "_MANAGED_OUTPUT_READ_CHUNK_BYTES", 2)
    monkeypatch.setattr(hf_sandbox_environment, "_MANAGED_OUTPUT_CHUNKS_PER_POLL", 1)
    sandbox = _ManagedSandbox(
        auto_complete=True,
        stdout_content="€",
        stderr_content="",
    )
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    callbacks = _RecordingCallbacks()
    await environment.open()

    execution = await environment.execute(
        ShellExecutionRequest(
            command="printf euro",
            terminate_after_idle=False,
        ),
        callbacks=callbacks,
    )

    assert execution.result.stdout == "€"
    assert "".join(text for event, text in callbacks.events if event == "stdout") == "€"


@pytest.mark.asyncio
async def test_managed_execute_cancellation_kills_remote_process() -> None:
    sandbox = _ManagedSandbox(auto_complete=False)
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    callbacks = _RecordingCallbacks()
    await environment.open()

    task = asyncio.create_task(
        environment.execute(
            ShellExecutionRequest(
                command="sleep 60",
                terminate_after_idle=False,
            ),
            callbacks=callbacks,
        )
    )
    while ("started", "9876") not in callbacks.events:
        await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert sandbox.process.kill_count == 1
    assert sandbox.process.running is False
    assert sandbox.managed_files.deleted


@pytest.mark.asyncio
async def test_managed_execute_cancellation_during_spawn_kills_process_after_spawn() -> None:
    sandbox = _ManagedSandbox(auto_complete=False, block_spawn=True)
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    await environment.open()

    task = asyncio.create_task(
        environment.execute(
            ShellExecutionRequest(
                command="sleep 60",
                terminate_after_idle=False,
            )
        )
    )
    await asyncio.to_thread(sandbox.spawn_entered.wait, 2)
    task.cancel()
    sandbox.spawn_release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert sandbox.process.kill_count == 1
    assert sandbox.process.running is False
    assert sandbox.managed_files.deleted


@pytest.mark.asyncio
async def test_managed_execute_polling_failure_kills_remote_process() -> None:
    sandbox = _ManagedPollingFailureSandbox(auto_complete=False)
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    await environment.open()

    with pytest.raises(RuntimeError, match="process listing failed"):
        await environment.execute(
            ShellExecutionRequest(
                command="sleep 60",
                terminate_after_idle=False,
            )
        )

    assert sandbox.process.kill_count == 1
    assert sandbox.process.running is False
    assert sandbox.managed_files.deleted


@pytest.mark.asyncio
async def test_shell_runtime_terminate_process_kills_huggingface_remote_process() -> None:
    sandbox = _ManagedSandbox(auto_complete=False)
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        shell_environment=environment,
    )
    await environment.open()

    started = await runtime.execute(
        {
            "command": "sleep 60",
            "background": True,
        }
    )
    terminated = await runtime.terminate_process({"process_id": "process-1"})

    assert started.isError is False
    assert terminated.isError is False
    assert sandbox.process.kill_count == 1
    assert sandbox.process.running is False
    await runtime.close()


@pytest.mark.asyncio
async def test_list_dir_returns_session_file_entries() -> None:
    sandbox = _Sandbox()
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    await environment.open()

    entries = await environment.list_dir("skills")

    assert [entry.name for entry in entries] == ["alpha", "readme.txt"]
    assert [entry.path for entry in entries] == [
        "/workspace/skills/alpha",
        "/workspace/skills/readme.txt",
    ]
    assert [entry.kind for entry in entries] == ["directory", "file"]
    assert isinstance(sandbox.commands[-1], list)
    assert sandbox.commands[-1][0] == "python3"
    assert sandbox.commands[-1][-1] == "/workspace/skills"


def test_create_sandbox_adds_fast_agent_version_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_update: dict[str, Any] | None = None

    class FakeSandbox:
        @staticmethod
        def create(
            image: str = "python:3.12",
            *,
            flavor: str = "cpu-basic",
            idle_timeout: int | float | str | None = None,
            env: dict[str, Any] | None = None,
            secrets: dict[str, Any] | None = None,
            volumes: list[Any] | None = None,
            namespace: str | None = None,
            forward_hf_token: bool = False,
            start_timeout: float = 120.0,
            token: str | None = None,
        ) -> _Sandbox:
            del (
                image,
                flavor,
                idle_timeout,
                env,
                secrets,
                volumes,
                namespace,
                forward_hf_token,
                start_timeout,
                token,
            )
            return _Sandbox()

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def update_job_labels(
            self,
            *,
            job_id: str,
            labels: dict[str, str],
            namespace: str | None = None,
            token: str | None = None,
        ) -> None:
            nonlocal captured_update
            captured_update = {
                "job_id": job_id,
                "labels": labels,
                "namespace": namespace,
                "token": token,
            }

    class FakeVolume:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeHfApi, Sandbox=FakeSandbox, Volume=FakeVolume),
    )
    monkeypatch.setattr(hf_sandbox_environment, "version", lambda package: "0.9.0")

    environment = HuggingFaceSandboxEnvironment(namespace="test-org", token="hf_test")

    environment._create_sandbox()

    assert captured_update == {
        "job_id": "sandbox-job-123",
        "labels": {FAST_AGENT_HF_SANDBOX_LABEL: "0_9_0"},
        "namespace": "test-org",
        "token": "hf_test",
    }


def test_create_sandbox_resolves_hf_token_reference_from_hub_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_token: str | None = None

    class FakeSandbox:
        @staticmethod
        def create(
            image: str = "python:3.12",
            *,
            flavor: str = "cpu-basic",
            idle_timeout: int | float | str | None = None,
            env: dict[str, Any] | None = None,
            secrets: dict[str, Any] | None = None,
            volumes: list[Any] | None = None,
            namespace: str | None = None,
            forward_hf_token: bool = False,
            start_timeout: float = 120.0,
            token: str | None = None,
        ) -> _Sandbox:
            del (
                image,
                flavor,
                idle_timeout,
                env,
                secrets,
                volumes,
                namespace,
                forward_hf_token,
                start_timeout,
            )
            nonlocal captured_token
            captured_token = token
            return _Sandbox()

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def update_job_labels(
            self,
            *,
            job_id: str,
            labels: dict[str, str],
            namespace: str | None = None,
            token: str | None = None,
        ) -> None:
            del job_id, labels, namespace, token

    class FakeVolume:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeHfApi, Sandbox=FakeSandbox, Volume=FakeVolume),
    )
    monkeypatch.setattr(
        hf_sandbox_environment,
        "get_huggingface_hub_token",
        lambda: "hf_cached_token",
    )
    environment = HuggingFaceSandboxEnvironment(token="${HF_TOKEN}")

    environment._create_sandbox()

    assert captured_token == "hf_cached_token"


def test_create_sandbox_rejects_unresolved_token_env_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hf_sandbox_environment, "get_huggingface_hub_token", lambda: None)
    environment = HuggingFaceSandboxEnvironment(token="${HF_TOKEN}")

    with pytest.raises(EnvironmentStartupError) as exc_info:
        environment._create_sandbox()

    assert exc_info.value.message == "Could not start Hugging Face sandbox environment."
    assert "Environment variable HF_TOKEN is not set" in exc_info.value.details


def test_create_sandbox_wraps_huggingface_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSandbox:
        @staticmethod
        def create(
            image: str = "python:3.12",
            *,
            flavor: str = "cpu-basic",
            idle_timeout: int | float | str | None = None,
            env: dict[str, Any] | None = None,
            secrets: dict[str, Any] | None = None,
            volumes: list[Any] | None = None,
            namespace: str | None = None,
            forward_hf_token: bool = False,
            start_timeout: float = 120.0,
            token: str | None = None,
        ) -> _Sandbox:
            del (
                image,
                flavor,
                idle_timeout,
                env,
                secrets,
                volumes,
                namespace,
                forward_hf_token,
                start_timeout,
                token,
            )
            raise RuntimeError("Invalid user token.")

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

    class FakeVolume:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeHfApi, Sandbox=FakeSandbox, Volume=FakeVolume),
    )
    environment = HuggingFaceSandboxEnvironment(token="bad-token")

    with pytest.raises(EnvironmentStartupError) as exc_info:
        environment._create_sandbox()

    assert exc_info.value.message == "Could not start Hugging Face sandbox environment."
    assert "Hugging Face rejected the configured token" in exc_info.value.details


def test_create_sandbox_passes_configured_hf_volume_mounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_volumes: list[Any] | None = None

    class FakeSandbox:
        @staticmethod
        def create(
            image: str = "python:3.12",
            *,
            flavor: str = "cpu-basic",
            idle_timeout: int | float | str | None = None,
            env: dict[str, Any] | None = None,
            secrets: dict[str, Any] | None = None,
            volumes: list[Any] | None = None,
            namespace: str | None = None,
            forward_hf_token: bool = False,
            start_timeout: float = 120.0,
            token: str | None = None,
        ) -> _Sandbox:
            del (
                image,
                flavor,
                idle_timeout,
                env,
                secrets,
                namespace,
                forward_hf_token,
                start_timeout,
                token,
            )
            nonlocal captured_volumes
            captured_volumes = volumes
            return _Sandbox()

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def update_job_labels(
            self,
            *,
            job_id: str,
            labels: dict[str, str],
            namespace: str | None = None,
            token: str | None = None,
        ) -> None:
            del job_id, labels, namespace, token

    class FakeVolume:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeHfApi, Sandbox=FakeSandbox, Volume=FakeVolume),
    )

    environment = HuggingFaceSandboxEnvironment(
        volume_mounts=(
            HuggingFaceVolumeMount(
                type="dataset",
                source="org/data",
                mount_path="/data",
                read_only=True,
                path="train",
                revision="main",
            ),
        )
    )

    environment._create_sandbox()

    assert captured_volumes is not None
    assert [volume.kwargs for volume in captured_volumes] == [
        {
            "type": "dataset",
            "source": "org/data",
            "mount_path": "/data",
            "read_only": True,
            "path": "train",
            "revision": "main",
        }
    ]
