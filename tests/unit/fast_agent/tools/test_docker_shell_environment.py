from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from fast_agent.core.exceptions import EnvironmentStartupError
from fast_agent.tools import docker_shell_environment as docker_environment_module
from fast_agent.tools.docker_shell_environment import (
    DockerManagedShellEnvironment,
    DockerMount,
    DockerShellEnvironment,
)
from fast_agent.tools.execution_environment import (
    EnvironmentFilesystemWithBytes,
    ShellExecutionRequest,
)
from fast_agent.tools.shell_output_spool import ShellOutputSpoolPaths
from fast_agent.tools.shell_runtime import ShellRuntime


class _DockerFsProcess:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.communicated_stdin: bytes | None = None

    async def communicate(self, stdin: bytes | None = None) -> tuple[bytes, bytes]:
        self.communicated_stdin = stdin
        return self._stdout, self._stderr


def test_docker_shell_environment_builds_bash_exec_argv() -> None:
    environment = DockerShellEnvironment(container="workspace", cwd="/workspace")

    argv = environment._exec_argv(
        ShellExecutionRequest(
            command="pwd",
            cwd="/work",
            env={"TOKEN": "value"},
        )
    )

    assert argv == [
        "docker",
        "exec",
        "-e",
        "TOKEN",
        "-w",
        "/work",
        "workspace",
        "bash",
        "-lc",
        "pwd",
    ]
    assert environment._exec_process_env(  # noqa: SLF001
        ShellExecutionRequest(command="pwd", cwd="/work", env={"TOKEN": "value"})
    )["TOKEN"] == "value"


def test_docker_shell_environment_builds_powershell_exec_argv() -> None:
    environment = DockerShellEnvironment(container="workspace", shell="pwsh", cwd="/workspace")

    argv = environment._exec_argv(ShellExecutionRequest(command="Get-Location"))

    assert argv == [
        "docker",
        "exec",
        "-w",
        "/workspace",
        "workspace",
        "pwsh",
        "-NoLogo",
        "-NoProfile",
        "-Command",
        "Get-Location",
    ]


@pytest.mark.asyncio
async def test_managed_powershell_execution_is_rejected_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def create_process(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail("managed PowerShell execution must not launch docker exec")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    environment = DockerShellEnvironment(container="workspace", shell="pwsh", cwd="/workspace")

    with pytest.raises(
        RuntimeError,
        match="Managed Docker execution is not supported for PowerShell",
    ):
        await environment.execute(
            ShellExecutionRequest(
                command="Get-Location",
                terminate_after_idle=False,
            )
        )


def test_docker_managed_exec_publishes_a_container_process_group_pid() -> None:
    environment = DockerShellEnvironment(container="workspace", cwd="/workspace")

    argv = environment._managed_exec_argv(
        ShellExecutionRequest(
            command="python -m http.server 8000",
            cwd="/workspace/subdir",
            terminate_after_idle=False,
        ),
        "/tmp/fast-agent-managed/process.pid",
    )

    assert argv[:6] == [
        "docker",
        "exec",
        "-w",
        "/workspace/subdir",
        "workspace",
        "sh",
    ]
    assert "setsid" in argv[7]
    assert argv[-3:] == [
        "/tmp/fast-agent-managed/process.pid",
        "bash",
        "python -m http.server 8000",
    ]


def test_detached_docker_managed_exec_redirects_to_spool_files() -> None:
    environment = DockerShellEnvironment(container="workspace", cwd="/workspace")
    spool = ShellOutputSpoolPaths(
        directory="/tmp/fast-agent-managed/run",
        stdout="/tmp/fast-agent-managed/run/stdout.log",
        stderr="/tmp/fast-agent-managed/run/stderr.log",
    )

    argv = environment._managed_exec_argv(
        ShellExecutionRequest(
            command="python -m http.server 8000",
            terminate_after_idle=False,
            detach=True,
        ),
        "/tmp/fast-agent-managed/run/process.pid",
        output_spool=spool,
    )

    assert '>"$stdout_file" 2>"$stderr_file"' in argv[7]
    assert argv[-2:] == [spool.stdout, spool.stderr]


@pytest.mark.asyncio
async def test_docker_managed_output_reader_requests_byte_range() -> None:
    class _ReadEnvironment(DockerShellEnvironment):
        def __init__(self) -> None:
            super().__init__(container="workspace")
            self.script = ""
            self.args: list[str] = []

        async def _docker_shell_bytes(
            self,
            script: str,
            args: list[str],
            *,
            stdin: bytes | None = None,
        ) -> docker_environment_module._DockerExecResult:
            assert stdin is None
            self.script = script
            self.args = args
            return docker_environment_module._DockerExecResult(
                exit_code=0,
                stdout=b"chunk",
                stderr="",
            )

    environment = _ReadEnvironment()

    payload = await environment._read_managed_output_chunk(
        "/tmp/output.log",
        128,
        4096,
    )

    assert payload == b"chunk"
    assert "tail -c" in environment.script
    assert environment.args == ["/tmp/output.log", "128", "4096"]


@pytest.mark.asyncio
async def test_docker_managed_termination_uses_term_then_kill_in_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = DockerShellEnvironment(container="workspace", cwd="/workspace")
    signals: list[tuple[int, str]] = []
    running = iter([True, False])

    async def signal(process_id: int, *, signal_name: str) -> None:
        signals.append((process_id, signal_name))

    async def is_running(process_id: int) -> bool:
        assert process_id == 4242
        return next(running)

    monkeypatch.setattr(
        docker_environment_module,
        "_MANAGED_PROCESS_TERM_GRACE_SECONDS",
        0,
    )
    monkeypatch.setattr(environment, "_signal_container_process_group", signal)
    monkeypatch.setattr(environment, "_container_process_is_running", is_running)

    await environment._kill_container_process_group(4242)

    assert signals == [(4242, "TERM"), (4242, "KILL")]


def test_docker_shell_environment_uses_configured_container_cli() -> None:
    environment = DockerShellEnvironment(
        container="workspace",
        container_cli="wslc",
        cwd="/workspace",
    )

    argv = environment._exec_argv(ShellExecutionRequest(command="pwd"))

    assert argv[:2] == ["wslc", "exec"]
    assert environment.runtime_info().provider == "wslc"


def test_docker_shell_environment_exposes_container_filesystem_protocol() -> None:
    environment = DockerShellEnvironment(container="workspace", cwd="/workspace/project")

    assert isinstance(environment, EnvironmentFilesystemWithBytes)
    assert environment.resolve_path("src/main.py") == "/workspace/project/src/main.py"
    assert environment.resolve_path("/tmp/file.txt") == "/tmp/file.txt"


def test_shell_runtime_resolves_relative_cwd_before_docker_exec() -> None:
    environment = DockerShellEnvironment(container="workspace", cwd="/workspace")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        shell_environment=environment,
    )

    cwd = runtime._resolve_managed_working_directory("subdir")
    argv = environment._exec_argv(
        ShellExecutionRequest(command="pwd", cwd=cwd)
    )

    assert cwd == "/workspace/subdir"
    assert argv[argv.index("-w") + 1] == "/workspace/subdir"


def test_docker_mount_formats_host_path() -> None:
    mount = DockerMount(source=Path("."), target="/workspace", mode="ro")

    assert mount.docker_arg().endswith(":/workspace:ro")


def test_managed_docker_environment_mount_args() -> None:
    environment = DockerManagedShellEnvironment(
        image="ubuntu:24.04",
        mounts=[DockerMount(source=Path("."), target="/workspace")],
    )

    args = environment._mount_args()

    assert args[0] == "-v"
    assert args[1].endswith(":/workspace:rw")


@pytest.mark.asyncio
async def test_docker_shell_environment_reads_text_from_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def create_process(*args: object, **kwargs: object) -> _DockerFsProcess:
        calls.append((args, kwargs))
        return _DockerFsProcess(stdout=b"hello")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    environment = DockerShellEnvironment(
        container="workspace",
        container_cli="wslc",
        cwd="/workspace",
    )

    content = await environment.read_text("README.md")

    assert content == "hello"
    assert calls[0][0][:3] == ("wslc", "exec", "workspace")
    assert calls[0][0][-1] == "/workspace/README.md"


@pytest.mark.asyncio
async def test_docker_shell_environment_writes_bytes_to_container_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes: list[_DockerFsProcess] = []
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def create_process(*args: object, **kwargs: object) -> _DockerFsProcess:
        process = _DockerFsProcess()
        processes.append(process)
        calls.append((args, kwargs))
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    environment = DockerShellEnvironment(container="workspace", cwd="/workspace")

    await environment.write_bytes("dir/file.bin", b"payload")

    assert calls[0][0][:4] == ("docker", "exec", "-i", "workspace")
    assert calls[0][0][-2:] == ("/workspace/dir/file.bin", "/workspace/dir")
    assert processes[0].communicated_stdin == b"payload"


@pytest.mark.asyncio
async def test_docker_shell_environment_lists_container_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"d\0/workspace/pkg\0pkg\0f\0/workspace/README.md\0README.md\0l\0/workspace/link\0link\0"

    async def create_process(*args: object, **kwargs: object) -> _DockerFsProcess:
        del args, kwargs
        return _DockerFsProcess(stdout=payload)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    environment = DockerShellEnvironment(container="workspace", cwd="/workspace")

    entries = await environment.list_dir(".")

    assert [(entry.name, entry.path, entry.kind) for entry in entries] == [
        ("pkg", "/workspace/pkg", "directory"),
        ("README.md", "/workspace/README.md", "file"),
        ("link", "/workspace/link", "other"),
    ]


@pytest.mark.asyncio
async def test_managed_docker_environment_reports_missing_container_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def missing_cli(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("docker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_cli)
    environment = DockerManagedShellEnvironment(image="ubuntu:24.04")

    with pytest.raises(EnvironmentStartupError, match="Container CLI not found: docker"):
        await environment.open()


@pytest.mark.asyncio
async def test_managed_docker_environment_emits_startup_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"container-id\n", b""

    async def create_process(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return _Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    environment = DockerManagedShellEnvironment(image="ubuntu:24.04")
    stages: list[str] = []
    environment.set_startup_progress_callback(stages.append)

    await environment.open()

    assert any(stage.startswith("starting docker container fast-agent-") for stage in stages)
    assert "waiting for container start" in stages
    assert any(stage.startswith("container ready fast-agent-") for stage in stages)
