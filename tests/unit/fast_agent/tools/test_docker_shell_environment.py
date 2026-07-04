from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from fast_agent.core.exceptions import EnvironmentStartupError
from fast_agent.tools.docker_shell_environment import (
    DockerManagedShellEnvironment,
    DockerMount,
    DockerShellEnvironment,
)
from fast_agent.tools.execution_environment import ShellExecutionRequest


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


def test_docker_shell_environment_uses_configured_container_cli() -> None:
    environment = DockerShellEnvironment(
        container="workspace",
        container_cli="wslc",
        cwd="/workspace",
    )

    argv = environment._exec_argv(ShellExecutionRequest(command="pwd"))

    assert argv[:2] == ["wslc", "exec"]
    assert environment.runtime_info().provider == "wslc"


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
