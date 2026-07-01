from __future__ import annotations

from pathlib import Path

from fast_agent.tools.docker_shell_environment import (
    DockerManagedShellEnvironment,
    DockerMount,
    DockerShellEnvironment,
)
from fast_agent.tools.session_environment import ShellExecutionRequest


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
        "TOKEN=value",
        "-w",
        "/work",
        "workspace",
        "bash",
        "-lc",
        "pwd",
    ]


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
