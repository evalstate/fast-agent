from __future__ import annotations

from fast_agent.tools.execution_environment import (
    ShellEnvironment,
    ShellExecution,
    ShellExecutionOptions,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellRuntimeInfo,
)


def test_shell_execution_result_is_structured() -> None:
    result = ShellExecutionResult(stdout="out", stderr="err", exit_code=2)

    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.exit_code == 2


def test_execution_environment_protocol_exports() -> None:
    assert ShellEnvironment.__name__ == "ShellEnvironment"


def test_shell_execution_request_keeps_remote_paths_as_strings() -> None:
    request = ShellExecutionRequest(command="pwd", cwd="/workspace")

    assert request.cwd == "/workspace"


def test_shell_execution_carries_full_metadata() -> None:
    result = ShellExecution(
        result=ShellExecutionResult(stdout="out", stderr="", exit_code=0),
        options=ShellExecutionOptions(timeout_seconds=3, warning_interval_seconds=1),
        timed_out=True,
        io_drain_timed_out=True,
    )

    assert result.result.stdout == "out"
    assert result.options.timeout_seconds == 3
    assert result.timed_out is True
    assert result.io_drain_timed_out is True


def test_shell_runtime_info_is_typed() -> None:
    info = ShellRuntimeInfo(name="bash", path="/bin/bash", kind="docker", provider="docker")

    assert info.name == "bash"
    assert info.kind == "docker"
