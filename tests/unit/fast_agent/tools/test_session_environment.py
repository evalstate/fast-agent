from __future__ import annotations

from fast_agent.tools.session_environment import (
    SessionEnvironment,
    ShellExecutionResult,
    ShellExecutor,
)


def test_shell_execution_result_is_structured() -> None:
    result = ShellExecutionResult(stdout="out", stderr="err", exit_code=2)

    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.exit_code == 2


def test_session_environment_protocol_exports() -> None:
    assert ShellExecutor.__name__ == "ShellExecutor"
    assert SessionEnvironment.__name__ == "SessionEnvironment"
