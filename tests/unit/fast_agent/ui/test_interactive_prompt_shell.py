from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.tools.execution_environment import ShellExecutionResult, ShellRuntimeInfo
from fast_agent.ui.interactive_prompt import InteractivePrompt, PendingCommandExecution

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.ui.console_display import ConsoleDisplay


class _ShellRuntime:
    def __init__(self, *, kind: str = "remote") -> None:
        self.kind = kind
        self.commands: list[str] = []
        self.direct_commands: list[str] = []

    def runtime_info(self) -> ShellRuntimeInfo:
        if self.kind == "local":
            return ShellRuntimeInfo(name="bash", kind="local")
        return ShellRuntimeInfo(
            name="sh",
            kind="remote",
            provider="huggingface",
            environment_name="hf-gpu",
        )

    async def execute_shell(self, command: str) -> ShellExecutionResult:
        self.commands.append(command)
        return ShellExecutionResult(stdout="remote output\n", stderr="", exit_code=0)

    async def execute_direct_shell(self, command: str) -> ShellExecutionResult:
        self.direct_commands.append(command)
        print("remote output")
        return ShellExecutionResult(stdout="remote output\n", stderr="", exit_code=0)


class _Agent:
    def __init__(self, shell_runtime: _ShellRuntime) -> None:
        self.shell_runtime = shell_runtime


class _Provider:
    def __init__(self, shell_runtime: _ShellRuntime) -> None:
        self._agent_obj = _Agent(shell_runtime)

    def _agent(self, _agent_name: str) -> _Agent:
        return self._agent_obj


class _Display:
    def __init__(self) -> None:
        self.exit_codes: list[int] = []

    def show_shell_exit_code(self, exit_code: int) -> None:
        self.exit_codes.append(exit_code)


@pytest.mark.asyncio
async def test_environment_shell_command_uses_active_shell_runtime(capsys: pytest.CaptureFixture[str]) -> None:
    runtime = _ShellRuntime()
    display = _Display()

    result = await InteractivePrompt()._execute_pending_shell_command(
        pending=PendingCommandExecution(shell_execute_cmd="pwd"),
        prompt_provider=cast("AgentApp", _Provider(runtime)),
        agent_name="agent",
        display=cast("ConsoleDisplay", display),
    )

    captured = capsys.readouterr()
    assert runtime.commands == []
    assert runtime.direct_commands == ["pwd"]
    assert result.exit_code == 0
    assert "remote output" in captured.out
    assert display.exit_codes == []


@pytest.mark.asyncio
async def test_local_environment_shell_command_uses_attached_terminal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _ShellRuntime(kind="local")
    display = _Display()

    result = await InteractivePrompt()._execute_pending_shell_command(
        pending=PendingCommandExecution(shell_execute_cmd="printf local-output"),
        prompt_provider=cast("AgentApp", _Provider(runtime)),
        agent_name="agent",
        display=cast("ConsoleDisplay", display),
    )

    captured = capsys.readouterr()
    assert runtime.commands == []
    assert runtime.direct_commands == []
    assert result.exit_code == 0
    assert result.stdout == "local-output"
    assert "$ printf local-output" in captured.out
    assert display.exit_codes == []


@pytest.mark.asyncio
async def test_bare_environment_shell_reports_interactive_unavailable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _ShellRuntime()
    display = _Display()

    result = await InteractivePrompt()._execute_pending_shell_command(
        pending=PendingCommandExecution(
            shell_execute_cmd="bash",
            shell_execute_interactive=True,
        ),
        prompt_provider=cast("AgentApp", _Provider(runtime)),
        agent_name="agent",
        display=cast("ConsoleDisplay", display),
    )

    captured = capsys.readouterr()
    assert result.exit_code == 1
    assert runtime.commands == []
    assert "Interactive shell is not available for hf-gpu" in captured.out
    assert "use `!!` to start a local shell" in captured.out
