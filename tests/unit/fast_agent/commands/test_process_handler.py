import pytest

from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers.display import handle_processes
from fast_agent.commands.results import CommandMessage
from fast_agent.tools.shell_runtime import ManagedProcessSnapshot


class _Runtime:
    async def process_snapshots(self) -> tuple[ManagedProcessSnapshot, ...]:
        return (
            ManagedProcessSnapshot(
                process_id="process-1",
                command="python -c 'import time; time.sleep(30)'",
                working_directory="/app",
                status="running",
                elapsed_seconds=12.4,
                os_process_id=4321,
                total_output_bytes=0,
                exit_code=None,
            ),
            ManagedProcessSnapshot(
                process_id="process-2",
                command="echo finished",
                working_directory="/app",
                status="completed",
                elapsed_seconds=0.2,
                os_process_id=4322,
                total_output_bytes=9,
                exit_code=0,
            ),
        )


class _Agent:
    shell_runtime = _Runtime()


class _IO(NonInteractiveCommandIOBase):
    async def emit(self, message: CommandMessage) -> None:
        del message


@pytest.mark.asyncio
async def test_handle_processes_renders_active_and_retained_summary() -> None:
    context = CommandContext(
        agent_provider=StaticAgentProvider({"main": _Agent()}),
        current_agent_name="main",
        io=_IO(),
        no_home=True,
    )

    outcome = await handle_processes(context, agent_name="main")

    assert len(outcome.messages) == 1
    message = outcome.messages[0]
    assert message.render_markdown is True
    text = message.plain_text()
    assert "# active managed processes" in text
    assert "↻ **1 active**" in text
    assert "`process-1` | running |" in text
    assert "4321" in text
    assert "process-2" not in text


@pytest.mark.asyncio
async def test_handle_processes_history_renders_only_finished_processes() -> None:
    context = CommandContext(
        agent_provider=StaticAgentProvider({"main": _Agent()}),
        current_agent_name="main",
        io=_IO(),
        no_home=True,
    )

    outcome = await handle_processes(
        context,
        agent_name="main",
        show_history=True,
    )

    text = outcome.messages[0].plain_text()
    assert "# finished managed processes" in text
    assert "**1 finished** · 2 retained" in text
    assert "`process-2` | completed (0) |" in text
    assert "process-1" not in text
