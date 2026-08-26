import json
import logging
import os
from pathlib import Path

import pytest

from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers.display import handle_processes
from fast_agent.commands.results import CommandMessage
from fast_agent.session.session_manager import SessionManager
from fast_agent.tools.durable_processes import DurableProcessSnapshot, DurableProcessStore
from fast_agent.tools.shell_runtime import ManagedProcessSnapshot, ShellRuntime

_SHELL = Path("/bin/sh")


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
                lifecycle="session",
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
                lifecycle="session",
            ),
        )

    async def discover_durable_processes(self) -> tuple[DurableProcessSnapshot, ...]:
        return ()

    async def attach_durable_process(
        self,
        process_id: str,
        *,
        session_id: str | None = None,
    ) -> DurableProcessSnapshot:
        del process_id, session_id
        raise AssertionError("not expected")


class _Agent:
    def __init__(self, shell_runtime: _Runtime | ShellRuntime | None = None) -> None:
        self.shell_runtime = shell_runtime or _Runtime()


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


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_handle_process_attach_falls_back_to_acp_session_id(tmp_path: Path) -> None:
    root = tmp_path / "processes"
    store = DurableProcessStore(root)
    created = store.create(command="exit 0", shell=_SHELL, cwd=tmp_path)
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )
    context = CommandContext(
        agent_provider=StaticAgentProvider({"main": _Agent(runtime)}),
        current_agent_name="main",
        io=_IO(),
        acp_session_id="acp-session-1",
        session_manager=SessionManager(
            cwd=tmp_path,
            home_override=tmp_path / "home",
        ),
    )

    try:
        outcome = await handle_processes(
            context,
            agent_name="main",
            attach_process_id=created.spec.process_id,
        )
    finally:
        await runtime.close()

    assert outcome.messages[0].channel == "system"
    assert store.get(created.spec.process_id).session_ids == ("acp-session-1",)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_handle_process_terminate_requests_stop_without_attaching(tmp_path: Path) -> None:
    root = tmp_path / "processes"
    store = DurableProcessStore(root)
    created = store.create(command="sleep 30", shell=_SHELL, cwd=tmp_path)
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )
    context = CommandContext(
        agent_provider=StaticAgentProvider({"main": _Agent(runtime)}),
        current_agent_name="main",
        io=_IO(),
        session_manager=SessionManager(
            cwd=tmp_path,
            home_override=tmp_path / "home",
        ),
    )

    try:
        outcome = await handle_processes(
            context,
            agent_name="main",
            terminate_process_id=created.spec.process_id,
        )
        attached = await runtime.process_snapshots()
    finally:
        await runtime.close()

    assert outcome.messages[0].channel == "system"
    assert "Termination requested" in outcome.messages[0].plain_text()
    assert store.request_stop(created.spec.process_id) is False
    assert attached == ()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_handle_process_terminate_refuses_unavailable_record(tmp_path: Path) -> None:
    root = tmp_path / "processes"
    store = DurableProcessStore(root)
    created = store.create(command="sleep 30", shell=_SHELL, cwd=tmp_path)
    status_path = store.directory(created.spec.process_id) / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "state": "running",
            "supervisor_pid": 99_999_999,
            "child_pid": 99_999_998,
        }
    )
    status_path.write_text(json.dumps(status), encoding="utf-8")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=root,
    )
    context = CommandContext(
        agent_provider=StaticAgentProvider({"main": _Agent(runtime)}),
        current_agent_name="main",
        io=_IO(),
    )

    try:
        outcome = await handle_processes(
            context,
            agent_name="main",
            terminate_process_id=created.spec.process_id,
        )
    finally:
        await runtime.close()

    assert outcome.messages[0].channel == "error"
    assert "no stop request was sent" in outcome.messages[0].plain_text()
    assert not (store.directory(created.spec.process_id) / "control").exists()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_handle_process_terminate_rejects_invalid_process_id(tmp_path: Path) -> None:
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        durable_process_root=tmp_path / "processes",
    )
    context = CommandContext(
        agent_provider=StaticAgentProvider({"main": _Agent(runtime)}),
        current_agent_name="main",
        io=_IO(),
    )

    try:
        outcome = await handle_processes(
            context,
            agent_name="main",
            terminate_process_id="process-1",
        )
    finally:
        await runtime.close()

    assert outcome.messages[0].channel == "error"
    assert "was not found" in outcome.messages[0].plain_text()
