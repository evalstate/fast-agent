from __future__ import annotations

import asyncio
import os

import pytest

from fast_agent.tools.docker_shell_environment import DockerShellEnvironment
from fast_agent.tools.execution_environment import ShellExecutionRequest

pytestmark = pytest.mark.integration


class _OutputCallbacks:
    def __init__(self) -> None:
        self.output_seen = asyncio.Event()
        self.stdout: list[str] = []
        self.stderr: list[str] = []

    async def on_started(self, process_id: int | None) -> None:
        assert process_id is not None

    async def on_stdout(self, text: str) -> None:
        self.stdout.append(text)
        self.output_seen.set()

    async def on_stderr(self, text: str) -> None:
        self.stderr.append(text)
        self.output_seen.set()

    async def on_idle_warning(self, elapsed: float, remaining: float) -> None:
        del elapsed, remaining

    async def on_timeout(self) -> None:
        return None


@pytest.mark.asyncio
async def test_detached_docker_process_streams_spooled_output() -> None:
    container = os.getenv("FAST_AGENT_TEST_DOCKER_CONTAINER")
    if not container:
        pytest.skip("set FAST_AGENT_TEST_DOCKER_CONTAINER to a running container")

    environment = DockerShellEnvironment(
        container=container,
        shell=os.getenv("FAST_AGENT_TEST_DOCKER_SHELL", "sh"),
        cwd=os.getenv("FAST_AGENT_TEST_DOCKER_CWD", "/tmp"),
    )
    request = ShellExecutionRequest(
        command="printf 'docker stdout\\n'; printf 'docker stderr\\n' >&2; sleep 30",
        terminate_after_idle=False,
        retain_output=False,
        terminate_on_cancel=False,
        detach=True,
    )
    callbacks = _OutputCallbacks()
    task = asyncio.create_task(environment.execute(request, callbacks=callbacks))

    try:
        await asyncio.wait_for(callbacks.output_seen.wait(), timeout=10)
        deadline = asyncio.get_running_loop().time() + 10
        while (
            not callbacks.stdout or not callbacks.stderr
        ) and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.1)

        assert "docker stdout\n" in "".join(callbacks.stdout)
        assert "docker stderr\n" in "".join(callbacks.stderr)
        assert request.output_spool_path is not None
    finally:
        request.terminate_on_cancel = True
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert request.output_spool_path is None
