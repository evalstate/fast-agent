from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING

import pytest
from mcp_types import TextContent

from fast_agent.config import Settings, ShellSettings
from fast_agent.tools.docker_shell_environment import DockerShellEnvironment
from fast_agent.tools.execution_environment import ShellExecutionRequest
from fast_agent.tools.shell_process import process_result_metadata
from fast_agent.tools.shell_runtime import ShellRuntime

if TYPE_CHECKING:
    from pathlib import Path

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


@pytest.mark.asyncio
async def test_docker_runtime_reads_host_retained_output_through_process(
    tmp_path: Path,
) -> None:
    container = os.getenv("FAST_AGENT_TEST_DOCKER_CONTAINER")
    if not container:
        pytest.skip("set FAST_AGENT_TEST_DOCKER_CONTAINER to a running container")

    environment = DockerShellEnvironment(
        container=container,
        shell=os.getenv("FAST_AGENT_TEST_DOCKER_SHELL", "sh"),
        cwd=os.getenv("FAST_AGENT_TEST_DOCKER_CWD", "/tmp"),
    )
    runtime = ShellRuntime(
        activation_reason="docker-retained-output-test",
        logger=logging.getLogger("docker-retained-output-test"),
        shell_environment=environment,
        output_byte_limit=256,
        config=Settings(
            shell_execution=ShellSettings(
                tool_profile="luna_exec",
                retain_truncated_output=True,
                retained_output_max_bytes=4096,
                retained_output_temp_directory=tmp_path,
            )
        ),
    )

    completed = await runtime.call_tool(
        "exec",
        {"command": ("python -c \"print('a'*300); print('retained-marker'); print('b'*300)\"")},
    )
    metadata = process_result_metadata(completed)
    assert metadata is not None
    completed_text = "\n".join(
        block.text for block in completed.content if isinstance(block, TextContent)
    )
    assert str(tmp_path) not in completed_text
    assert "action='read_output'" in completed_text

    readback = await runtime.call_tool(
        "process",
        {
            "process_id": metadata["process_id"],
            "action": "read_output",
            "query": "retained-marker",
        },
    )
    readback_text = "\n".join(
        block.text for block in readback.content if isinstance(block, TextContent)
    )
    payload = json.loads(readback_text)

    assert readback.is_error is False
    assert payload["match_count"] == 1
    assert payload["content"] == "retained-marker\n"

    await runtime.close()
