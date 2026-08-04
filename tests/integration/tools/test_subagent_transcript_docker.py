from __future__ import annotations

import os

import pytest

from fast_agent.tools.docker_shell_environment import (
    DockerMountedEnvironment,
    DockerShellEnvironment,
)
from fast_agent.tools.execution_environment import ShellExecutionRequest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_existing_docker_container_exposes_temporary_artifact_to_files_and_shell() -> None:
    container = os.getenv("FAST_AGENT_TEST_DOCKER_CONTAINER")
    if not container:
        pytest.skip("set FAST_AGENT_TEST_DOCKER_CONTAINER to a running container")

    environment = DockerShellEnvironment(
        container=container,
        shell=os.getenv("FAST_AGENT_TEST_DOCKER_SHELL", "sh"),
        cwd=os.getenv("FAST_AGENT_TEST_DOCKER_CWD", "/tmp"),
    )
    artifact = await environment.write_temporary_text(
        prefix="fast-agent-subagent-",
        suffix=".log",
        content="FAST_AGENT_DOCKER_TRANSCRIPT_MARKER\n",
        max_bytes=2 * 1024 * 1024,
    )
    assert await environment.read_text(artifact.path) == "FAST_AGENT_DOCKER_TRANSCRIPT_MARKER\n"
    search = await environment.execute(
        ShellExecutionRequest(
            command=f"grep -n FAST_AGENT_DOCKER_TRANSCRIPT_MARKER -- {artifact.path}"
        )
    )
    assert search.result.exit_code == 0

    await environment.remove_temporary_artifact(artifact)
    assert not await environment.exists(artifact.path)
    await environment.close()


@pytest.mark.asyncio
async def test_mounted_docker_routes_owned_temporary_artifact_to_container(tmp_path) -> None:
    image = os.getenv("FAST_AGENT_TEST_DOCKER_IMAGE")
    if not image:
        pytest.skip("set FAST_AGENT_TEST_DOCKER_IMAGE to a local Linux container image")

    environment = DockerMountedEnvironment(
        image=image,
        workspace=tmp_path,
        shell=os.getenv("FAST_AGENT_TEST_DOCKER_SHELL", "sh"),
    )
    await environment.open()
    try:
        artifact = await environment.write_temporary_text(
            prefix="fast-agent-subagent-",
            suffix=".log",
            content="FAST_AGENT_MOUNTED_TRANSCRIPT_MARKER\n",
            max_bytes=2 * 1024 * 1024,
        )
        assert (
            await environment.read_text(artifact.path) == "FAST_AGENT_MOUNTED_TRANSCRIPT_MARKER\n"
        )
        search = await environment.execute(
            ShellExecutionRequest(
                command=f"grep -n FAST_AGENT_MOUNTED_TRANSCRIPT_MARKER -- {artifact.path}"
            )
        )
        assert search.result.exit_code == 0
    finally:
        await environment.close()
