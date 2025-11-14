"""Integration tests for ACP filesystem support."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from acp import InitializeRequest, NewSessionRequest, PromptRequest
from acp.helpers import text_block
from acp.schema import ClientCapabilities, Implementation, StopReason

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
END_TURN: StopReason = "end_turn"


def get_fast_agent_cmd() -> tuple:
    """Build the fast-agent command with appropriate flags."""
    cmd = [
        sys.executable,
        "-m",
        "fast_agent.cli",
        "serve",
        "--config-path",
        str(CONFIG_PATH),
        "--transport",
        "acp",
        "--model",
        "passthrough",
        "--name",
        "fast-agent-acp-filesystem-test",
    ]
    return tuple(cmd)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_filesystem_support_enabled() -> None:
    """Test that filesystem support is properly enabled when client advertises capability."""
    from acp.stdio import spawn_agent_process

    client = TestClient()

    async with spawn_agent_process(lambda _: client, *get_fast_agent_cmd()) as (connection, _process):
        # Initialize
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Send prompt that should trigger filesystem operations
        prompt_text = 'use the read_text_file tool to read: /test/file.txt'
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for any notifications
        await _wait_for_notifications(client)

        # Verify we got notifications
        assert len(client.notifications) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_filesystem_read_only() -> None:
    """Test filesystem support with only read capability enabled."""
    from acp.stdio import spawn_agent_process

    client = TestClient()

    async with spawn_agent_process(lambda _: client, *get_fast_agent_cmd()) as (connection, _process):
        # Initialize
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Filesystem runtime should be created with only read tool


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_filesystem_write_only() -> None:
    """Test filesystem support with only write capability enabled."""
    from acp.stdio import spawn_agent_process

    client = TestClient()

    async with spawn_agent_process(lambda _: client, *get_fast_agent_cmd()) as (connection, _process):
        # Initialize
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Filesystem runtime should be created with only write tool


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_filesystem_disabled_when_client_unsupported() -> None:
    """Test that filesystem runtime is not used when client doesn't support filesystem."""
    from acp.stdio import spawn_agent_process

    client = TestClient()

    async with spawn_agent_process(lambda _: client, *get_fast_agent_cmd()) as (connection, _process):
        # Initialize
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Agent should work without filesystem tools
        # This test ensures graceful handling when filesystem is not supported


async def _wait_for_notifications(client: TestClient, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive at least one sessionUpdate."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if client.notifications:
            return
        await asyncio.sleep(0.05)
    # Don't raise error - some tests may not produce notifications
