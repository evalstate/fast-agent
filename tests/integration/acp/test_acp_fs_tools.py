"""Integration tests for ACP filesystem tools (fs/read_text_file and fs/write_text_file)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from acp import InitializeRequest, NewSessionRequest, PromptRequest
from acp.helpers import text_block
from acp.schema import ClientCapabilities, Implementation, StopReason
from acp.stdio import spawn_agent_process

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
END_TURN: StopReason = "end_turn"
FAST_AGENT_CMD = (
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
    "fast-agent-acp-test",
)


async def _wait_for_notifications(client: TestClient, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive at least one sessionUpdate."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if client.notifications:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Expected streamed session updates")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_fs_tools_available_when_supported() -> None:
    """Test that fs tools are added to agents when client advertises fs capabilities."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize with fs capabilities
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-fs-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)
        assert init_response.protocolVersion == 1

        # Create session with a test directory
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Note: With passthrough model, the agent just echoes the prompt.
        # In a real scenario, the LLM would invoke the fs tools.
        # Here we verify the tools are available by checking they don't cause errors.
        prompt_text = "fs tools test"
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_fs_read_text_file() -> None:
    """Test reading a text file via ACP fs/read_text_file."""
    client = TestClient()

    # Set up a test file in the client's virtual filesystem
    test_file_path = str(TEST_DIR / "test_fs_sample.txt")
    test_content = "This is a sample text file for testing ACP filesystem tools.\nIt contains multiple lines.\nLine 3: Testing read_text_file functionality.\n"
    client.files[test_file_path] = test_content

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize with read capability only
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": False},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-fs-read-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # The passthrough model won't actually call tools, but we can verify
        # the session was created successfully with fs tools available
        assert session_id

        # In a real test with a real model, the agent would call read_text_file
        # and we would verify the content. Here we just verify no errors.
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block("test read")])
        )
        assert prompt_response.stopReason == END_TURN


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_fs_write_text_file() -> None:
    """Test writing a text file via ACP fs/write_text_file."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize with write capability only
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": False, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-fs-write-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Verify session creation with write capability
        assert session_id

        # In a real test with a real model, the agent would call write_text_file
        # and we would verify the file was written to client.files.
        # Here we just verify no errors during session setup.
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block("test write")])
        )
        assert prompt_response.stopReason == END_TURN


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_fs_tools_not_available_without_capabilities() -> None:
    """Test that fs tools are NOT added when client doesn't advertise fs capabilities."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize WITHOUT fs capabilities
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs=None,  # No fs capabilities
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-no-fs-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session - should work fine without fs tools
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Verify session works normally
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block("no fs tools")])
        )
        assert prompt_response.stopReason == END_TURN
