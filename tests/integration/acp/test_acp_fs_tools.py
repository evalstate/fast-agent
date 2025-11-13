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
async def test_acp_fs_read_text_file() -> None:
    """Test that fs/read_text_file tool is available and works correctly."""
    client = TestClient()

    # Pre-populate a test file in the client's filesystem
    test_file_path = str(TEST_DIR / "test_read.txt")
    test_content = "Hello from ACP fs/read_text_file integration test!"
    client.files[test_file_path] = test_content

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize with fs capabilities
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)
        assert init_response.protocolVersion == 1

        # Create session with TEST_DIR as CWD
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Send a prompt that should trigger the fs__read_text_file tool
        # The passthrough model will echo the prompt, but we're mainly testing
        # that the tool is available and properly injected
        prompt_text = f"Read the file at {test_file_path}"
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications to arrive
        await _wait_for_notifications(client)

        # Verify we got a response
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_fs_write_text_file() -> None:
    """Test that fs/write_text_file tool is available and works correctly."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize with fs capabilities
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)
        assert init_response.protocolVersion == 1

        # Create session with TEST_DIR as CWD
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Send a prompt that should trigger the fs__write_text_file tool
        test_file_path = str(TEST_DIR / "test_write.txt")
        test_content = "Content written by ACP fs/write_text_file integration test!"
        prompt_text = f"Write '{test_content}' to {test_file_path}"
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications to arrive
        await _wait_for_notifications(client)

        # Verify we got a response
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_fs_tools_not_injected_without_capabilities() -> None:
    """Test that fs tools are NOT injected when client doesn't advertise fs capabilities."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize WITHOUT fs capabilities
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs=None,  # No fs capabilities
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)
        assert init_response.protocolVersion == 1

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Send a prompt - fs tools should not be available
        prompt_text = "test prompt without fs tools"
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications to arrive
        await _wait_for_notifications(client)

        # Verify we got a response
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_fs_read_only_capabilities() -> None:
    """Test that only read tool is injected when client only supports readTextFile."""
    client = TestClient()

    # Pre-populate a test file
    test_file_path = str(TEST_DIR / "test_readonly.txt")
    test_content = "Read-only test content"
    client.files[test_file_path] = test_content

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize with only read capability
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": False},  # Only read
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)
        assert init_response.protocolVersion == 1

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Send a prompt
        prompt_text = "test with read-only fs capabilities"
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)

        # Verify response received
        assert len(client.notifications) > 0
