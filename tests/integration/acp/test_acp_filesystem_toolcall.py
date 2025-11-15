"""Integration tests for ACP filesystem tool calling."""

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
        "passthrough",  # Use passthrough model for deterministic testing
        "--name",
        "fast-agent-acp-filesystem-toolcall-test",
    ]
    return tuple(cmd)


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
async def test_acp_filesystem_read_tool_call() -> None:
    """Test that read_text_file tool can be called via passthrough model."""
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

        # Set up a test file in the client
        test_path = "/test/sample.txt"
        test_content = "Hello from test file!"
        client.files[test_path] = test_content

        # Use passthrough model's ***CALL_TOOL directive to invoke read_text_file
        prompt_text = f'***CALL_TOOL read_text_file {{"path": "{test_path}"}}'
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )

        # Should complete successfully
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)

        # Verify we got notifications
        assert len(client.notifications) > 0

        # Verify the file content appears in the notifications
        # This confirms the read_text_file tool was called and returned content
        notification_text = str(client.notifications)
        assert test_content in notification_text or test_path in notification_text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_filesystem_write_tool_call() -> None:
    """Test that write_text_file tool can be called via passthrough model."""
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

        # Use passthrough model's ***CALL_TOOL directive to invoke write_text_file
        test_path = "/test/output.txt"
        test_content = "Test content from tool call"
        prompt_text = f'***CALL_TOOL write_text_file {{"path": "{test_path}", "content": "{test_content}"}}'

        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )

        # Should complete successfully
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)

        # Verify the file was written
        assert test_path in client.files
        assert client.files[test_path] == test_content
