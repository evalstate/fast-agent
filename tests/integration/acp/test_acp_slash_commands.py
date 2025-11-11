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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_slash_command_advertisement() -> None:
    """Test that available commands are advertised after session creation."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
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

        # Wait for available_commands_update notification
        await _wait_for_notifications(client)

        # Find the available_commands_update notification
        command_updates = [
            n for n in client.notifications
            if hasattr(n.update, "sessionUpdate")
            and n.update.sessionUpdate == "available_commands_update"
        ]

        assert len(command_updates) > 0, "Should receive available_commands_update notification"

        # Check that /status command is advertised
        available_commands = command_updates[0].update.availableCommands
        assert available_commands is not None
        assert len(available_commands) > 0

        # Find the status command
        status_cmd = next((cmd for cmd in available_commands if cmd["name"] == "status"), None)
        assert status_cmd is not None, "Status command should be advertised"
        assert "description" in status_cmd


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_status_slash_command() -> None:
    """Test that /status slash command returns session information."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
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

        # Clear notifications from session creation
        client.notifications.clear()

        # Send /status command
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block("/status")])
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for the status response
        await _wait_for_notifications(client)

        # Check the response contains status information
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id

        # Get the text content
        content_text = getattr(last_update.update.content, "text", None)
        assert content_text is not None

        # Check for expected status fields
        assert "Version:" in content_text
        assert "Model:" in content_text
        assert "Turns:" in content_text
        assert "Tool Calls:" in content_text
        assert "Context Usage:" in content_text
        assert session_id in content_text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_status_updates_after_prompt() -> None:
    """Test that /status command shows updated turn count after prompts."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
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

        # Send a regular prompt first
        await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block("test prompt")])
        )
        await _wait_for_notifications(client)

        # Clear notifications
        client.notifications.clear()

        # Now send /status command
        await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block("/status")])
        )
        await _wait_for_notifications(client)

        # Check the response
        last_update = client.notifications[-1]
        content_text = getattr(last_update.update.content, "text", None)
        assert content_text is not None

        # Should show at least 1 turn
        assert "Turns: 1" in content_text or "Turns: 2" in content_text


async def _wait_for_notifications(client: TestClient, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive at least one sessionUpdate."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if client.notifications:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Expected streamed session updates")
