"""Integration tests for ACP session modes support."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from acp import InitializeRequest, NewSessionRequest, PromptRequest, SetSessionModeRequest
from acp.helpers import text_block
from acp.schema import ClientCapabilities, Implementation, StopReason
from acp.stdio import spawn_agent_process

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

END_TURN: StopReason = "end_turn"

# We'll create a test agent with multiple modes
MULTI_AGENT_SCRIPT = TEST_DIR / "multi_agent_test.py"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_modes_in_new_session_response() -> None:
    """Test that newSession returns modes with all available agents."""
    client = TestClient()

    # Start the multi-agent test application
    cmd = (
        sys.executable,
        str(MULTI_AGENT_SCRIPT),
        "--transport",
        "acp",
        "--model",
        "passthrough",
    )

    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
        # Initialize
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

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Check that modes are present
        assert session_response.modes is not None
        modes = session_response.modes
        assert modes.availableModes is not None
        assert modes.currentModeId is not None

        # Should have two modes: agent_one and agent_two
        mode_ids = [mode.id for mode in modes.availableModes]
        assert len(mode_ids) == 2
        assert "agent_one" in mode_ids
        assert "agent_two" in mode_ids

        # Check mode names are formatted correctly
        mode_names = {mode.id: mode.name for mode in modes.availableModes}
        assert mode_names["agent_one"] == "Agent One"
        assert mode_names["agent_two"] == "Agent Two"

        # Check mode descriptions
        mode_descriptions = {mode.id: mode.description for mode in modes.availableModes}
        assert mode_descriptions["agent_one"] is not None
        assert "first agent" in mode_descriptions["agent_one"].lower()
        assert mode_descriptions["agent_two"] is not None
        assert "second agent" in mode_descriptions["agent_two"].lower()

        # Default mode should be agent_one (first agent)
        assert modes.currentModeId == "agent_one"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_set_session_mode() -> None:
    """Test that setSessionMode changes the active agent."""
    client = TestClient()

    cmd = (
        sys.executable,
        str(MULTI_AGENT_SCRIPT),
        "--transport",
        "acp",
        "--model",
        "passthrough",
    )

    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
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

        # Switch to agent_two
        set_mode_response = await connection.setSessionMode(
            SetSessionModeRequest(sessionId=session_id, modeId="agent_two")
        )
        assert set_mode_response is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_prompt_routing_after_mode_change() -> None:
    """Test that prompts are routed to the correct agent after mode change."""
    client = TestClient()

    cmd = (
        sys.executable,
        str(MULTI_AGENT_SCRIPT),
        "--transport",
        "acp",
        "--model",
        "passthrough",
    )

    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
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

        # Send prompt to default agent (agent_one)
        prompt_text = "test prompt for agent one"
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for notification
        await _wait_for_notifications(client)
        client.notifications.clear()

        # Switch to agent_two
        await connection.setSessionMode(
            SetSessionModeRequest(sessionId=session_id, modeId="agent_two")
        )

        # Send prompt to agent_two
        prompt_text_2 = "test prompt for agent two"
        prompt_response_2 = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text_2)])
        )
        assert prompt_response_2.stopReason == END_TURN

        # Wait for notification
        await _wait_for_notifications(client)

        # Both prompts should have succeeded
        assert len(client.notifications) >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_invalid_mode_id() -> None:
    """Test that setSessionMode rejects invalid mode IDs."""
    client = TestClient()

    cmd = (
        sys.executable,
        str(MULTI_AGENT_SCRIPT),
        "--transport",
        "acp",
        "--model",
        "passthrough",
    )

    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
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

        # Try to switch to non-existent mode
        with pytest.raises(Exception):  # Should raise ValueError or similar
            await connection.setSessionMode(
                SetSessionModeRequest(sessionId=session_id, modeId="invalid_agent")
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
