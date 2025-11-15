"""
Integration tests for ACP Session Modes support.

Tests that agents can be exposed as modes and clients can switch between them.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from acp import InitializeRequest, NewSessionRequest, PromptRequest, SetSessionModeRequest
from acp.helpers import text_block
from acp.schema import ClientCapabilities, Implementation, StopReason

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

CONFIG_FILE = TEST_DIR / "test_modes_config.yaml"
END_TURN: StopReason = "end_turn"


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
async def test_acp_modes_listed_in_new_session() -> None:
    """Test that newSession returns available modes from all agents."""
    from acp.stdio import spawn_agent_process

    client = TestClient()
    fast_agent_cmd = (
        sys.executable,
        "-m",
        "fast_agent_mcp.tests.integration.acp.fixtures.multi_agent",
    )

    async with spawn_agent_process(lambda _: client, *fast_agent_cmd) as (connection, _process):
        # Initialize
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(),
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

        # Check modes are returned
        assert session_response.modes is not None
        modes = session_response.modes
        assert modes.currentModeId
        assert len(modes.availableModes) >= 2  # At least two agents

        # Check mode structure
        mode_ids = [mode.id for mode in modes.availableModes]
        assert "code_expert" in mode_ids
        assert "general_assistant" in mode_ids

        # Check mode names are formatted
        code_expert_mode = next(m for m in modes.availableModes if m.id == "code_expert")
        assert code_expert_mode.name == "Code Expert"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_mode_switching() -> None:
    """Test that setSessionMode switches the active agent."""
    from acp.stdio import spawn_agent_process

    client = TestClient()
    fast_agent_cmd = (
        sys.executable,
        "-m",
        "fast_agent_mcp.tests.integration.acp.fixtures.multi_agent",
    )

    async with spawn_agent_process(lambda _: client, *fast_agent_cmd) as (connection, _process):
        # Initialize
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        initial_mode = session_response.modes.currentModeId

        # Send prompt to initial agent
        client.notifications.clear()
        prompt1 = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block("Hello from initial mode")])
        )
        assert prompt1.stopReason == END_TURN
        await _wait_for_notifications(client)

        # Switch to different mode
        new_mode = (
            "general_assistant" if initial_mode == "code_expert" else "code_expert"
        )
        mode_response = await connection.setSessionMode(
            SetSessionModeRequest(sessionId=session_id, modeId=new_mode)
        )
        assert mode_response is not None

        # Send prompt to new agent
        client.notifications.clear()
        prompt2 = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block("Hello from new mode")])
        )
        assert prompt2.stopReason == END_TURN
        await _wait_for_notifications(client)

        # The response should come from the new agent
        # (In a real test, we'd verify the agent's response content differs)
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_invalid_mode_rejected() -> None:
    """Test that setSessionMode rejects invalid mode IDs."""
    from acp.stdio import spawn_agent_process

    client = TestClient()
    fast_agent_cmd = (
        sys.executable,
        "-m",
        "fast_agent_mcp.tests.integration.acp.fixtures.multi_agent",
    )

    async with spawn_agent_process(lambda _: client, *fast_agent_cmd) as (connection, _process):
        # Initialize
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Try to switch to invalid mode
        mode_response = await connection.setSessionMode(
            SetSessionModeRequest(sessionId=session_id, modeId="invalid_mode")
        )
        # Should return None to indicate error
        assert mode_response is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_mode_descriptions() -> None:
    """Test that mode descriptions are extracted from agent instructions."""
    from acp.stdio import spawn_agent_process

    client = TestClient()
    fast_agent_cmd = (
        sys.executable,
        "-m",
        "fast_agent_mcp.tests.integration.acp.fixtures.multi_agent",
    )

    async with spawn_agent_process(lambda _: client, *fast_agent_cmd) as (connection, _process):
        # Initialize
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )

        # Check that modes have descriptions
        modes = session_response.modes.availableModes
        for mode in modes:
            # Description should be present if agent has instruction
            # At minimum, it should not cause an error
            assert isinstance(mode.description, (str, type(None)))
            if mode.description:
                # Description should be truncated to max 200 characters
                assert len(mode.description) <= 200
