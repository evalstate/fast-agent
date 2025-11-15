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

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
MULTI_AGENT_SCRIPT = TEST_DIR / "multi_agent_test.py"
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
MULTI_AGENT_CMD = (
    sys.executable,
    str(MULTI_AGENT_SCRIPT),
    "--transport",
    "acp",
    "--model",
    "passthrough",
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_initialize_and_prompt_roundtrip() -> None:
    """Ensure the ACP transport initializes, creates a session, and echoes prompts."""
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
        init_response = await connection.initialize(init_request)

        assert init_response.protocolVersion == 1
        assert init_response.agentCapabilities is not None
        assert init_response.agentInfo.name == "fast-agent-acp-test"
        # AgentCapabilities schema changed upstream; ensure we advertised prompt support.
        prompt_caps = getattr(init_response.agentCapabilities, "prompts", None) or getattr(
            init_response.agentCapabilities, "promptCapabilities", None
        )
        assert prompt_caps is not None

        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        prompt_text = "echo from ACP integration test"
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN

        await _wait_for_notifications(client)
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id
        assert last_update.update.sessionUpdate == "agent_message_chunk"
        # Passthrough model mirrors user input, so the agent content should match the prompt.
        assert getattr(last_update.update.content, "text", None) == prompt_text


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
async def test_acp_session_modes_included_in_new_session() -> None:
    """Test that session/new response includes modes field."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
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

        # Verify modes are included in the response
        assert hasattr(session_response, "modes"), "NewSessionResponse should include modes field"
        assert session_response.modes is not None, "Modes should not be None"

        # Verify modes structure
        modes = session_response.modes
        assert hasattr(modes, "availableModes"), "SessionModeState should have availableModes"
        assert hasattr(modes, "currentModeId"), "SessionModeState should have currentModeId"
        assert len(modes.availableModes) > 0, "Should have at least one available mode"
        assert modes.currentModeId, "Should have a current mode set"

        # Verify the current mode is in available modes
        available_mode_ids = [mode.id for mode in modes.availableModes]
        assert modes.currentModeId in available_mode_ids, (
            "Current mode should be in available modes"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_set_session_mode_basic() -> None:
    """Test that setSessionMode endpoint accepts valid mode changes."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *MULTI_AGENT_CMD) as (connection, _process):
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

        # Verify we have multiple modes available
        modes = session_response.modes
        assert modes is not None
        assert len(modes.availableModes) >= 2, "Multi-agent app should have at least 2 modes"

        # Find a mode different from the current one
        target_mode = None
        for mode in modes.availableModes:
            if mode.id != modes.currentModeId:
                target_mode = mode.id
                break

        assert target_mode is not None, "Should have at least one alternative mode"

        # Switch to the alternative mode
        set_mode_response = await connection.setSessionMode(
            SetSessionModeRequest(sessionId=session_id, modeId=target_mode)
        )
        # Should return successfully without raising an exception
        assert set_mode_response is not None

        # Optionally send a prompt to verify it doesn't crash after mode switch
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block("test after mode switch")])
        )
        assert prompt_response.stopReason == END_TURN


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_set_session_mode_invalid_mode() -> None:
    """Test that setSessionMode rejects invalid mode IDs gracefully."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *MULTI_AGENT_CMD) as (connection, _process):
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

        # Try to switch to a non-existent mode
        # This should raise an exception (graceful error handling)
        with pytest.raises(Exception):  # ValueError or ACP protocol error
            await connection.setSessionMode(
                SetSessionModeRequest(sessionId=session_id, modeId="nonexistent_agent_mode")
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_overlapping_prompts_are_refused() -> None:
    """
    Test that overlapping prompt requests for the same session are refused.

    Per ACP protocol, only one prompt can be active per session at a time.
    If a second prompt arrives while one is in progress, it should be immediately
    refused with stopReason="refusal".
    """
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
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

        # Send two prompts truly concurrently (no sleep between them)
        # This ensures they both arrive before either completes
        prompt1_task = asyncio.create_task(
            connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block("first prompt")])
            )
        )

        # Send immediately without waiting - ensures actual overlap
        prompt2_task = asyncio.create_task(
            connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block("overlapping prompt")])
            )
        )

        # Wait for both to complete
        prompt1_response, prompt2_response = await asyncio.gather(prompt1_task, prompt2_task)

        # One should succeed, one should be refused
        # (We don't know which one arrives first due to async scheduling)
        responses = [prompt1_response.stopReason, prompt2_response.stopReason]
        assert "end_turn" in responses, "One prompt should succeed"
        assert "refusal" in responses, "One prompt should be refused"

        # After both complete, a new prompt should succeed
        prompt3_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block("third prompt")])
        )
        assert prompt3_response.stopReason == END_TURN
