from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from acp.helpers import text_block
from acp.schema import StopReason

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

pytestmark = pytest.mark.asyncio(loop_scope="module")

END_TURN: StopReason = "end_turn"


@pytest.mark.integration
async def test_acp_initialize_and_prompt_roundtrip(
    acp_basic: tuple[object, TestClient, object],
) -> None:
    """Ensure the ACP transport initializes, creates a session, and echoes prompts."""
    connection, client, init_response = acp_basic

    assert init_response.protocol_version == 1
    assert init_response.agent_capabilities is not None
    assert init_response.agent_info.name == "fast-agent-acp-test"
    # AgentCapabilities schema changed upstream; ensure we advertised prompt support.
    prompt_caps = getattr(init_response.agent_capabilities, "prompts", None) or getattr(
        init_response.agent_capabilities, "prompt_capabilities", None
    )
    assert prompt_caps is not None

    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = session_response.session_id
    assert session_id

    prompt_text = "echo from ACP integration test"
    prompt_response = await connection.prompt(
        session_id=session_id, prompt=[text_block(prompt_text)]
    )
    assert prompt_response.stop_reason == END_TURN

    await _wait_for_notifications(client)

    # TestClient now stores notifications as dicts with session_id and update keys
    # Find the agent_message_chunk notification (may not be the last one due to commands update)
    # The update can be either an object with sessionUpdate attr or a dict with sessionUpdate key
    def get_session_update_type(update: Any) -> str | None:
        if hasattr(update, "sessionUpdate"):
            return update.sessionUpdate
        if isinstance(update, dict):
            return update.get("sessionUpdate")
        return None

    message_updates = [
        n
        for n in client.notifications
        if n["session_id"] == session_id
        and get_session_update_type(n["update"]) == "agent_message_chunk"
    ]
    assert message_updates, (
        f"Expected agent_message_chunk, got: {[get_session_update_type(n['update']) for n in client.notifications]}"
    )
    update = message_updates[-1]["update"]
    # Passthrough model mirrors user input, so the agent content should match the prompt.
    content = update.content if hasattr(update, "content") else update.get("content")
    assert getattr(content, "text", None) == prompt_text


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
async def test_acp_session_modes_included_in_new_session(
    acp_basic: tuple[object, TestClient, object],
) -> None:
    """Test that session/new response includes modes field."""
    connection, _client, init_response = acp_basic

    assert init_response.protocolVersion == 1

    # Create session
    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = session_response.session_id
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
    assert modes.currentModeId in available_mode_ids, "Current mode should be in available modes"


@pytest.mark.integration
async def test_acp_overlapping_prompts_are_refused(
    acp_basic: tuple[object, TestClient, object],
) -> None:
    """
    Test that overlapping prompt requests for the same session are refused.

    Per ACP protocol, only one prompt can be active per session at a time.
    If a second prompt arrives while one is in progress, it should be immediately
    refused with stopReason="refusal".
    """
    connection, _client, init_response = acp_basic

    assert init_response.protocolVersion == 1

    # Create session
    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = session_response.session_id
    assert session_id

    # Send two prompts truly concurrently (no sleep between them)
    # This ensures they both arrive before either completes
    prompt1_task = asyncio.create_task(
        connection.prompt(session_id=session_id, prompt=[text_block("first prompt")])
    )

    # Send immediately without waiting - ensures actual overlap
    prompt2_task = asyncio.create_task(
        connection.prompt(session_id=session_id, prompt=[text_block("overlapping prompt")])
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
        session_id=session_id, prompt=[text_block("third prompt")]
    )
    assert prompt3_response.stopReason == END_TURN
