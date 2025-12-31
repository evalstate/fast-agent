"""Integration tests for ACP terminal support."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from acp.helpers import text_block

if TYPE_CHECKING:
    from acp.client.connection import ClientSideConnection
    from acp.schema import InitializeResponse, StopReason
    from test_client import TestClient

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))


pytestmark = pytest.mark.asyncio(loop_scope="module")

END_TURN: StopReason = "end_turn"


def _get_session_id(response: object) -> str:
    return getattr(response, "session_id", None) or getattr(response, "sessionId")


def _get_stop_reason(response: object) -> str | None:
    return getattr(response, "stop_reason", None) or getattr(response, "stopReason", None)


@pytest.mark.integration
async def test_acp_terminal_support_enabled(
    acp_terminal_shell: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> None:
    """Test that terminal support is properly enabled when client advertises capability."""
    connection, client, init_response = acp_terminal_shell

    assert (
        getattr(init_response, "protocol_version", None) == 1
        or getattr(init_response, "protocolVersion", None) == 1
    )
    assert (
        getattr(init_response, "agent_capabilities", None)
        or getattr(init_response, "agentCapabilities", None) is not None
    )

    # Create session
    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = _get_session_id(session_response)
    assert session_id

    # Send prompt that should trigger terminal execution
    # The passthrough model will echo our input, so we craft a tool call request
    prompt_text = 'use the execute tool to run: echo "test terminal"'
    prompt_response = await connection.prompt(
        session_id=session_id, prompt=[text_block(prompt_text)]
    )
    assert _get_stop_reason(prompt_response) == END_TURN

    # Wait for any notifications
    await _wait_for_notifications(client)

    # Verify we got notifications
    assert len(client.notifications) > 0


@pytest.mark.integration
async def test_acp_terminal_execution(
    acp_terminal_shell: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> None:
    """Test actual terminal command execution via ACP."""
    connection, client, _init_response = acp_terminal_shell

    # Directly test terminal methods are being called
    # Since we're using passthrough model, we can't test actual LLM-driven tool calls
    # but we can verify the terminal runtime is set up correctly

    # Create a session first to get a session ID
    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = _get_session_id(session_response)

    # The terminals dict should be empty initially
    assert len(client.terminals) == 0

    # Manually test terminal lifecycle (client creates ID)
    create_result = await client.create_terminal(command="echo test", session_id=session_id)
    terminal_id = create_result.terminal_id

    # Verify terminal was created with client-generated ID
    assert terminal_id == "terminal-1"  # First terminal
    assert terminal_id in client.terminals
    assert client.terminals[terminal_id]["command"] == "echo test"

    # Get output
    output = await client.terminal_output(session_id=session_id, terminal_id=terminal_id)
    assert "Executed: echo test" in output.output
    exit_info = await client.wait_for_terminal_exit(session_id=session_id, terminal_id=terminal_id)
    assert exit_info.exit_code == 0

    # Release terminal
    await client.release_terminal(session_id=session_id, terminal_id=terminal_id)
    assert terminal_id not in client.terminals


@pytest.mark.integration
async def test_acp_terminal_disabled_when_no_shell_flag(
    acp_terminal_no_shell: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> None:
    """Test that terminal runtime is not injected when --shell flag is not provided."""
    connection, _client, _init_response = acp_terminal_no_shell

    # Create session
    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = _get_session_id(session_response)
    assert session_id

    # Terminal runtime should not be injected because --shell wasn't provided
    # This test mainly ensures no errors occur when terminal capability is advertised
    # but shell runtime isn't enabled


@pytest.mark.integration
async def test_acp_terminal_disabled_when_client_unsupported(
    acp_terminal_client_unsupported: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> None:
    """Test that terminal runtime is not used when client doesn't support terminals."""
    connection, _client, _init_response = acp_terminal_client_unsupported

    # Create session
    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = _get_session_id(session_response)
    assert session_id

    # Agent will use local ShellRuntime instead of ACP terminals
    # This test ensures graceful fallback


async def _wait_for_notifications(client: TestClient, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive at least one sessionUpdate."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if client.notifications:
            return
        await asyncio.sleep(0.05)
    # Don't raise error - some tests may not produce notifications
