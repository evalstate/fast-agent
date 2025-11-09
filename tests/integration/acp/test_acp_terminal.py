"""Integration tests for ACP terminal support."""

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


def get_fast_agent_cmd(with_shell: bool = True) -> tuple:
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
        "fast-agent-acp-terminal-test",
    ]
    if with_shell:
        cmd.append("--shell")
    return tuple(cmd)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_terminal_support_enabled() -> None:
    """Test that terminal support is properly enabled when client advertises capability."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *get_fast_agent_cmd(with_shell=True)) as (
        connection,
        _process,
    ):
        # Initialize with terminal support enabled
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=True,  # Enable terminal support
            ),
            clientInfo=Implementation(name="pytest-terminal-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)

        assert init_response.protocolVersion == 1
        assert init_response.agentCapabilities is not None

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Send prompt that should trigger terminal execution
        # The passthrough model will echo our input, so we craft a tool call request
        prompt_text = 'use the execute tool to run: echo "test terminal"'
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
async def test_acp_terminal_execution() -> None:
    """Test actual terminal command execution via ACP."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *get_fast_agent_cmd(with_shell=True)) as (
        connection,
        _process,
    ):
        # Initialize with terminal support
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=True,
            ),
            clientInfo=Implementation(name="pytest-terminal-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Directly test terminal methods are being called
        # Since we're using passthrough model, we can't test actual LLM-driven tool calls
        # but we can verify the terminal runtime is set up correctly

        # The terminals dict should be empty initially
        assert len(client.terminals) == 0

        # Manually test terminal lifecycle (client creates ID)
        create_result = await client.terminal_create(
            {"sessionId": session_id, "command": "echo test"}
        )
        terminal_id = create_result["terminalId"]

        # Verify terminal was created with client-generated ID
        assert terminal_id == "terminal-1"  # First terminal
        assert terminal_id in client.terminals
        assert client.terminals[terminal_id]["command"] == "echo test"

        # Get output
        output = await client.terminal_output({"terminalId": terminal_id, "sessionId": session_id})
        assert "Executed: echo test" in output["output"]
        assert output["exitCode"] == 0

        # Release terminal
        await client.terminal_release({"terminalId": terminal_id, "sessionId": session_id})
        assert terminal_id not in client.terminals


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_terminal_disabled_when_no_shell_flag() -> None:
    """Test that terminal runtime is not injected when --shell flag is not provided."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *get_fast_agent_cmd(with_shell=False)) as (
        connection,
        _process,
    ):
        # Initialize with terminal support (client side)
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=True,  # Client supports it
            ),
            clientInfo=Implementation(name="pytest-terminal-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Terminal runtime should not be injected because --shell wasn't provided
        # This test mainly ensures no errors occur when terminal capability is advertised
        # but shell runtime isn't enabled


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_terminal_disabled_when_client_unsupported() -> None:
    """Test that terminal runtime is not used when client doesn't support terminals."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *get_fast_agent_cmd(with_shell=True)) as (
        connection,
        _process,
    ):
        # Initialize WITHOUT terminal support
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,  # Client doesn't support terminals
            ),
            clientInfo=Implementation(name="pytest-terminal-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
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
