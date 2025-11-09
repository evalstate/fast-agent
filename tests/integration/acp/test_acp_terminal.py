"""Integration tests for ACP terminal support."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from acp import InitializeRequest, NewSessionRequest
from acp.schema import ClientCapabilities, Implementation
from acp.stdio import spawn_agent_process

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_terminal_capability_advertisement() -> None:
    """Ensure ACP server advertises terminal capability when shell is enabled."""
    client = TestClient()

    # Command WITH --shell flag to enable terminal support
    fast_agent_cmd_with_shell = (
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
        "--shell",  # Enable shell/terminal support
    )

    async with spawn_agent_process(
        lambda _: client, *fast_agent_cmd_with_shell
    ) as (connection, _process):
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=True,  # Client supports terminals
            ),
            clientInfo=Implementation(name="pytest-terminal-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)

        assert init_response.protocolVersion == 1
        assert init_response.agentCapabilities is not None

        # Check that terminal capability is advertised
        assert hasattr(init_response.agentCapabilities, "terminal")
        assert init_response.agentCapabilities.terminal is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_terminal_not_advertised_without_shell_flag() -> None:
    """Ensure ACP server does NOT advertise terminal capability when shell is disabled."""
    client = TestClient()

    # Command WITHOUT --shell flag
    fast_agent_cmd_no_shell = (
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
        "fast-agent-acp-no-terminal-test",
        # NO --shell flag
    )

    async with spawn_agent_process(
        lambda _: client, *fast_agent_cmd_no_shell
    ) as (connection, _process):
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-no-terminal-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)

        assert init_response.protocolVersion == 1
        assert init_response.agentCapabilities is not None

        # Check that terminal capability is NOT advertised (or is False)
        terminal_capability = getattr(init_response.agentCapabilities, "terminal", False)
        assert terminal_capability is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_terminal_create_and_execute() -> None:
    """Test creating a terminal and executing a simple command."""
    client = TestClient()

    fast_agent_cmd_with_shell = (
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
        "fast-agent-acp-terminal-exec-test",
        "--shell",
    )

    async with spawn_agent_process(
        lambda _: client, *fast_agent_cmd_with_shell
    ) as (connection, _process):
        # Initialize
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=True,
            ),
            clientInfo=Implementation(name="pytest-terminal-exec-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Create terminal and execute a simple command
        # Use 'echo' which works cross-platform
        terminal_response = await connection.createTerminal(
            {
                "sessionId": session_id,
                "command": "echo",
                "args": ["Hello from ACP terminal!"],
                "cwd": str(TEST_DIR),
            }
        )

        assert "terminalId" in terminal_response
        terminal_id = terminal_response["terminalId"]

        # Wait a bit for command to execute
        await asyncio.sleep(0.5)

        # Get terminal output
        output_response = await connection.terminalOutput({"terminalId": terminal_id})

        assert "output" in output_response
        assert "Hello from ACP terminal!" in output_response["output"]
        assert "truncated" in output_response
        assert output_response["truncated"] is False

        # Wait for exit
        exit_response = await connection.waitForTerminalExit({"terminalId": terminal_id})

        assert "exitCode" in exit_response
        assert exit_response["exitCode"] == 0

        # Release terminal
        release_response = await connection.releaseTerminal({"terminalId": terminal_id})
        assert release_response == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_terminal_kill() -> None:
    """Test killing a long-running terminal process."""
    client = TestClient()

    fast_agent_cmd_with_shell = (
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
        "fast-agent-acp-terminal-kill-test",
        "--shell",
    )

    async with spawn_agent_process(
        lambda _: client, *fast_agent_cmd_with_shell
    ) as (connection, _process):
        # Initialize
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=True,
            ),
            clientInfo=Implementation(name="pytest-terminal-kill-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Create terminal with a long-running command (sleep)
        # Use python -c for cross-platform compatibility
        terminal_response = await connection.createTerminal(
            {
                "sessionId": session_id,
                "command": sys.executable,
                "args": ["-c", "import time; time.sleep(10)"],
                "cwd": str(TEST_DIR),
            }
        )

        terminal_id = terminal_response["terminalId"]

        # Give it a moment to start
        await asyncio.sleep(0.2)

        # Kill the terminal
        kill_response = await connection.killTerminal({"terminalId": terminal_id})
        assert kill_response == {}

        # Wait for it to exit
        exit_response = await connection.waitForTerminalExit({"terminalId": terminal_id})

        # After killing, it should have exited (exit code might be non-zero or -1)
        assert "exitCode" in exit_response

        # Release terminal
        await connection.releaseTerminal({"terminalId": terminal_id})
