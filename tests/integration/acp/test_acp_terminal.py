"""Integration test for ACP terminal support."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

import pytest
from acp import InitializeRequest, NewSessionRequest, PromptRequest
from acp.helpers import text_block
from acp.schema import (
    ClientCapabilities,
    CreateTerminalRequest,
    Implementation,
    KillTerminalCommandRequest,
    KillTerminalCommandResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    StopReason,
    TerminalExitStatus,
    TerminalHandle,
    TerminalOutputRequest,
    TerminalOutputResponse,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
)
from acp.stdio import spawn_agent_process

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
END_TURN: StopReason = "end_turn"


class TerminalCapableTestClient(TestClient):
    """
    Test client that implements terminal protocol methods.

    This simulates a client (editor/IDE) that can execute commands
    in its environment.
    """

    def __init__(self) -> None:
        super().__init__()
        self.terminals: dict[str, dict[str, any]] = {}  # terminalId -> terminal info
        self.terminal_outputs: dict[str, str] = {}  # terminalId -> output
        self.terminal_exit_status: dict[str, TerminalExitStatus] = {}  # terminalId -> status

    async def createTerminal(self, params: CreateTerminalRequest) -> TerminalHandle:
        """Simulate terminal creation and command execution."""
        terminal_id = str(uuid.uuid4())

        # Store terminal info
        self.terminals[terminal_id] = {
            "sessionId": params.sessionId,
            "command": params.command,
            "args": params.args,
            "cwd": params.cwd,
            "env": params.env,
        }

        # Simulate command execution
        # For testing, we'll echo back the command
        if params.command == "echo":
            # Echo command - concatenate args
            output = " ".join(params.args) if params.args else ""
            exit_code = 0
        elif params.command == "pwd":
            # Print working directory
            output = params.cwd or "/test/dir"
            exit_code = 0
        elif params.command == "exit":
            # Exit with specific code
            exit_code = int(params.args[0]) if params.args else 0
            output = ""
        elif params.command == "fail":
            # Simulate command failure
            output = "Command failed"
            exit_code = 1
        else:
            # Unknown command - simulate shell behavior
            output = f"Command: {params.command}"
            if params.args:
                output += f" {' '.join(params.args)}"
            exit_code = 0

        # Store output and exit status
        self.terminal_outputs[terminal_id] = output
        self.terminal_exit_status[terminal_id] = TerminalExitStatus(
            exitCode=exit_code, signal=None
        )

        return TerminalHandle(terminalId=terminal_id)

    async def terminalOutput(self, params: TerminalOutputRequest) -> TerminalOutputResponse:
        """Return terminal output."""
        terminal_id = params.terminalId
        output = self.terminal_outputs.get(terminal_id, "")
        exit_status = self.terminal_exit_status.get(terminal_id)

        return TerminalOutputResponse(
            output=output, truncated=False, exitStatus=exit_status
        )

    async def waitForTerminalExit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:
        """Wait for terminal to exit (returns immediately in test)."""
        terminal_id = params.terminalId
        exit_status = self.terminal_exit_status.get(terminal_id)

        if exit_status:
            return WaitForTerminalExitResponse(
                exitCode=exit_status.exitCode, signal=exit_status.signal
            )
        else:
            # Terminal not found or not exited
            return WaitForTerminalExitResponse(exitCode=1, signal=None)

    async def killTerminal(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse | None:
        """Kill a terminal (no-op in test)."""
        terminal_id = params.terminalId
        if terminal_id in self.terminals:
            # Simulate SIGTERM
            self.terminal_exit_status[terminal_id] = TerminalExitStatus(
                exitCode=None, signal="SIGTERM"
            )
        return KillTerminalCommandResponse()

    async def releaseTerminal(
        self, params: ReleaseTerminalRequest
    ) -> ReleaseTerminalResponse | None:
        """Release terminal resources."""
        terminal_id = params.terminalId
        # Clean up terminal data
        self.terminals.pop(terminal_id, None)
        self.terminal_outputs.pop(terminal_id, None)
        self.terminal_exit_status.pop(terminal_id, None)
        return ReleaseTerminalResponse()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_terminal_echo_command() -> None:
    """Test ACP terminal support with echo command."""
    client = TerminalCapableTestClient()

    # Launch fast-agent with --shell flag to enable terminal support
    cmd = (
        sys.executable,
        "-m",
        "fast_agent.cli",
        "acp",
        "--config-path",
        str(CONFIG_PATH),
        "--model",
        "passthrough",
        "--name",
        "fast-agent-acp-terminal-test",
        "--shell",  # Enable shell runtime / terminal support
    )

    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
        # Initialize with terminal capability
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=True,  # Advertise terminal support
            ),
            clientInfo=Implementation(name="pytest-terminal-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)

        assert init_response.protocolVersion == 1
        assert init_response.agentInfo.name == "fast-agent-acp-terminal-test"

        # Create a session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Send a prompt that asks the agent to use the execute tool
        # The passthrough model will echo our request, but we're really testing
        # that the terminal infrastructure is wired up correctly
        prompt_text = 'Use the execute tool to run: echo "Hello from ACP terminal"'
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)

        # Verify that the agent got the message
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_terminal_without_client_support() -> None:
    """Test that ACP gracefully handles clients without terminal support."""
    client = TestClient()  # Regular client without terminal support

    cmd = (
        sys.executable,
        "-m",
        "fast_agent.cli",
        "acp",
        "--config-path",
        str(CONFIG_PATH),
        "--model",
        "passthrough",
        "--name",
        "fast-agent-acp-no-terminal-test",
        "--shell",  # Request shell runtime but client doesn't support it
    )

    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
        # Initialize WITHOUT terminal capability
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,  # No terminal support
            ),
            clientInfo=Implementation(name="pytest-no-terminal-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)

        assert init_response.protocolVersion == 1

        # Create a session - should succeed even without terminal support
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # The agent should work normally, just without terminal support
        prompt_text = "test prompt"
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN


async def _wait_for_notifications(client: TestClient, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive at least one sessionUpdate."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if client.notifications:
            return
        await asyncio.sleep(0.05)
    # Don't raise - some tests may not generate notifications
