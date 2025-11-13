"""Integration tests for ACP filesystem tool calling."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
        "claude-3-5-sonnet-20241022",  # Use real model
        "--name",
        "fast-agent-acp-filesystem-toolcall-test",
    ]
    return tuple(cmd)


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_acp_filesystem_read_tool_call() -> None:
    """Test that read_text_file tool can be called by the LLM."""
    from acp.stdio import spawn_agent_process

    client = TestClient()

    # Set up a test file in the client
    test_path = "/test/sample.txt"
    test_content = "Hello from test file!"
    client.files[test_path] = test_content

    async with spawn_agent_process(lambda _: client, *get_fast_agent_cmd()) as (
        connection,
        _process,
    ):
        # Initialize with filesystem support enabled
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-filesystem-client", version="0.0.1"),
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

        # Ask the LLM to read the test file
        prompt_text = f'Please use the read_text_file tool to read the file at {test_path}'
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )

        # Should complete successfully
        assert prompt_response.stopReason == END_TURN

        # Verify the file was actually accessed
        # The LLM should have called read_text_file which routes through the client
        # We can verify this by checking client notifications contain the file content
        assert len(client.notifications) > 0


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_acp_filesystem_write_tool_call() -> None:
    """Test that write_text_file tool can be called by the LLM."""
    from acp.stdio import spawn_agent_process

    client = TestClient()

    async with spawn_agent_process(lambda _: client, *get_fast_agent_cmd()) as (
        connection,
        _process,
    ):
        # Initialize with filesystem support enabled
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-filesystem-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Ask the LLM to write to a file
        test_path = "/test/output.txt"
        test_content = "Test content from LLM"
        prompt_text = f'Please use the write_text_file tool to write "{test_content}" to {test_path}'

        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )

        # Should complete successfully
        assert prompt_response.stopReason == END_TURN

        # Verify the file was written
        assert test_path in client.files
        assert client.files[test_path] == test_content
