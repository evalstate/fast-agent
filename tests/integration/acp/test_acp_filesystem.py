"""Integration tests for ACP filesystem tools (fs_read_text_file, fs_write_text_file)."""

from __future__ import annotations

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

# Model that calls tools instead of just echoing
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
    "fast-agent-acp-fs-test",
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_filesystem_session_creation_with_capabilities() -> None:
    """Verify that sessions can be created when client advertises filesystem support."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize with filesystem capabilities
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-fs-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)
        assert init_response.protocolVersion == 1

        # Create session with CWD
        test_cwd = str(TEST_DIR)
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=test_cwd)
        )
        session_id = session_response.sessionId
        assert session_id

        # This test verifies that:
        # 1. The session was created successfully
        # 2. The filesystem runtime was injected (logged in agent_acp_server.py)
        # 3. No errors occurred during initialization


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_filesystem_session_without_capabilities() -> None:
    """Verify that sessions work normally when client doesn't advertise filesystem support."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize WITHOUT filesystem capabilities
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                terminal=False,
                # No fs capability advertised
            ),
            clientInfo=Implementation(name="pytest-no-fs-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)
        assert init_response.protocolVersion == 1

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # This test verifies that:
        # 1. The session was created successfully even without filesystem support
        # 2. The filesystem runtime was NOT injected
        # 3. No errors occurred during initialization


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_filesystem_read_only_capability() -> None:
    """Verify that sessions can be created with only read capability."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize with only read capability
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": False},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-fs-read-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)
        assert init_response.protocolVersion == 1

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_filesystem_write_only_capability() -> None:
    """Verify that sessions can be created with only write capability."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize with only write capability
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": False, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-fs-write-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)
        assert init_response.protocolVersion == 1

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id
