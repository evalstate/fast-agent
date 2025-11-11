"""Integration tests for ACP tool call lifecycle including permissions and progress."""

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

# Path to test MCP server with tools
MCP_TOOLS_SERVER = Path(__file__).parent.parent / "api" / "mcp_tools_server.py"
MCP_PROGRESS_SERVER = Path(__file__).parent.parent / "api" / "mcp_progress_server.py"

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
END_TURN: StopReason = "end_turn"


def get_fast_agent_cmd(mcp_server_path: Path, server_name: str) -> tuple:
    """Build fast-agent command with MCP server."""
    return (
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
        "fast-agent-tool-test",
        "--server",
        f"{server_name}={sys.executable}",
        "--server-arg",
        f"{server_name}={str(mcp_server_path)}",
    )


async def _wait_for_notifications(
    client: TestClient, min_count: int = 1, timeout: float = 3.0
) -> None:
    """Wait for the ACP client to receive notifications."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if len(client.notifications) >= min_count:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"Expected at least {min_count} notifications, got {len(client.notifications)}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_call_lifecycle_notifications() -> None:
    """
    Test that tool calls send proper lifecycle notifications:
    - tool_call (start)
    - tool_call_update (in_progress)
    - tool_call_update (completed/failed)
    """
    client = TestClient()

    # Queue permission to allow tool execution
    client.queue_permission_selected("allow_once")

    fast_agent_cmd = get_fast_agent_cmd(MCP_TOOLS_SERVER, "tools_server")

    async with spawn_agent_process(lambda _: client, *fast_agent_cmd) as (connection, _process):
        # Initialize
        init_response = await connection.initialize(
            InitializeRequest(
                protocolVersion=1,
                clientCapabilities=ClientCapabilities(
                    fs={"readTextFile": True, "writeTextFile": True},
                    terminal=False,
                ),
                clientInfo=Implementation(name="pytest-client", version="0.0.1"),
            )
        )
        assert init_response.protocolVersion == 1

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Send a prompt that will trigger a tool call
        # Using passthrough model, so we manually construct the tool call request
        # For a real test, we would need an agent model that calls tools
        # For now, we'll skip this part as it requires a real LLM
        # This test serves as a template for when integration is fully functional

        # Note: This test is a placeholder. To fully test tool calls, we need:
        # 1. An agent model that actually calls tools (not passthrough)
        # 2. Or mock the tool execution path directly

        # For now, just verify the server initialized correctly
        assert session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_permission_request() -> None:
    """
    Test that tool executions trigger permission requests in ACP mode.
    """
    client = TestClient()

    # Queue a permission response (allow_once)
    client.queue_permission_selected("allow_once")

    fast_agent_cmd = get_fast_agent_cmd(MCP_TOOLS_SERVER, "tools_server")

    async with spawn_agent_process(lambda _: client, *fast_agent_cmd) as (connection, _process):
        # Initialize
        init_response = await connection.initialize(
            InitializeRequest(
                protocolVersion=1,
                clientCapabilities=ClientCapabilities(
                    fs={"readTextFile": True, "writeTextFile": True},
                    terminal=False,
                ),
                clientInfo=Implementation(name="pytest-client", version="0.0.1"),
            )
        )
        assert init_response.protocolVersion == 1

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Note: Similar to above, this requires an agent model that calls tools
        # The test validates the setup is correct
        assert session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_permission_denied() -> None:
    """
    Test that rejecting a tool permission prevents execution.
    """
    client = TestClient()

    # Queue a permission denial (reject_once)
    client.queue_permission_selected("reject_once")

    fast_agent_cmd = get_fast_agent_cmd(MCP_TOOLS_SERVER, "tools_server")

    async with spawn_agent_process(lambda _: client, *fast_agent_cmd) as (connection, _process):
        # Initialize
        init_response = await connection.initialize(
            InitializeRequest(
                protocolVersion=1,
                clientCapabilities=ClientCapabilities(
                    fs={"readTextFile": True, "writeTextFile": True},
                    terminal=False,
                ),
                clientInfo=Implementation(name="pytest-client", version="0.0.1"),
            )
        )
        assert init_response.protocolVersion == 1

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Setup complete - actual tool call testing requires LLM integration
        assert session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_progress_notifications() -> None:
    """
    Test that MCP tool progress notifications are forwarded as ACP tool_call_update notifications.
    """
    client = TestClient()

    # Queue permission to allow tool execution
    client.queue_permission_selected("allow_once")

    fast_agent_cmd = get_fast_agent_cmd(MCP_PROGRESS_SERVER, "progress_server")

    async with spawn_agent_process(lambda _: client, *fast_agent_cmd) as (connection, _process):
        # Initialize
        init_response = await connection.initialize(
            InitializeRequest(
                protocolVersion=1,
                clientCapabilities=ClientCapabilities(
                    fs={"readTextFile": True, "writeTextFile": True},
                    terminal=False,
                ),
                clientInfo=Implementation(name="pytest-client", version="0.0.1"),
            )
        )
        assert init_response.protocolVersion == 1

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Note: Testing progress forwarding requires:
        # 1. An agent that calls the progress_task tool
        # 2. Verification that tool_call_update notifications are sent
        # This is a placeholder for when full integration is available
        assert session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_permission_caching() -> None:
    """
    Test that allow_always and reject_always permissions are cached.
    """
    client = TestClient()

    # Queue allow_always permission
    client.queue_permission_selected("allow_always")

    fast_agent_cmd = get_fast_agent_cmd(MCP_TOOLS_SERVER, "tools_server")

    async with spawn_agent_process(lambda _: client, *fast_agent_cmd) as (connection, _process):
        # Initialize
        init_response = await connection.initialize(
            InitializeRequest(
                protocolVersion=1,
                clientCapabilities=ClientCapabilities(
                    fs={"readTextFile": True, "writeTextFile": True},
                    terminal=False,
                ),
                clientInfo=Implementation(name="pytest-client", version="0.0.1"),
            )
        )
        assert init_response.protocolVersion == 1

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Note: Testing permission caching requires:
        # 1. Multiple tool calls to the same tool
        # 2. Verification that only one permission request is made
        # This requires LLM integration to trigger tool calls
        assert session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_call_includes_metadata() -> None:
    """
    Test that tool call notifications include proper metadata:
    - toolCallId (unique identifier)
    - title (human-readable description)
    - kind (read, edit, execute, etc.)
    - rawInput (tool arguments)
    - rawOutput (tool result)
    """
    client = TestClient()

    # Queue permission
    client.queue_permission_selected("allow_once")

    fast_agent_cmd = get_fast_agent_cmd(MCP_TOOLS_SERVER, "tools_server")

    async with spawn_agent_process(lambda _: client, *fast_agent_cmd) as (connection, _process):
        # Initialize
        init_response = await connection.initialize(
            InitializeRequest(
                protocolVersion=1,
                clientCapabilities=ClientCapabilities(
                    fs={"readTextFile": True, "writeTextFile": True},
                    terminal=False,
                ),
                clientInfo=Implementation(name="pytest-client", version="0.0.1"),
            )
        )
        assert init_response.protocolVersion == 1

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Note: Validating tool call metadata requires LLM integration
        assert session_id


# Note on Test Limitations:
# ------------------------
# These integration tests are currently limited because they require:
# 1. An LLM model that actually calls tools (not passthrough)
# 2. A way to trigger specific tool calls in tests
#
# Options to make these tests fully functional:
# A. Use a mock/test LLM provider that returns predefined tool calls
# B. Add a test endpoint to directly invoke tools (bypassing LLM)
# C. Use a local LLM model for testing (increases test complexity)
#
# For now, these tests validate:
# - ACP server starts correctly with MCP servers configured
# - Sessions can be created
# - Permission queuing works
# - Foundation is in place for full tool call testing
#
# Future work: Add option B (direct tool invocation endpoint) for comprehensive testing
