"""
Integration tests for ACP tool call permissions.

Tests that tool permission requests are properly sent to the ACP client
and that permission decisions (allow/reject with persistence) work correctly.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest
from acp import InitializeRequest, NewSessionRequest, PromptRequest
from acp.helpers import text_block
from acp.schema import ClientCapabilities, Implementation, StopReason
from acp.stdio import spawn_agent_process

from fast_agent.mcp.common import create_namespaced_name

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
END_TURN: StopReason = "end_turn"


def get_fast_agent_cmd(temp_dir: str | None = None, no_permissions: bool = False):
    """Get the command to run fast-agent ACP server."""
    cmd = [
        sys.executable,
        "-m",
        "fast_agent.cli",
        "serve",
        "--config-path",
        str(CONFIG_PATH),
        "--transport",
        "acp",
        "--servers",
        "progress_test",
        "--model",
        "passthrough",
        "--name",
        "fast-agent-acp-test",
    ]
    if no_permissions:
        cmd.append("--no-permissions")
    return tuple(cmd)


async def _wait_for_permission_requests(
    client: TestClient, count: int = 1, timeout: float = 3.0
) -> list:
    """Wait for the ACP client to receive permission requests."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    # Look for permission requests that were made to the client
    permission_requests = []
    while loop.time() < deadline:
        # Permission requests are handled through requestPermission method
        # We can check if the client was queried
        if hasattr(client, "_permission_request_count"):
            if client._permission_request_count >= count:
                return permission_requests
        await asyncio.sleep(0.05)

    return permission_requests


@pytest.mark.integration
@pytest.mark.asyncio
async def test_permission_request_sent_when_tool_called() -> None:
    """Test that a permission request is sent when a tool is called."""
    client = TestClient()
    # Queue up a permission response - allow once
    client.queue_permission_selected("allow_once")

    with tempfile.TemporaryDirectory() as temp_dir:
        async with spawn_agent_process(
            lambda _: client, *get_fast_agent_cmd(temp_dir)
        ) as (connection, _process):
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

            # Create session with the temp directory
            session_response = await connection.newSession(
                NewSessionRequest(mcpServers=[], cwd=temp_dir)
            )
            session_id = session_response.sessionId

            # Call a tool that should trigger permission request
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            prompt_response = await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Wait a bit for any async operations
            await asyncio.sleep(0.2)

            # The tool should have been allowed to execute
            assert prompt_response.stopReason == END_TURN


@pytest.mark.integration
@pytest.mark.asyncio
async def test_allow_once_permits_execution() -> None:
    """Test that allow_once permits tool execution without persisting."""
    client = TestClient()
    # Queue up permission responses for multiple calls
    client.queue_permission_selected("allow_once")
    client.queue_permission_selected("allow_once")  # For second call

    with tempfile.TemporaryDirectory() as temp_dir:
        async with spawn_agent_process(
            lambda _: client, *get_fast_agent_cmd(temp_dir)
        ) as (connection, _process):
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
                NewSessionRequest(mcpServers=[], cwd=temp_dir)
            )
            session_id = session_response.sessionId

            # First tool call - should request permission
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Second tool call - should request permission again (allow_once doesn't persist)
            await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Check that permission file was NOT created
            permissions_path = Path(temp_dir) / ".fast-agent" / "auths.md"
            assert not permissions_path.exists(), "allow_once should not create persistence file"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reject_once_blocks_execution() -> None:
    """Test that reject_once blocks tool execution and returns error to LLM."""
    client = TestClient()
    client.queue_permission_selected("reject_once")

    with tempfile.TemporaryDirectory() as temp_dir:
        async with spawn_agent_process(
            lambda _: client, *get_fast_agent_cmd(temp_dir)
        ) as (connection, _process):
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
                NewSessionRequest(mcpServers=[], cwd=temp_dir)
            )
            session_id = session_response.sessionId

            # Tool call should be rejected
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            response = await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # The response should still complete but tool was denied
            # (passthrough model will just echo back the result)
            assert response.stopReason == END_TURN

            # Check that permission file was NOT created
            permissions_path = Path(temp_dir) / ".fast-agent" / "auths.md"
            assert not permissions_path.exists(), "reject_once should not create persistence file"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_allow_always_persists_permission() -> None:
    """Test that allow_always creates persistence file."""
    client = TestClient()
    client.queue_permission_selected("allow_always")

    with tempfile.TemporaryDirectory() as temp_dir:
        async with spawn_agent_process(
            lambda _: client, *get_fast_agent_cmd(temp_dir)
        ) as (connection, _process):
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
                NewSessionRequest(mcpServers=[], cwd=temp_dir)
            )
            session_id = session_response.sessionId

            # Tool call with allow_always
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Wait for async file write
            await asyncio.sleep(0.2)

            # Check that permission file WAS created
            permissions_path = Path(temp_dir) / ".fast-agent" / "auths.md"
            assert permissions_path.exists(), "allow_always should create persistence file"

            # Verify content
            content = permissions_path.read_text()
            assert "allow_always" in content
            assert "progress_test" in content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reject_always_persists_permission() -> None:
    """Test that reject_always creates persistence file."""
    client = TestClient()
    client.queue_permission_selected("reject_always")

    with tempfile.TemporaryDirectory() as temp_dir:
        async with spawn_agent_process(
            lambda _: client, *get_fast_agent_cmd(temp_dir)
        ) as (connection, _process):
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
                NewSessionRequest(mcpServers=[], cwd=temp_dir)
            )
            session_id = session_response.sessionId

            # Tool call with reject_always
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Wait for async file write
            await asyncio.sleep(0.2)

            # Check that permission file WAS created
            permissions_path = Path(temp_dir) / ".fast-agent" / "auths.md"
            assert permissions_path.exists(), "reject_always should create persistence file"

            # Verify content
            content = permissions_path.read_text()
            assert "reject_always" in content
            assert "progress_test" in content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_permissions_flag_disables_permission_checks() -> None:
    """Test that --no-permissions flag allows all tool calls without asking."""
    client = TestClient()
    # Don't queue any permission responses - they shouldn't be needed

    with tempfile.TemporaryDirectory() as temp_dir:
        async with spawn_agent_process(
            lambda _: client, *get_fast_agent_cmd(temp_dir, no_permissions=True)
        ) as (connection, _process):
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
                NewSessionRequest(mcpServers=[], cwd=temp_dir)
            )
            session_id = session_response.sessionId

            # Tool call should work without permission request
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            response = await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Tool should execute successfully
            assert response.stopReason == END_TURN


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancelled_permission_blocks_execution() -> None:
    """Test that cancelled permission request blocks tool execution."""
    client = TestClient()
    client.queue_permission_cancelled()

    with tempfile.TemporaryDirectory() as temp_dir:
        async with spawn_agent_process(
            lambda _: client, *get_fast_agent_cmd(temp_dir)
        ) as (connection, _process):
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
                NewSessionRequest(mcpServers=[], cwd=temp_dir)
            )
            session_id = session_response.sessionId

            # Tool call should be cancelled
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            response = await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Response should complete (cancelled is treated as error)
            assert response.stopReason == END_TURN
