"""
Integration tests for ACP tool call permissions.

Tests that tool permission requests are properly sent to the ACP client,
and that the user's decision is respected (allow/reject, once/always).
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

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
END_TURN: StopReason = "end_turn"


def _get_fast_agent_cmd(cwd: str | None = None) -> tuple:
    """Build the fast-agent command for spawning."""
    from acp.stdio import spawn_agent_process  # noqa: F401

    return (
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
    )


async def _wait_for_notifications(client: TestClient, count: int = 1, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive specified number of notifications."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if len(client.notifications) >= count:
            return
        await asyncio.sleep(0.05)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_permission_allow_once() -> None:
    """Test that allow_once permits tool execution without persistence."""
    from acp.stdio import spawn_agent_process

    from fast_agent.mcp.common import create_namespaced_name

    client = TestClient()
    # Queue permission response: allow once
    client.queue_permission_selected("allow_once")

    # Use temp directory for session cwd to avoid polluting test dir
    with tempfile.TemporaryDirectory() as tmpdir:
        async with spawn_agent_process(lambda _: client, *_get_fast_agent_cmd()) as (
            connection,
            _process,
        ):
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

            # Create session with temp directory
            session_response = await connection.newSession(
                NewSessionRequest(mcpServers=[], cwd=tmpdir)
            )
            session_id = session_response.sessionId

            # Send a prompt that triggers a tool call
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            prompt_response = await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Should succeed
            assert prompt_response.stopReason == END_TURN

            # Wait for notifications
            await _wait_for_notifications(client, count=3, timeout=3.0)

            # Check that tool completed successfully
            tool_notifications = [
                n
                for n in client.notifications
                if hasattr(n.update, "sessionUpdate")
                and n.update.sessionUpdate in ["tool_call", "tool_call_update"]
            ]

            # Should have tool notifications indicating success
            assert len(tool_notifications) > 0

            # Check no auth file was created (allow_once doesn't persist)
            auth_file = Path(tmpdir) / ".fast-agent" / "auths.md"
            assert not auth_file.exists(), "allow_once should not create auth file"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_permission_reject_once() -> None:
    """Test that reject_once blocks tool execution with error message."""
    from acp.stdio import spawn_agent_process

    from fast_agent.mcp.common import create_namespaced_name

    client = TestClient()
    # Queue permission response: reject once
    client.queue_permission_selected("reject_once")

    with tempfile.TemporaryDirectory() as tmpdir:
        async with spawn_agent_process(lambda _: client, *_get_fast_agent_cmd()) as (
            connection,
            _process,
        ):
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
                NewSessionRequest(mcpServers=[], cwd=tmpdir)
            )
            session_id = session_response.sessionId

            # Send a prompt that triggers a tool call
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            prompt_response = await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Should still return end_turn (agent handled the error)
            assert prompt_response.stopReason == END_TURN

            # Wait for notifications
            await _wait_for_notifications(client, count=2, timeout=3.0)

            # Check agent message contains the decline message
            agent_messages = [
                n
                for n in client.notifications
                if hasattr(n.update, "sessionUpdate") and n.update.sessionUpdate == "agent_message_chunk"
            ]

            # The agent should report the tool error in its response
            assert len(agent_messages) > 0
            # Note: passthrough model echoes tool results, so decline message should appear
            # in the response. The key test is that tool didn't execute.

            # Check no auth file was created (reject_once doesn't persist)
            auth_file = Path(tmpdir) / ".fast-agent" / "auths.md"
            assert not auth_file.exists(), "reject_once should not create auth file"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_permission_allow_always_persists() -> None:
    """Test that allow_always persists to .fast-agent/auths.md file."""
    from acp.stdio import spawn_agent_process

    from fast_agent.mcp.common import create_namespaced_name

    client = TestClient()
    # Queue permission response: allow always
    client.queue_permission_selected("allow_always")

    with tempfile.TemporaryDirectory() as tmpdir:
        async with spawn_agent_process(lambda _: client, *_get_fast_agent_cmd()) as (
            connection,
            _process,
        ):
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
                NewSessionRequest(mcpServers=[], cwd=tmpdir)
            )
            session_id = session_response.sessionId

            # Send a prompt that triggers a tool call
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            prompt_response = await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            assert prompt_response.stopReason == END_TURN

            # Wait for notifications
            await _wait_for_notifications(client, count=3, timeout=3.0)

            # Check that auth file was created with allow entry
            auth_file = Path(tmpdir) / ".fast-agent" / "auths.md"
            assert auth_file.exists(), "allow_always should create auth file"

            content = auth_file.read_text()
            assert "## always_allow" in content
            assert "progress_test/progress_task" in content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_permission_reject_always_persists() -> None:
    """Test that reject_always persists to .fast-agent/auths.md file."""
    from acp.stdio import spawn_agent_process

    from fast_agent.mcp.common import create_namespaced_name

    client = TestClient()
    # Queue permission response: reject always
    client.queue_permission_selected("reject_always")

    with tempfile.TemporaryDirectory() as tmpdir:
        async with spawn_agent_process(lambda _: client, *_get_fast_agent_cmd()) as (
            connection,
            _process,
        ):
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
                NewSessionRequest(mcpServers=[], cwd=tmpdir)
            )
            session_id = session_response.sessionId

            # Send a prompt that triggers a tool call
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            prompt_response = await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            assert prompt_response.stopReason == END_TURN

            # Wait for notifications
            await _wait_for_notifications(client, count=2, timeout=3.0)

            # Check that auth file was created with reject entry
            auth_file = Path(tmpdir) / ".fast-agent" / "auths.md"
            assert auth_file.exists(), "reject_always should create auth file"

            content = auth_file.read_text()
            assert "## always_reject" in content
            assert "progress_test/progress_task" in content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_permission_cancelled() -> None:
    """Test that cancelled permission request blocks tool execution."""
    from acp.stdio import spawn_agent_process

    from fast_agent.mcp.common import create_namespaced_name

    client = TestClient()
    # Queue permission response: cancelled
    client.queue_permission_cancelled()

    with tempfile.TemporaryDirectory() as tmpdir:
        async with spawn_agent_process(lambda _: client, *_get_fast_agent_cmd()) as (
            connection,
            _process,
        ):
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
                NewSessionRequest(mcpServers=[], cwd=tmpdir)
            )
            session_id = session_response.sessionId

            # Send a prompt that triggers a tool call
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            prompt_response = await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Should still return end_turn
            assert prompt_response.stopReason == END_TURN

            # Wait for notifications
            await _wait_for_notifications(client, count=2, timeout=3.0)

            # No auth file should be created
            auth_file = Path(tmpdir) / ".fast-agent" / "auths.md"
            assert not auth_file.exists(), "cancelled should not create auth file"
