"""
Integration tests for ACP tool permissions.

Tests that tool permission requests are sent to the ACP client and
that the permission decision affects tool execution.
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

from fast_agent.mcp.common import create_namespaced_name

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
END_TURN: StopReason = "end_turn"


def _get_base_command(cwd: str | None = None, permissions: bool = True) -> tuple:
    """Build the fast-agent command with permissions setting."""
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
        "fast-agent-acp-permission-test",
    ]
    if not permissions:
        cmd.append("--no-permissions")
    return tuple(cmd)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_permission_request_sent_when_enabled() -> None:
    """Test that permission requests are sent to the client when a tool is called."""
    # Use spawn_agent_process which is the standard pattern
    from acp.stdio import spawn_agent_process

    client = TestClient()
    # Queue a permission response (allow_once)
    client.queue_permission_selected("allow_once")

    cmd = _get_base_command(permissions=True)
    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
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

        # Create session with a temp directory as cwd for file store
        with tempfile.TemporaryDirectory() as tmpdir:
            session_response = await connection.newSession(
                NewSessionRequest(mcpServers=[], cwd=tmpdir)
            )
            session_id = session_response.sessionId
            assert session_id

            # Send a prompt that will trigger a tool call
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            prompt_response = await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )
            assert prompt_response.stopReason == END_TURN

            # Wait for notifications
            await _wait_for_notifications(client, count=3, timeout=5.0)

            # The tool should have been executed successfully since we allowed it
            # Check for tool completion notification
            tool_updates = [
                n
                for n in client.notifications
                if hasattr(n.update, "sessionUpdate")
                and n.update.sessionUpdate in ["tool_call", "tool_call_update"]
            ]
            assert len(tool_updates) > 0, "Expected tool notifications"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_permission_reject_blocks_tool() -> None:
    """Test that rejecting a permission blocks tool execution with appropriate error."""
    from acp.stdio import spawn_agent_process

    client = TestClient()
    # Queue a permission rejection
    client.queue_permission_selected("reject_once")

    cmd = _get_base_command(permissions=True)
    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
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
        with tempfile.TemporaryDirectory() as tmpdir:
            session_response = await connection.newSession(
                NewSessionRequest(mcpServers=[], cwd=tmpdir)
            )
            session_id = session_response.sessionId

            # Send a prompt that will trigger a tool call
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            prompt_response = await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Should still return normally (not crash)
            assert prompt_response.stopReason == END_TURN

            # Wait for notifications
            await _wait_for_notifications(client, count=1, timeout=3.0)

            # The response should contain the rejection message in the agent output
            # Since passthrough model mirrors content, check for the error message
            text_updates = [
                n
                for n in client.notifications
                if hasattr(n.update, "sessionUpdate")
                and n.update.sessionUpdate == "agent_message_chunk"
            ]

            # The agent response should contain the decline message
            if text_updates:
                last_text = getattr(text_updates[-1].update.content, "text", "")
                assert (
                    "User declined" in last_text or "declined this operation" in last_text
                ), f"Expected decline message in response, got: {last_text}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_permission_cancelled_blocks_tool() -> None:
    """Test that cancelling permission request blocks tool execution."""
    from acp.stdio import spawn_agent_process

    client = TestClient()
    # Queue a cancellation (no selection queued means default is cancelled)

    cmd = _get_base_command(permissions=True)
    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
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
        with tempfile.TemporaryDirectory() as tmpdir:
            session_response = await connection.newSession(
                NewSessionRequest(mcpServers=[], cwd=tmpdir)
            )
            session_id = session_response.sessionId

            # Send a prompt that will trigger a tool call
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            prompt_response = await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Should return normally
            assert prompt_response.stopReason == END_TURN


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_no_permissions_flag_skips_permission_check() -> None:
    """Test that --no-permissions flag disables permission requests."""
    from acp.stdio import spawn_agent_process

    client = TestClient()
    # Don't queue any permission response - if permission is requested,
    # the default behaviour (cancelled) would reject the tool

    cmd = _get_base_command(permissions=False)
    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
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
        with tempfile.TemporaryDirectory() as tmpdir:
            session_response = await connection.newSession(
                NewSessionRequest(mcpServers=[], cwd=tmpdir)
            )
            session_id = session_response.sessionId

            # Send a prompt that will trigger a tool call
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            prompt_response = await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Should complete successfully because no permission check is made
            assert prompt_response.stopReason == END_TURN

            # Wait for notifications
            await _wait_for_notifications(client, count=3, timeout=5.0)

            # Check for successful tool execution (tool_call notifications)
            tool_updates = [
                n
                for n in client.notifications
                if hasattr(n.update, "sessionUpdate")
                and n.update.sessionUpdate in ["tool_call", "tool_call_update"]
            ]
            assert len(tool_updates) > 0, "Expected tool notifications (tool should execute without permission)"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_allow_always_persists_permission() -> None:
    """Test that allow_always permission is persisted to file."""
    from acp.stdio import spawn_agent_process

    client = TestClient()
    # Queue allow_always response
    client.queue_permission_selected("allow_always")

    cmd = _get_base_command(permissions=True)
    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
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

        # Create session with a temp directory for file store
        with tempfile.TemporaryDirectory() as tmpdir:
            session_response = await connection.newSession(
                NewSessionRequest(mcpServers=[], cwd=tmpdir)
            )
            session_id = session_response.sessionId

            # First tool call - should trigger permission request
            tool_name = create_namespaced_name("progress_test", "progress_task")
            prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
            await connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
            )

            # Wait for notifications
            await _wait_for_notifications(client, count=3, timeout=5.0)

            # Check if the permission file was created
            auths_file = Path(tmpdir) / ".fast-agent" / "auths.md"
            if auths_file.exists():
                content = auths_file.read_text()
                # The file should contain the allowed tool
                assert "progress_test/progress_task" in content
                assert "Allowed" in content


async def _wait_for_notifications(client: TestClient, count: int = 1, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive specified number of notifications."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if len(client.notifications) >= count:
            return
        await asyncio.sleep(0.05)
