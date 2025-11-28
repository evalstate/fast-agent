"""
Integration tests for ACP tool call permissions.

Tests that permission requests are sent and handled correctly
according to the ACP protocol.
"""

from __future__ import annotations

import asyncio
import os
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


def _get_fast_agent_cmd(cwd: str | None = None, no_permissions: bool = False) -> tuple:
    """Build the fast-agent command with optional flags."""
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
async def test_permission_request_sent_when_tool_called() -> None:
    """Test that a permission request is sent when a tool is called."""
    client = TestClient()
    # Queue a rejection so the tool doesn't actually execute
    client.queue_permission_cancelled()

    async with spawn_agent_process(lambda _: client, *_get_fast_agent_cmd()) as (connection, _process):
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
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Send a prompt that will trigger a tool call
        tool_name = create_namespaced_name("progress_test", "progress_task")
        prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )

        # The tool should have been denied (permission cancelled)
        assert prompt_response.stopReason == END_TURN
        # Response should contain permission denied error
        response_text = prompt_response.message[0].text if prompt_response.message else ""
        assert "Permission" in response_text or "denied" in response_text.lower() or "cancelled" in response_text.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_allow_once_permits_execution_without_persistence() -> None:
    """Test that allow_once permits execution but doesn't persist."""
    client = TestClient()
    # Queue allow_once
    client.queue_permission_selected("allow_once")

    with tempfile.TemporaryDirectory() as tmpdir:
        async with spawn_agent_process(lambda _: client, *_get_fast_agent_cmd()) as (connection, _process):
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

            # Create session with temp dir as cwd
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

            # The tool should have executed successfully
            assert prompt_response.stopReason == END_TURN
            response_text = prompt_response.message[0].text if prompt_response.message else ""
            # Should have success message, not permission denied
            assert "Successfully completed" in response_text or "1 steps" in response_text

            # No auths.md file should exist (allow_once doesn't persist)
            auths_file = Path(tmpdir) / ".fast-agent" / "auths.md"
            assert not auths_file.exists()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_allow_always_persists() -> None:
    """Test that allow_always permits execution and persists."""
    client = TestClient()
    # Queue allow_always
    client.queue_permission_selected("allow_always")

    with tempfile.TemporaryDirectory() as tmpdir:
        async with spawn_agent_process(lambda _: client, *_get_fast_agent_cmd()) as (connection, _process):
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

            # Create session with temp dir as cwd
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

            # The tool should have executed successfully
            assert prompt_response.stopReason == END_TURN
            response_text = prompt_response.message[0].text if prompt_response.message else ""
            assert "Successfully completed" in response_text or "1 steps" in response_text

            # auths.md file should exist with allow_always
            auths_file = Path(tmpdir) / ".fast-agent" / "auths.md"
            assert auths_file.exists()
            content = auths_file.read_text()
            assert "allow_always" in content
            assert "progress_task" in content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reject_once_blocks_without_persistence() -> None:
    """Test that reject_once blocks execution but doesn't persist."""
    client = TestClient()
    # Queue reject_once
    client.queue_permission_selected("reject_once")

    with tempfile.TemporaryDirectory() as tmpdir:
        async with spawn_agent_process(lambda _: client, *_get_fast_agent_cmd()) as (connection, _process):
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

            # Create session with temp dir as cwd
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

            # The tool should have been rejected
            assert prompt_response.stopReason == END_TURN
            response_text = prompt_response.message[0].text if prompt_response.message else ""
            assert "Permission denied" in response_text or "denied" in response_text.lower()

            # No auths.md file should exist (reject_once doesn't persist)
            auths_file = Path(tmpdir) / ".fast-agent" / "auths.md"
            assert not auths_file.exists()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reject_always_blocks_and_persists() -> None:
    """Test that reject_always blocks execution and persists."""
    client = TestClient()
    # Queue reject_always
    client.queue_permission_selected("reject_always")

    with tempfile.TemporaryDirectory() as tmpdir:
        async with spawn_agent_process(lambda _: client, *_get_fast_agent_cmd()) as (connection, _process):
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

            # Create session with temp dir as cwd
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

            # The tool should have been rejected
            assert prompt_response.stopReason == END_TURN
            response_text = prompt_response.message[0].text if prompt_response.message else ""
            assert "Permission denied" in response_text or "denied" in response_text.lower()

            # auths.md file should exist with reject_always
            auths_file = Path(tmpdir) / ".fast-agent" / "auths.md"
            assert auths_file.exists()
            content = auths_file.read_text()
            assert "reject_always" in content
            assert "progress_task" in content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_permissions_flag_disables_checks() -> None:
    """Test that --no-permissions flag allows all tool executions."""
    client = TestClient()
    # Don't queue any permission response - should not be needed

    async with spawn_agent_process(lambda _: client, *_get_fast_agent_cmd(no_permissions=True)) as (connection, _process):
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
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Send a prompt that will trigger a tool call
        tool_name = create_namespaced_name("progress_test", "progress_task")
        prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )

        # The tool should have executed without permission request
        assert prompt_response.stopReason == END_TURN
        response_text = prompt_response.message[0].text if prompt_response.message else ""
        # Should have success message
        assert "Successfully completed" in response_text or "1 steps" in response_text
