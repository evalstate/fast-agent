"""
Integration tests for ACP runtime telemetry.

Tests that ACP runtime operations (execute, read_text_file, write_text_file)
trigger tool call notifications via the Tool Calling and Workflow Telemetry system.
"""

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
        "fast-agent-acp-runtime-telemetry-test",
    ]
    if with_shell:
        cmd.append("--shell")
    return tuple(cmd)


async def _wait_for_notifications(client: TestClient, count: int = 1, timeout: float = 3.0) -> None:
    """Wait for the ACP client to receive specified number of notifications."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if len(client.notifications) >= count:
            return
        await asyncio.sleep(0.05)
    # Don't raise error, just return - some tests may not reach the expected count


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_terminal_runtime_telemetry() -> None:
    """Test that terminal execute operations trigger tool call notifications."""
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
                terminal=True,
            ),
            clientInfo=Implementation(name="pytest-telemetry-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)
        assert init_response.protocolVersion == 1

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Call the execute tool via passthrough model
        prompt_text = '***CALL_TOOL execute {"command": "echo test"}'
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client, count=2, timeout=3.0)

        # Check that we received tool call notifications
        tool_notifications = [
            n
            for n in client.notifications
            if hasattr(n.update, "sessionUpdate")
            and n.update.sessionUpdate in ["tool_call", "tool_call_update"]
        ]

        # Should have at least one tool_call notification
        assert len(tool_notifications) > 0, "Expected tool call notifications for execute"

        # First notification should be tool_call (initial)
        first_notif = tool_notifications[0]
        assert first_notif.update.sessionUpdate == "tool_call"
        assert hasattr(first_notif.update, "toolCallId")
        assert hasattr(first_notif.update, "title")
        assert hasattr(first_notif.update, "kind")
        assert hasattr(first_notif.update, "status")

        # Verify the title contains "execute" and "acp_terminal"
        title = first_notif.update.title
        assert "execute" in title.lower() or "acp_terminal" in title.lower()

        # Status should start as pending
        assert first_notif.update.status == "pending"

        # Last notification should be completed or failed
        if len(tool_notifications) > 1:
            last_status = tool_notifications[-1].update.status
            assert last_status in ["completed", "failed"], (
                f"Expected final status, got {last_status}"
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_filesystem_read_runtime_telemetry() -> None:
    """Test that read_text_file operations trigger tool call notifications."""
    client = TestClient()

    # Set up a test file in the client
    test_path = "/test/sample.txt"
    test_content = "Hello from test file!"
    client.files[test_path] = test_content

    async with spawn_agent_process(lambda _: client, *get_fast_agent_cmd(with_shell=False)) as (
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
            clientInfo=Implementation(name="pytest-telemetry-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Call the read_text_file tool via passthrough model
        prompt_text = f'***CALL_TOOL read_text_file {{"path": "{test_path}"}}'
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client, count=2, timeout=3.0)

        # Check that we received tool call notifications
        tool_notifications = [
            n
            for n in client.notifications
            if hasattr(n.update, "sessionUpdate")
            and n.update.sessionUpdate in ["tool_call", "tool_call_update"]
        ]

        # Should have at least one tool_call notification
        assert len(tool_notifications) > 0, "Expected tool call notifications for read_text_file"

        # First notification should be tool_call (initial)
        first_notif = tool_notifications[0]
        assert first_notif.update.sessionUpdate == "tool_call"
        assert hasattr(first_notif.update, "toolCallId")
        assert hasattr(first_notif.update, "title")

        # Verify the title contains "read_text_file" and "acp_filesystem"
        title = first_notif.update.title
        assert "read_text_file" in title.lower() or "acp_filesystem" in title.lower()

        # Last notification should be completed
        if len(tool_notifications) > 1:
            last_status = tool_notifications[-1].update.status
            assert last_status in ["completed", "failed"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_filesystem_write_runtime_telemetry() -> None:
    """Test that write_text_file operations trigger tool call notifications."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *get_fast_agent_cmd(with_shell=False)) as (
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
            clientInfo=Implementation(name="pytest-telemetry-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Call the write_text_file tool via passthrough model
        test_path = "/test/output.txt"
        test_content = "Test content from tool call"
        prompt_text = f'***CALL_TOOL write_text_file {{"path": "{test_path}", "content": "{test_content}"}}'
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client, count=2, timeout=3.0)

        # Check that we received tool call notifications
        tool_notifications = [
            n
            for n in client.notifications
            if hasattr(n.update, "sessionUpdate")
            and n.update.sessionUpdate in ["tool_call", "tool_call_update"]
        ]

        # Should have at least one tool_call notification
        assert len(tool_notifications) > 0, "Expected tool call notifications for write_text_file"

        # First notification should be tool_call (initial)
        first_notif = tool_notifications[0]
        assert first_notif.update.sessionUpdate == "tool_call"
        assert hasattr(first_notif.update, "toolCallId")
        assert hasattr(first_notif.update, "title")

        # Verify the title contains "write_text_file" and "acp_filesystem"
        title = first_notif.update.title
        assert "write_text_file" in title.lower() or "acp_filesystem" in title.lower()

        # Verify the file was written
        assert test_path in client.files
        assert client.files[test_path] == test_content

        # Last notification should be completed
        if len(tool_notifications) > 1:
            last_status = tool_notifications[-1].update.status
            assert last_status in ["completed", "failed"]
