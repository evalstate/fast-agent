"""
Integration tests for ACP runtime telemetry.

Tests that ACP runtime operations (execute, read_text_file, write_text_file)
trigger tool call notifications via the Tool Calling and Workflow Telemetry system.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from acp.helpers import text_block
from acp.schema import StopReason

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

pytestmark = pytest.mark.asyncio(loop_scope="module")

END_TURN: StopReason = "end_turn"


def _get_session_id(response: object) -> str:
    return getattr(response, "session_id", None) or getattr(response, "sessionId")


def _get_stop_reason(response: object) -> str | None:
    return getattr(response, "stop_reason", None) or getattr(response, "stopReason", None)


def _get_session_update_type(update: Any) -> str | None:
    session_update = getattr(update, "sessionUpdate", None)
    if session_update is not None:
        return str(session_update)
    if isinstance(update, dict):
        result = update.get("sessionUpdate")
        return str(result) if result is not None else None
    return None


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
async def test_acp_terminal_runtime_telemetry(
    acp_runtime_telemetry_shell: tuple[object, TestClient, object],
) -> None:
    """Test that terminal execute operations trigger tool call notifications."""
    connection, client, init_response = acp_runtime_telemetry_shell

    assert getattr(init_response, "protocol_version", None) == 1 or getattr(
        init_response, "protocolVersion", None
    ) == 1

    # Create session
    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = _get_session_id(session_response)
    assert session_id

    # Call the execute tool via passthrough model
    prompt_text = '***CALL_TOOL execute {"command": "echo test"}'
    prompt_response = await connection.prompt(session_id=session_id, prompt=[text_block(prompt_text)])
    assert _get_stop_reason(prompt_response) == END_TURN

    # Wait for notifications
    await _wait_for_notifications(client, count=2, timeout=3.0)

    # Check that we received tool call notifications
    tool_notifications = [
        n
        for n in client.notifications
        if _get_session_update_type(n["update"]) in ["tool_call", "tool_call_update"]
    ]

    # Should have at least one tool_call notification
    assert len(tool_notifications) > 0, "Expected tool call notifications for execute"

    # First notification should be tool_call (initial)
    first_notif = tool_notifications[0]["update"]
    assert _get_session_update_type(first_notif) == "tool_call"
    assert hasattr(first_notif, "toolCallId")
    assert hasattr(first_notif, "title")
    assert hasattr(first_notif, "kind")
    assert hasattr(first_notif, "status")

    # Verify the title contains "execute" and "acp_terminal"
    title = first_notif.title
    assert "execute" in title.lower() or "acp_terminal" in title.lower()

    # Status should start as pending
    assert first_notif.status == "pending"

    # Last notification should be completed or failed
    if len(tool_notifications) > 1:
        last_status = tool_notifications[-1]["update"].status
        assert last_status in ["completed", "failed"], f"Expected final status, got {last_status}"


@pytest.mark.integration
async def test_acp_filesystem_read_runtime_telemetry(
    acp_runtime_telemetry: tuple[object, TestClient, object],
) -> None:
    """Test that read_text_file operations trigger tool call notifications."""
    connection, client, _init_response = acp_runtime_telemetry

    # Set up a test file in the client
    test_path = "/test/sample.txt"
    test_content = "Hello from test file!"
    client.files[test_path] = test_content

    # Create session
    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = _get_session_id(session_response)

    # Call the read_text_file tool via passthrough model
    prompt_text = f'***CALL_TOOL read_text_file {{"path": "{test_path}"}}'
    prompt_response = await connection.prompt(session_id=session_id, prompt=[text_block(prompt_text)])
    assert _get_stop_reason(prompt_response) == END_TURN

    # Wait for notifications
    await _wait_for_notifications(client, count=2, timeout=3.0)

    # Check that we received tool call notifications
    tool_notifications = [
        n
        for n in client.notifications
        if _get_session_update_type(n["update"]) in ["tool_call", "tool_call_update"]
    ]

    # Should have at least one tool_call notification
    assert len(tool_notifications) > 0, "Expected tool call notifications for read_text_file"

    # First notification should be tool_call (initial)
    first_notif = tool_notifications[0]["update"]
    assert _get_session_update_type(first_notif) == "tool_call"
    assert hasattr(first_notif, "toolCallId")
    assert hasattr(first_notif, "title")

    # Verify the title contains "read_text_file" and "acp_filesystem"
    title = first_notif.title
    assert "read_text_file" in title.lower() or "acp_filesystem" in title.lower()

    # Last notification should be completed
    if len(tool_notifications) > 1:
        last_status = tool_notifications[-1]["update"].status
        assert last_status in ["completed", "failed"]


@pytest.mark.integration
async def test_acp_filesystem_write_runtime_telemetry(
    acp_runtime_telemetry: tuple[object, TestClient, object],
) -> None:
    """Test that write_text_file operations trigger tool call notifications."""
    connection, client, _init_response = acp_runtime_telemetry

    # Create session
    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = _get_session_id(session_response)

    # Call the write_text_file tool via passthrough model
    test_path = "/test/output.txt"
    test_content = "Test content from tool call"
    prompt_text = f'***CALL_TOOL write_text_file {{"path": "{test_path}", "content": "{test_content}"}}'
    prompt_response = await connection.prompt(session_id=session_id, prompt=[text_block(prompt_text)])
    assert _get_stop_reason(prompt_response) == END_TURN

    # Wait for notifications
    await _wait_for_notifications(client, count=2, timeout=3.0)

    # Check that we received tool call notifications
    tool_notifications = [
        n
        for n in client.notifications
        if _get_session_update_type(n["update"]) in ["tool_call", "tool_call_update"]
    ]

    # Should have at least one tool_call notification
    assert len(tool_notifications) > 0, "Expected tool call notifications for write_text_file"

    # First notification should be tool_call (initial)
    first_notif = tool_notifications[0]["update"]
    assert _get_session_update_type(first_notif) == "tool_call"
    assert hasattr(first_notif, "toolCallId")
    assert hasattr(first_notif, "title")

    # Verify the title contains "write_text_file" and "acp_filesystem"
    title = first_notif.title
    assert "write_text_file" in title.lower() or "acp_filesystem" in title.lower()

    # Verify the file was written
    assert test_path in client.files
    assert client.files[test_path] == test_content

    # Last notification should be completed
    if len(tool_notifications) > 1:
        last_status = tool_notifications[-1]["update"].status
        assert last_status in ["completed", "failed"]
