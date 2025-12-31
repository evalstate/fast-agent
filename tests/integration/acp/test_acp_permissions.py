"""
Integration tests for ACP tool call permissions.

Tests that permission requests are sent and handled correctly
according to the ACP protocol.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from acp.helpers import text_block

from fast_agent.mcp.common import create_namespaced_name

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))


if TYPE_CHECKING:
    from acp.client.connection import ClientSideConnection
    from acp.schema import InitializeResponse, StopReason
    from test_client import TestClient

pytestmark = pytest.mark.asyncio(loop_scope="module")

END_TURN: StopReason = "end_turn"


def _get_session_id(response: object) -> str:
    """Helper to support both camelCase and snake_case session id fields."""
    return getattr(response, "session_id", None) or getattr(response, "sessionId")


def _get_stop_reason(response: object) -> str | None:
    """Helper to support both camelCase and snake_case stop reason fields."""
    return getattr(response, "stop_reason", None) or getattr(response, "stopReason", None)


async def _wait_for_notifications(client: TestClient, count: int = 1, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive specified number of notifications."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if len(client.notifications) >= count:
            return
        await asyncio.sleep(0.05)


def _tool_executed_successfully(client: TestClient) -> bool:
    """Check if tool executed successfully by examining notifications.

    Look for a tool_call_update notification with status 'completed'.
    """
    for n in client.notifications:
        update = n["update"]
        if hasattr(update, "sessionUpdate") and update.sessionUpdate == "tool_call_update":
            if hasattr(update, "status") and update.status == "completed":
                return True
    return False


def _tool_was_denied(client: TestClient) -> bool:
    """Check if tool execution was denied by examining notifications.

    Look for a tool_call_update notification with status 'failed'.
    """
    for n in client.notifications:
        update = n["update"]
        if hasattr(update, "sessionUpdate") and update.sessionUpdate == "tool_call_update":
            if hasattr(update, "status") and update.status == "failed":
                return True
    return False


@pytest.mark.integration
async def test_permission_request_sent_when_tool_called(
    acp_permissions: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> None:
    """Test that a permission request is sent when a tool is called."""
    connection, client, _init_response = acp_permissions
    # Queue a rejection so the tool doesn't actually execute
    client.queue_permission_cancelled()

    # Create session
    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = _get_session_id(session_response)

    # Send a prompt that will trigger a tool call
    tool_name = create_namespaced_name("progress_test", "progress_task")
    prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
    prompt_response = await connection.prompt(
        session_id=session_id, prompt=[text_block(prompt_text)]
    )

    # The tool should have been denied (permission cancelled)
    assert _get_stop_reason(prompt_response) == END_TURN

    # Wait for notifications to be received
    await _wait_for_notifications(client, count=2, timeout=3.0)

    # Tool should not have executed successfully (permission was cancelled)
    assert not _tool_executed_successfully(client), (
        "Tool should not have executed when permission cancelled"
    )


@pytest.mark.integration
async def test_allow_once_permits_execution_without_persistence(
    acp_permissions: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> None:
    """Test that allow_once permits execution but doesn't persist."""
    connection, client, _init_response = acp_permissions
    # Queue allow_once
    client.queue_permission_selected("allow_once")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create session with temp dir as cwd
        session_response = await connection.new_session(mcp_servers=[], cwd=tmpdir)
        session_id = _get_session_id(session_response)

        # Send a prompt that will trigger a tool call
        tool_name = create_namespaced_name("progress_test", "progress_task")
        prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
        prompt_response = await connection.prompt(
            session_id=session_id, prompt=[text_block(prompt_text)]
        )

        # The tool should have executed successfully
        assert _get_stop_reason(prompt_response) == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client, count=3, timeout=3.0)

        # Tool should have executed successfully
        assert _tool_executed_successfully(client), "Tool should have executed with allow_once"

        # No auths.md file should exist (allow_once doesn't persist)
        auths_file = Path(tmpdir) / ".fast-agent" / "auths.md"
        assert not auths_file.exists()


@pytest.mark.integration
async def test_allow_always_persists(
    acp_permissions: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> None:
    """Test that allow_always permits execution and persists."""
    connection, client, _init_response = acp_permissions
    # Queue allow_always
    client.queue_permission_selected("allow_always")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create session with temp dir as cwd
        session_response = await connection.new_session(mcp_servers=[], cwd=tmpdir)
        session_id = _get_session_id(session_response)

        # Send a prompt that will trigger a tool call
        tool_name = create_namespaced_name("progress_test", "progress_task")
        prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
        prompt_response = await connection.prompt(
            session_id=session_id, prompt=[text_block(prompt_text)]
        )

        # The tool should have executed successfully
        assert _get_stop_reason(prompt_response) == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client, count=3, timeout=3.0)

        # Tool should have executed successfully
        assert _tool_executed_successfully(client), "Tool should have executed with allow_always"

        # auths.md file should exist with allow_always
        auths_file = Path(tmpdir) / ".fast-agent" / "auths.md"
        assert auths_file.exists()
        content = auths_file.read_text()
        assert "allow_always" in content
        assert "progress_task" in content


@pytest.mark.integration
async def test_reject_once_blocks_without_persistence(
    acp_permissions: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> None:
    """Test that reject_once blocks execution but doesn't persist."""
    connection, client, _init_response = acp_permissions
    # Queue reject_once
    client.queue_permission_selected("reject_once")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create session with temp dir as cwd
        session_response = await connection.new_session(mcp_servers=[], cwd=tmpdir)
        session_id = _get_session_id(session_response)

        # Send a prompt that will trigger a tool call
        tool_name = create_namespaced_name("progress_test", "progress_task")
        prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
        prompt_response = await connection.prompt(
            session_id=session_id, prompt=[text_block(prompt_text)]
        )

        # The tool should have been rejected
        assert _get_stop_reason(prompt_response) == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client, count=2, timeout=3.0)

        # Tool should not have executed successfully
        assert not _tool_executed_successfully(client), (
            "Tool should not have executed with reject_once"
        )

        # No auths.md file should exist (reject_once doesn't persist)
        auths_file = Path(tmpdir) / ".fast-agent" / "auths.md"
        assert not auths_file.exists()


@pytest.mark.integration
async def test_reject_always_blocks_and_persists(
    acp_permissions: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> None:
    """Test that reject_always blocks execution and persists."""
    connection, client, _init_response = acp_permissions
    # Queue reject_always
    client.queue_permission_selected("reject_always")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create session with temp dir as cwd
        session_response = await connection.new_session(mcp_servers=[], cwd=tmpdir)
        session_id = _get_session_id(session_response)

        # Send a prompt that will trigger a tool call
        tool_name = create_namespaced_name("progress_test", "progress_task")
        prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
        prompt_response = await connection.prompt(
            session_id=session_id, prompt=[text_block(prompt_text)]
        )

        # The tool should have been rejected
        assert _get_stop_reason(prompt_response) == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client, count=2, timeout=3.0)

        # Tool should not have executed successfully
        assert not _tool_executed_successfully(client), (
            "Tool should not have executed with reject_always"
        )

        # auths.md file should exist with reject_always
        auths_file = Path(tmpdir) / ".fast-agent" / "auths.md"
        assert auths_file.exists()
        content = auths_file.read_text()
        assert "reject_always" in content
        assert "progress_task" in content


@pytest.mark.integration
async def test_no_permissions_flag_disables_checks(
    acp_permissions_no_perms: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> None:
    """Test that --no-permissions flag allows all tool executions."""
    connection, client, _init_response = acp_permissions_no_perms
    # Don't queue any permission response - should not be needed

    # Create session
    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = _get_session_id(session_response)

    # Send a prompt that will trigger a tool call
    tool_name = create_namespaced_name("progress_test", "progress_task")
    prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 1}}'
    prompt_response = await connection.prompt(
        session_id=session_id, prompt=[text_block(prompt_text)]
    )

    # The tool should have executed without permission request
    assert _get_stop_reason(prompt_response) == END_TURN

    # Wait for notifications
    await _wait_for_notifications(client, count=3, timeout=3.0)

    # Tool should have executed successfully without needing permission
    assert _tool_executed_successfully(client), (
        "Tool should have executed with --no-permissions flag"
    )
