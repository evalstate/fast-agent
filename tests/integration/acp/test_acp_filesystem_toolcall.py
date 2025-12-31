"""Integration tests for ACP filesystem tool calling."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

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


async def _wait_for_notifications(client: TestClient, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive at least one sessionUpdate."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if client.notifications:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Expected streamed session updates")


@pytest.mark.integration
async def test_acp_filesystem_read_tool_call(
    acp_filesystem_toolcall: tuple[object, TestClient, object],
) -> None:
    """Test that read_text_file tool can be called via passthrough model."""
    connection, client, init_response = acp_filesystem_toolcall

    # Set up a test file in the client
    test_path = "/test/sample.txt"
    test_content = "Hello from test file!"
    client.files[test_path] = test_content

    assert getattr(init_response, "protocol_version", None) == 1 or getattr(
        init_response, "protocolVersion", None
    ) == 1
    assert (
        getattr(init_response, "agent_capabilities", None)
        or getattr(init_response, "agentCapabilities", None)
        is not None
    )

    # Create session
    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = _get_session_id(session_response)
    assert session_id

    # Use passthrough model's ***CALL_TOOL directive to invoke read_text_file
    prompt_text = f'***CALL_TOOL read_text_file {{"path": "{test_path}"}}'
    prompt_response = await connection.prompt(session_id=session_id, prompt=[text_block(prompt_text)])

    # Should complete successfully
    assert _get_stop_reason(prompt_response) == END_TURN

    # Wait for notifications
    await _wait_for_notifications(client)

    # Verify we got notifications
    assert len(client.notifications) > 0

    # Verify the file content appears in the notifications
    # This confirms the read_text_file tool was called and returned content
    notification_text = str(client.notifications)
    assert test_content in notification_text or test_path in notification_text


@pytest.mark.integration
async def test_acp_filesystem_write_tool_call(
    acp_filesystem_toolcall: tuple[object, TestClient, object],
) -> None:
    """Test that write_text_file tool can be called via passthrough model."""
    connection, client, _init_response = acp_filesystem_toolcall

    # Create session
    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = _get_session_id(session_response)
    assert session_id

    # Use passthrough model's ***CALL_TOOL directive to invoke write_text_file
    test_path = "/test/output.txt"
    test_content = "Test content from tool call"
    prompt_text = f'***CALL_TOOL write_text_file {{"path": "{test_path}", "content": "{test_content}"}}'

    prompt_response = await connection.prompt(session_id=session_id, prompt=[text_block(prompt_text)])

    # Should complete successfully
    assert _get_stop_reason(prompt_response) == END_TURN

    # Wait for notifications
    await _wait_for_notifications(client)

    # Verify the file was written
    assert test_path in client.files
    assert client.files[test_path] == test_content
