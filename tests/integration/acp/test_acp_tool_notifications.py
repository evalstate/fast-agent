"""
Integration tests for ACP tool call notifications.

Tests that tool call progress is properly reported to the ACP client via
sessionUpdate notifications with tool_call and tool_call_update types.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from acp.helpers import text_block
from acp.schema import ClientCapabilities, FileSystemCapability, Implementation, StopReason
from acp.stdio import spawn_agent_process

from fast_agent.mcp.common import create_namespaced_name

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
END_TURN: StopReason = "end_turn"


def _get_session_id(response: object) -> str:
    return getattr(response, "session_id", None) or getattr(response, "sessionId")


def _get_stop_reason(response: object) -> str | None:
    return getattr(response, "stop_reason", None) or getattr(response, "stopReason", None)


def _get_session_update_type(update: object) -> str | None:
    if hasattr(update, "sessionUpdate"):
        return update.sessionUpdate
    if isinstance(update, dict):
        return update.get("sessionUpdate")
    return None
FAST_AGENT_CMD = (
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
    # Disable permissions for these tests as they focus on notifications, not permissions
    "--no-permissions",
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_tool_call_notifications() -> None:
    """Test that tool calls generate appropriate ACP notifications."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize
        init_response = await connection.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=False,
            ),
            client_info=Implementation(name="pytest-client", version="0.0.1"),
        )
        assert getattr(init_response, "protocol_version", None) == 1 or getattr(
            init_response, "protocolVersion", None
        ) == 1

        # Create session
        session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
        session_id = _get_session_id(session_response)
        assert session_id

        # Send a prompt that will trigger a tool call
        # Using the ***CALL_TOOL directive that the passthrough model supports
        tool_name = create_namespaced_name("progress_test", "progress_task")
        prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 3}}'
        prompt_response = await connection.prompt(session_id=session_id, prompt=[text_block(prompt_text)])
        assert _get_stop_reason(prompt_response) == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client, count=5, timeout=3.0)

        # Check notifications for tool_call and tool_call_update types
        tool_notifications = [
            n
            for n in client.notifications
            if _get_session_update_type(n["update"]) in ["tool_call", "tool_call_update"]
        ]

        # Should have at least one tool_call notification
        assert len(tool_notifications) > 0, "Expected tool call notifications"

        # First notification should be tool_call (initial)
        first_tool_notif = tool_notifications[0]["update"]
        assert _get_session_update_type(first_tool_notif) == "tool_call"
        assert hasattr(first_tool_notif, "toolCallId")
        assert hasattr(first_tool_notif, "title")
        assert hasattr(first_tool_notif, "kind")
        assert hasattr(first_tool_notif, "status")

        # Status should be pending initially
        assert first_tool_notif.status == "pending"

        # Subsequent notifications should be tool_call_update
        if len(tool_notifications) > 1:
            for notif in tool_notifications[1:]:
                assert _get_session_update_type(notif["update"]) == "tool_call_update"
                update = notif["update"]
                assert hasattr(update, "toolCallId") or hasattr(update, "tool_call_id")
                assert hasattr(update, "status")

            # Last notification should be completed or failed
            last_status = tool_notifications[-1]["update"].status
            assert last_status in ["completed", "failed"], (
                f"Expected final status, got {last_status}"
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_tool_progress_updates() -> None:
    """Test that tool progress updates are sent via tool_call_update notifications."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize
        await connection.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=False,
            ),
            client_info=Implementation(name="pytest-client", version="0.0.1"),
        )

        # Create session
        session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
        session_id = _get_session_id(session_response)

        # Call a tool that reports progress
        tool_name = create_namespaced_name("progress_test", "progress_task")
        prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 5}}'
        await connection.prompt(session_id=session_id, prompt=[text_block(prompt_text)])

        # Wait for multiple progress updates
        await _wait_for_notifications(client, count=7, timeout=5.0)

        # Check for progress updates
        tool_updates = [
            n
            for n in client.notifications
            if _get_session_update_type(n["update"]) == "tool_call_update"
        ]

        # Should have received progress updates
        assert len(tool_updates) > 0, "Expected tool progress updates"

        # Updates should have content with progress messages
        updates_with_content = [
            n for n in tool_updates if hasattr(n["update"], "content") and n["update"].content
        ]

        # At least some updates should have progress content
        # (MCP progress notifications include messages)
        assert len(updates_with_content) > 0, "Expected progress updates with content"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_tool_kinds_inferred() -> None:
    """Test that tool kinds are properly inferred from tool names."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize
        await connection.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=False,
            ),
            client_info=Implementation(name="pytest-client", version="0.0.1"),
        )

        # Create session
        session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
        session_id = _get_session_id(session_response)

        # Call a tool - progress_task should be inferred as "other"
        tool_name = create_namespaced_name("progress_test", "progress_task")
        prompt_text = f'***CALL_TOOL {tool_name} {{"steps": 2}}'
        await connection.prompt(session_id=session_id, prompt=[text_block(prompt_text)])

        # Wait for notifications
        await _wait_for_notifications(client, count=3, timeout=3.0)

        # Find the initial tool_call notification
        tool_call_notif = next(
            (
                n
                for n in client.notifications
                if _get_session_update_type(n["update"]) == "tool_call"
            ),
            None,
        )

        assert tool_call_notif is not None, "Expected tool_call notification"
        assert hasattr(tool_call_notif["update"], "kind")
        # progress_task doesn't match any specific pattern, should be "other"
        assert tool_call_notif["update"].kind == "other"


async def _wait_for_notifications(client: TestClient, count: int = 1, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive specified number of notifications."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if len(client.notifications) >= count:
            return
        await asyncio.sleep(0.05)
    # Don't raise error, just return - some tests may not reach the expected count
    # and that's okay for progress-based tests where timing can vary
