from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from acp.helpers import text_block
from acp.schema import McpServerStdio

from fast_agent.mcp.common import create_namespaced_name

TEST_DIR = Path(__file__).parent
PROGRESS_SERVER_PATH = (TEST_DIR.parent / "api" / "mcp_progress_server.py").resolve()

if TYPE_CHECKING:
    from acp.client.connection import ClientSideConnection
    from acp.schema import InitializeResponse, StopReason
    from test_client import TestClient

pytestmark = pytest.mark.asyncio(loop_scope="module")

END_TURN: StopReason = "end_turn"


def _get_stop_reason(response: object) -> str | None:
    return getattr(response, "stop_reason", None) or getattr(response, "stopReason", None)


def _get_session_update_type(update: Any) -> str | None:
    if hasattr(update, "sessionUpdate"):
        return update.sessionUpdate
    if isinstance(update, dict):
        return update.get("sessionUpdate")
    return None


async def _wait_for_agent_message(
    client: TestClient,
    session_id: str,
    *,
    timeout: float = 4.0,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if any(
            notification
            for notification in client.notifications
            if notification["session_id"] == session_id
            and _get_session_update_type(notification["update"]) == "agent_message_chunk"
        ):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Expected an ACP agent_message_chunk update")


@pytest.mark.integration
async def test_acp_session_new_attaches_client_supplied_mcp_servers(
    acp_permissions_no_perms: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> None:
    connection, client, _init_response = acp_permissions_no_perms

    server_name = "session_progress"
    session_response = await connection.new_session(
        cwd=str(TEST_DIR),
        mcp_servers=[
            McpServerStdio(
                name=server_name,
                command=sys.executable,
                args=[str(PROGRESS_SERVER_PATH)],
                env=[],
            )
        ],
    )
    session_id = session_response.session_id

    tool_name = create_namespaced_name(server_name, "progress_task")
    prompt_response = await connection.prompt(
        session_id=session_id,
        prompt=[text_block(f'***CALL_TOOL {tool_name} {{"steps": 1}}')],
    )

    assert _get_stop_reason(prompt_response) == END_TURN

    await _wait_for_agent_message(client, session_id)

    tool_updates = [
        notification["update"]
        for notification in client.notifications
        if notification["session_id"] == session_id
        and _get_session_update_type(notification["update"]) in {"tool_call", "tool_call_update"}
    ]

    assert tool_updates
    assert any(server_name in str(getattr(update, "title", "")) for update in tool_updates)
    assert tool_updates[-1].status == "completed"
    assert getattr(tool_updates[-1], "raw_output", None) == "Successfully completed 1 steps"
