from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from acp.helpers import text_block

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))


if TYPE_CHECKING:
    from acp.client.connection import ClientSideConnection
    from acp.schema import InitializeResponse, StopReason
    from test_client import TestClient

pytestmark = pytest.mark.asyncio(loop_scope="module")

END_TURN: StopReason = "end_turn"


def _extract_status_line(meta: object) -> str | None:
    if not isinstance(meta, dict):
        return None
    meta_dict = cast("dict[str, object]", meta)
    field_meta = meta_dict.get("field_meta")
    if isinstance(field_meta, dict):
        field_meta_dict = cast("dict[str, object]", field_meta)
        metrics = field_meta_dict.get("openhands.dev/metrics")
    else:
        metrics = meta_dict.get("openhands.dev/metrics")
    if not isinstance(metrics, dict):
        return None
    metrics_dict = cast("dict[str, object]", metrics)
    status_line = metrics_dict.get("status_line")
    if isinstance(status_line, str) and status_line.strip():
        return status_line
    return None


async def _wait_for_status_line(client: TestClient, timeout: float = 2.0) -> str:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for notification in client.notifications:
            status_line = _extract_status_line(notification.get("meta"))
            if status_line:
                return status_line
        await asyncio.sleep(0.05)
    raise AssertionError("Expected status line metadata in session updates")


@pytest.mark.integration
async def test_acp_status_line_meta_is_emitted(
    acp_basic: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> None:
    connection, client, _init_response = acp_basic

    session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
    session_id = session_response.session_id
    assert session_id

    prompt_text = "status line integration test"
    prompt_response = await connection.prompt(
        session_id=session_id, prompt=[text_block(prompt_text)]
    )
    assert prompt_response.stop_reason == END_TURN

    status_line = await _wait_for_status_line(client)
    assert re.search(r"\d[\d,]* in, \d[\d,]* out", status_line), status_line
