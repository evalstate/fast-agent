from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import pytest
from acp.helpers import text_block
from acp.schema import ClientCapabilities, FileSystemCapability, Implementation, StopReason
from acp.stdio import spawn_agent_process

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
END_TURN: StopReason = "end_turn"
FAST_AGENT_CMD = (
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
    "fast-agent-acp-test",
)


def _extract_status_line(meta: object) -> str | None:
    if not isinstance(meta, dict):
        return None
    field_meta = meta.get("field_meta")
    if isinstance(field_meta, dict):
        metrics = field_meta.get("openhands.dev/metrics")
    else:
        metrics = meta.get("openhands.dev/metrics")
    if not isinstance(metrics, dict):
        return None
    status_line = metrics.get("status_line")
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
@pytest.mark.asyncio
async def test_acp_status_line_meta_is_emitted() -> None:
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        await connection.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=False,
            ),
            client_info=Implementation(name="pytest-client", version="0.0.1"),
        )

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
