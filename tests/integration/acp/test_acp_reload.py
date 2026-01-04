from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from acp.helpers import text_block
from acp.schema import ClientCapabilities, FileSystemCapability, Implementation
from acp.stdio import spawn_agent_process

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _write_card(path: Path, instruction: str) -> None:
    content = f"name: reload-test\ninstruction: |\n  {instruction}\n"
    path.write_text(content, encoding="utf-8")


def _get_session_update_type(update: Any) -> str | None:
    if hasattr(update, "sessionUpdate"):
        return update.sessionUpdate
    if isinstance(update, dict):
        return update.get("sessionUpdate")
    return None


def _get_update_text(update: Any) -> str | None:
    if hasattr(update, "content"):
        content = update.content
    elif isinstance(update, dict):
        content = update.get("content")
    else:
        content = None
    if not content:
        return None
    return getattr(content, "text", None)


def _get_stop_reason(response: object) -> str | None:
    return getattr(response, "stop_reason", None) or getattr(response, "stopReason", None)


async def _wait_for_message_text(
    client: TestClient,
    session_id: str,
    needle: str,
    *,
    start_index: int = 0,
    timeout: float = 2.0,
) -> str:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for notification in client.notifications[start_index:]:
            if notification["session_id"] != session_id:
                continue
            update = notification["update"]
            if _get_session_update_type(update) != "agent_message_chunk":
                continue
            text = _get_update_text(update)
            if text and needle in text:
                return text
        await asyncio.sleep(0.05)
    raise AssertionError(f"Expected message containing {needle!r}")


@pytest.mark.integration
async def test_acp_reload_agent_cards(tmp_path: Path) -> None:
    card_dir = tmp_path / "cards"
    card_dir.mkdir()
    card_path = card_dir / "reload_agent.yaml"
    _write_card(card_path, "You are a helpful assistant.")

    client = TestClient()
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
        "fast-agent-acp-reload-test",
        "--card",
        str(card_dir),
        "--reload",
    ]

    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
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

        start_index = len(client.notifications)
        response = await connection.prompt(
            session_id=session_id, prompt=[text_block("/reload")]
        )
        assert _get_stop_reason(response) == "end_turn"
        await _wait_for_message_text(
            client,
            session_id,
            "No AgentCard changes detected.",
            start_index=start_index,
        )

        _write_card(card_path, "You are a reloaded assistant.")
        await asyncio.sleep(0.05)

        start_index = len(client.notifications)
        response = await connection.prompt(
            session_id=session_id, prompt=[text_block("/reload")]
        )
        assert _get_stop_reason(response) == "end_turn"
        await _wait_for_message_text(
            client,
            session_id,
            "Reloaded AgentCards.",
            start_index=start_index,
        )
