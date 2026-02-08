from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from acp.helpers import text_block
from acp.schema import ClientCapabilities, FileSystemCapability, Implementation, StopReason
from acp.stdio import spawn_agent_process

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

if TYPE_CHECKING:
    from acp.schema import InitializeResponse

pytestmark = pytest.mark.asyncio(loop_scope="module")

END_TURN: StopReason = "end_turn"


@pytest.mark.integration
async def test_acp_watch_allows_prompt_reload(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    card_path = agents_dir / "watcher.md"
    card_path.write_text(
        "---\n"
        "type: agent\n"
        "name: watcher\n"
        "---\n"
        "Echo test.\n",
        encoding="utf-8",
    )

    cmd = (
        sys.executable,
        "-m",
        "fast_agent.cli",
        "serve",
        "--config-path",
        str(TEST_DIR / "fastagent.config.yaml"),
        "--transport",
        "acp",
        "--model",
        "passthrough",
        "--name",
        "fast-agent-acp-watch",
        "--card",
        str(agents_dir),
        "--watch",
    )

    client = TestClient()
    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
        init_response: InitializeResponse = await connection.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=False,
            ),
            client_info=Implementation(name="pytest-client", version="0.0.1"),
        )
        assert init_response

        session_response = await connection.new_session(mcp_servers=[], cwd=str(tmp_path))
        session_id = session_response.session_id
        assert session_id

        prompt_response = await connection.prompt(
            session_id=session_id, prompt=[text_block("watch test")]
        )
        assert prompt_response.stop_reason == END_TURN

        card_path.write_text(
            "---\n"
            "type: agent\n"
            "name: watcher\n"
            "---\n"
            "Echo test updated.\n",
            encoding="utf-8",
        )
        await asyncio.sleep(0.25)

        prompt_response = await connection.prompt(
            session_id=session_id, prompt=[text_block("watch test 2")]
        )
        assert prompt_response.stop_reason == END_TURN
