from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from acp.helpers import text_block
from acp.schema import ClientCapabilities, FileSystemCapability, Implementation
from acp.stdio import spawn_agent_process

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

if TYPE_CHECKING:
    from acp.client.connection import ClientSideConnection

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.integration
async def test_acp_prompt_saves_session_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = TEST_DIR / "fastagent.config.yaml"
    cmd = [
        sys.executable,
        "-m",
        "fast_agent.cli",
        "serve",
        "--config-path",
        str(config_path),
        "--transport",
        "acp",
        "--model",
        "passthrough",
        "--name",
        "fast-agent-acp-session-test",
    ]

    client = TestClient()
    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
        await _initialize_connection(connection)
        session_response = await connection.new_session(mcp_servers=[], cwd=str(tmp_path))
        prompt_text = "session history integration test"
        await connection.prompt(
            session_id=session_response.session_id,
            prompt=[text_block(prompt_text)],
        )

    sessions_root = tmp_path / ".fast-agent" / "sessions"
    assert sessions_root.exists()
    session_dirs = [path for path in sessions_root.iterdir() if path.is_dir()]
    assert len(session_dirs) == 1
    session_dir = session_dirs[0]
    session_meta_path = session_dir / "session.json"
    assert session_meta_path.exists()
    metadata = json.loads(session_meta_path.read_text())
    history_files = metadata.get("history_files") or []
    assert history_files
    for filename in history_files:
        assert (session_dir / filename).exists()


async def _initialize_connection(connection: "ClientSideConnection") -> None:
    await connection.initialize(
        protocol_version=1,
        client_capabilities=ClientCapabilities(
            fs=FileSystemCapability(read_text_file=True, write_text_file=True),
            terminal=False,
        ),
        client_info=Implementation(name="pytest-client", version="0.0.1"),
    )
