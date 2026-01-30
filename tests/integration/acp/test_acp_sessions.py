from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from acp.helpers import text_block
from acp.schema import ClientCapabilities, FileSystemCapability, Implementation
from acp.stdio import spawn_agent_process
from mcp.types import TextContent

from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.session import get_session_manager
from fast_agent.session import session_manager as session_manager_module

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

if TYPE_CHECKING:
    from acp.client.connection import ClientSideConnection

    from fast_agent.interfaces import AgentProtocol


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
        await _wait_for_session_info_update(client, session_response.session_id)

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

    info_updates = [
        note["update"]
        for note in client.notifications
        if note["session_id"] == session_response.session_id
        and _get_session_update_type(note["update"]) == "session_info_update"
    ]
    assert info_updates
    assert _get_update_title(info_updates[-1]) == prompt_text


@pytest.mark.integration
async def test_acp_session_title_command_emits_info_update(
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
        "fast-agent-acp-session-title",
    ]

    client = TestClient()
    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
        await _initialize_connection(connection)
        session_response = await connection.new_session(mcp_servers=[], cwd=str(tmp_path))
        title = "ACP session title"
        await connection.prompt(
            session_id=session_response.session_id,
            prompt=[text_block(f"/session title {title}")],
        )
        await _wait_for_session_info_update(client, session_response.session_id)

    updates = [
        note["update"]
        for note in client.notifications
        if note["session_id"] == session_response.session_id
        and _get_session_update_type(note["update"]) == "session_info_update"
    ]
    assert updates
    assert _get_update_title(updates[-1]) == title


@pytest.mark.integration
async def test_acp_session_resume_emits_current_mode_update(
    tmp_path: Path,
) -> None:
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    alpha_card = cards_dir / "alpha.md"
    beta_card = cards_dir / "beta.md"

    alpha_card.write_text(
        "---\n"
        "type: agent\n"
        "name: alpha\n"
        "model: passthrough\n"
        "instruction: Alpha agent.\n"
        "---\n"
    )
    beta_card.write_text(
        "---\n"
        "type: agent\n"
        "name: beta\n"
        "default: true\n"
        "model: passthrough\n"
        "instruction: Beta agent.\n"
        "---\n"
    )

    original_cwd = Path.cwd()
    original_env_dir = os.environ.get("ENVIRONMENT_DIR")
    environment_dir = tmp_path / ".fast-agent"
    os.environ["ENVIRONMENT_DIR"] = str(environment_dir)
    os.chdir(tmp_path)
    session_manager_module._session_manager = None
    try:
        manager = get_session_manager()
        session = manager.create_session()
        history_message = PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="resume me")],
        )

        class StubAgent:
            def __init__(self) -> None:
                self.name = "alpha"
                self.message_history = [history_message]

        await session.save_history(cast("AgentProtocol", StubAgent()))
    finally:
        session_manager_module._session_manager = None
        os.chdir(original_cwd)
        if original_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = original_env_dir

    assert session is not None

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
        "--agent-cards",
        str(alpha_card),
        "--agent-cards",
        str(beta_card),
    ]

    client = TestClient()
    async with spawn_agent_process(
        lambda _: client,
        *cmd,
        cwd=tmp_path,
        env={"ENVIRONMENT_DIR": str(environment_dir)},
    ) as (connection, _process):
        await _initialize_connection(connection)
        session_response = await connection.new_session(mcp_servers=[], cwd=str(tmp_path))
        await connection.prompt(
            session_id=session_response.session_id,
            prompt=[text_block(f"/session resume {session.info.name}")],
        )

    mode_updates = [
        note["update"]
        for note in client.notifications
        if getattr(note["update"], "session_update", None) == "current_mode_update"
    ]
    assert mode_updates
    assert mode_updates[-1].current_mode_id == "alpha"


@pytest.mark.integration
async def test_acp_session_list_returns_saved_sessions(
    tmp_path: Path,
) -> None:
    original_cwd = Path.cwd()
    original_env_dir = os.environ.get("ENVIRONMENT_DIR")
    environment_dir = tmp_path / ".fast-agent"
    os.environ["ENVIRONMENT_DIR"] = str(environment_dir)
    os.chdir(tmp_path)
    session_manager_module._session_manager = None
    session = None
    try:
        manager = get_session_manager()
        session = manager.create_session(metadata={"title": "ACP list test"})
    finally:
        session_manager_module._session_manager = None
        os.chdir(original_cwd)
        if original_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = original_env_dir

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
        "fast-agent-acp-session-list",
    ]

    client = TestClient()
    async with spawn_agent_process(
        lambda _: client,
        *cmd,
        cwd=tmp_path,
        env={"ENVIRONMENT_DIR": str(environment_dir)},
    ) as (connection, _process):
        await _initialize_connection(connection)
        response = await connection.list_sessions(cwd=str(tmp_path))

    assert response.sessions
    assert session is not None
    matching = [info for info in response.sessions if info.session_id == session.info.name]
    assert matching
    assert matching[0].title == "ACP list test"
    assert Path(matching[0].cwd) == tmp_path.resolve()


@pytest.mark.integration
async def test_acp_load_session_streams_history(
    tmp_path: Path,
) -> None:
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    alpha_card = cards_dir / "alpha.md"
    alpha_card.write_text(
        "---\n"
        "type: agent\n"
        "name: alpha\n"
        "default: true\n"
        "model: passthrough\n"
        "instruction: Alpha agent.\n"
        "---\n"
    )

    original_cwd = Path.cwd()
    original_env_dir = os.environ.get("ENVIRONMENT_DIR")
    environment_dir = tmp_path / ".fast-agent"
    os.environ["ENVIRONMENT_DIR"] = str(environment_dir)
    os.chdir(tmp_path)
    session_manager_module._session_manager = None
    session = None
    try:
        manager = get_session_manager()
        session = manager.create_session(metadata={"title": "History load"})
        history_messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="hello")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="hi")],
            ),
        ]

        class StubAgent:
            def __init__(self) -> None:
                self.name = "alpha"
                self.message_history = history_messages

        await session.save_history(cast("AgentProtocol", StubAgent()))
    finally:
        session_manager_module._session_manager = None
        os.chdir(original_cwd)
        if original_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = original_env_dir

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
        "--agent-cards",
        str(alpha_card),
    ]

    client = TestClient()
    async with spawn_agent_process(
        lambda _: client,
        *cmd,
        cwd=tmp_path,
        env={"ENVIRONMENT_DIR": str(environment_dir)},
    ) as (connection, _process):
        await _initialize_connection(connection)
        await connection.load_session(
            session_id=session.info.name,
            cwd=str(tmp_path),
            mcp_servers=[],
        )
        await _wait_for_session_updates(client, session.info.name)

    updates = [
        note["update"]
        for note in client.notifications
        if note["session_id"] == session.info.name
    ]
    update_types = [_get_session_update_type(update) for update in updates]
    assert "user_message_chunk" in update_types
    assert "agent_message_chunk" in update_types

    user_texts = [
        _get_update_text(update)
        for update in updates
        if _get_session_update_type(update) == "user_message_chunk"
    ]
    agent_texts = [
        _get_update_text(update)
        for update in updates
        if _get_session_update_type(update) == "agent_message_chunk"
    ]
    assert "hello" in user_texts
    assert "hi" in agent_texts


def _get_session_update_type(update: Any) -> str | None:
    if hasattr(update, "session_update"):
        return update.session_update
    if hasattr(update, "sessionUpdate"):
        return update.sessionUpdate
    if isinstance(update, dict):
        return update.get("session_update") or update.get("sessionUpdate")
    return None


def _get_update_text(update: Any) -> str | None:
    content = update.content if hasattr(update, "content") else None
    if content is None and isinstance(update, dict):
        content = update.get("content")
    if content is None:
        return None
    if hasattr(content, "text"):
        return content.text
    if isinstance(content, dict):
        return content.get("text")
    return None


def _get_update_title(update: Any) -> str | None:
    if hasattr(update, "title"):
        return update.title
    if isinstance(update, dict):
        return update.get("title")
    return None


async def _wait_for_session_updates(
    client: TestClient,
    session_id: str,
    timeout: float = 2.0,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if any(
            note["session_id"] == session_id
            and _get_session_update_type(note["update"])
            in {"user_message_chunk", "agent_message_chunk"}
            for note in client.notifications
        ):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Expected session updates after load_session")


async def _wait_for_session_info_update(
    client: TestClient,
    session_id: str,
    timeout: float = 2.0,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if any(
            note["session_id"] == session_id
            and _get_session_update_type(note["update"]) == "session_info_update"
            for note in client.notifications
        ):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Expected session_info_update after session title change")


async def _initialize_connection(connection: "ClientSideConnection") -> None:
    await connection.initialize(
        protocol_version=1,
        client_capabilities=ClientCapabilities(
            fs=FileSystemCapability(read_text_file=True, write_text_file=True),
            terminal=False,
        ),
        client_info=Implementation(name="pytest-client", version="0.0.1"),
    )
