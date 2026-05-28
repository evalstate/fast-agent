from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from acp.exceptions import RequestError
from acp.helpers import text_block
from acp.schema import ClientCapabilities, FileSystemCapabilities, Implementation
from acp.stdio import spawn_agent_process
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
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


def _as_object_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _persisted_history_files(payload: dict[str, object]) -> list[str]:
    history_files = payload.get("history_files")
    if isinstance(history_files, list):
        return [item for item in history_files if isinstance(item, str)]

    continuation = _as_object_dict(payload.get("continuation"))
    if continuation is None:
        return []
    agents = _as_object_dict(continuation.get("agents"))
    if agents is None:
        return []

    selected: list[str] = []
    for agent_payload in agents.values():
        agent_entry = _as_object_dict(agent_payload)
        if agent_entry is None:
            continue
        history_file = agent_entry.get("history_file")
        if isinstance(history_file, str) and history_file not in selected:
            selected.append(history_file)
    return selected


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
    assert metadata["schema_version"] == 2
    active_agent = metadata["continuation"]["active_agent"]
    assert isinstance(active_agent, str)
    assert active_agent
    active_agent_payload = metadata["continuation"]["agents"][active_agent]
    assert active_agent_payload["resolved_prompt"]
    assert active_agent_payload["model"] == "passthrough"
    assert active_agent_payload["provider"] == "fast-agent"
    assert isinstance(active_agent_payload["request_settings"], dict)
    assert metadata["continuation"]["lineage"]["acp_session_id"] == session_response.session_id
    history_files = _persisted_history_files(metadata)
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
async def test_acp_prompt_saves_session_history_in_workspace_store_for_session_cwd(
    tmp_path: Path,
) -> None:
    server_cwd = tmp_path / "server"
    session_cwd = tmp_path / "workspace"
    server_cwd.mkdir()
    session_cwd.mkdir()

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
        "fast-agent-acp-session-cwd-test",
    ]

    client = TestClient()
    async with spawn_agent_process(
        lambda _: client,
        *cmd,
        cwd=server_cwd,
    ) as (connection, _process):
        await _initialize_connection(connection)
        session_response = await connection.new_session(
            mcp_servers=[],
            cwd=str(session_cwd),
        )
        await connection.prompt(
            session_id=session_response.session_id,
            prompt=[text_block("persist under request cwd")],
        )
        await _wait_for_session_info_update(client, session_response.session_id)
        listed = await connection.list_sessions(cwd=str(session_cwd))

    session_dir = session_cwd / ".fast-agent" / "sessions" / session_response.session_id
    assert session_dir.exists()
    metadata = json.loads((session_dir / "session.json").read_text())
    assert metadata["continuation"]["cwd"] == str(session_cwd.resolve())
    assert any(info.session_id == session_response.session_id for info in listed.sessions)


@pytest.mark.integration
async def test_acp_prompt_saves_session_history_in_configured_environment_dir(
    tmp_path: Path,
) -> None:
    server_cwd = tmp_path / "server"
    session_cwd = tmp_path / "workspace"
    server_cwd.mkdir()
    session_cwd.mkdir()

    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        (TEST_DIR / "fastagent.config.yaml").read_text()
        + '\nenvironment_dir: ".custom-fast-agent"\n'
    )

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
        "fast-agent-acp-session-custom-env",
    ]

    client = TestClient()
    async with spawn_agent_process(
        lambda _: client,
        *cmd,
        cwd=server_cwd,
    ) as (connection, _process):
        await _initialize_connection(connection)
        session_response = await connection.new_session(
            mcp_servers=[],
            cwd=str(session_cwd),
        )
        await connection.prompt(
            session_id=session_response.session_id,
            prompt=[text_block("persist under configured env dir")],
        )
        await _wait_for_session_info_update(client, session_response.session_id)
        listed = await connection.list_sessions(cwd=str(session_cwd))

    session_dir = (
        session_cwd
        / ".custom-fast-agent"
        / "sessions"
        / session_response.session_id
    )
    assert session_dir.exists()
    assert not (session_cwd / ".fast-agent" / "sessions" / session_response.session_id).exists()
    metadata = json.loads((session_dir / "session.json").read_text())
    assert metadata["continuation"]["cwd"] == str(session_cwd.resolve())
    assert any(info.session_id == session_response.session_id for info in listed.sessions)


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
async def test_acp_prompt_emits_session_info_update_when_only_updated_at_changes(
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
        "fast-agent-acp-session-updated-at",
    ]

    client = TestClient()
    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
        await _initialize_connection(connection)
        session_response = await connection.new_session(mcp_servers=[], cwd=str(tmp_path))
        await connection.prompt(
            session_id=session_response.session_id,
            prompt=[text_block("first prompt title seed")],
        )
        await _wait_for_session_info_update(client, session_response.session_id)
        first_count = len(
            [
                note
                for note in client.notifications
                if note["session_id"] == session_response.session_id
                and _get_session_update_type(note["update"]) == "session_info_update"
            ]
        )

        await connection.prompt(
            session_id=session_response.session_id,
            prompt=[text_block("second prompt keeps same title")],
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2.0
        while loop.time() < deadline:
            update_count = len(
                [
                    note
                    for note in client.notifications
                    if note["session_id"] == session_response.session_id
                    and _get_session_update_type(note["update"]) == "session_info_update"
                ]
            )
            if update_count > first_count:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("Expected a second session_info_update after follow-up prompt")


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
                self.instruction = "Alpha agent."
                self.config = AgentConfig("alpha", instruction=self.instruction, model="passthrough")
                self.llm = None
                self.usage_accumulator = None
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
        session = manager.create_session(
            metadata={"title": "ACP list test", "cwd": str(tmp_path.resolve())}
        )
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
async def test_acp_session_list_reads_workspace_scoped_sessions_when_server_runs_elsewhere(
    tmp_path: Path,
) -> None:
    server_cwd = tmp_path / "server"
    session_cwd = tmp_path / "workspace"
    server_cwd.mkdir()
    session_cwd.mkdir()

    original_cwd = Path.cwd()
    session_manager_module._session_manager = None
    session = None
    try:
        os.chdir(session_cwd)
        manager = get_session_manager(
            cwd=session_cwd,
            environment_override=".fast-agent",
        )
        session = manager.create_session(metadata={"title": "Workspace scoped list"})
    finally:
        session_manager_module._session_manager = None
        os.chdir(original_cwd)

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
        "fast-agent-acp-session-list-workspace-scope",
    ]

    client = TestClient()
    async with spawn_agent_process(
        lambda _: client,
        *cmd,
        cwd=server_cwd,
    ) as (connection, _process):
        await _initialize_connection(connection)
        response = await connection.list_sessions(cwd=str(session_cwd))

    assert response.sessions
    assert session is not None
    matching = [info for info in response.sessions if info.session_id == session.info.name]
    assert matching
    assert matching[0].title == "Workspace scoped list"
    assert Path(matching[0].cwd) == session_cwd.resolve()


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
                self.instruction = "Alpha agent."
                self.config = AgentConfig("alpha", instruction=self.instruction, model="passthrough")
                self.llm = None
                self.usage_accumulator = None
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


@pytest.mark.integration
async def test_acp_load_session_reads_workspace_scoped_session_when_server_runs_elsewhere(
    tmp_path: Path,
) -> None:
    server_cwd = tmp_path / "server"
    session_cwd = tmp_path / "workspace"
    cards_dir = tmp_path / "cards"
    server_cwd.mkdir()
    session_cwd.mkdir()
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
    session_manager_module._session_manager = None
    session = None
    try:
        os.chdir(session_cwd)
        manager = get_session_manager(
            cwd=session_cwd,
            environment_override=".fast-agent",
        )
        session = manager.create_session(metadata={"title": "Workspace scoped load"})
        history_messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="hello from workspace")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="hi from workspace")],
            ),
        ]

        class StubAgent:
            def __init__(self) -> None:
                self.name = "alpha"
                self.instruction = "Alpha agent."
                self.config = AgentConfig("alpha", instruction=self.instruction, model="passthrough")
                self.llm = None
                self.usage_accumulator = None
                self.message_history = history_messages

        await session.save_history(cast("AgentProtocol", StubAgent()))
    finally:
        session_manager_module._session_manager = None
        os.chdir(original_cwd)

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
        "--name",
        "fast-agent-acp-session-load-workspace-scope",
    ]

    client = TestClient()
    async with spawn_agent_process(
        lambda _: client,
        *cmd,
        cwd=server_cwd,
    ) as (connection, _process):
        await _initialize_connection(connection)
        await connection.load_session(
            session_id=session.info.name,
            cwd=str(session_cwd),
            mcp_servers=[],
        )
        await _wait_for_session_updates(client, session.info.name)

    updates = [
        note["update"]
        for note in client.notifications
        if note["session_id"] == session.info.name
    ]
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
    assert "hello from workspace" in user_texts
    assert "hi from workspace" in agent_texts


@pytest.mark.integration
async def test_acp_load_session_missing_returns_resource_not_found(
    tmp_path: Path,
) -> None:
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
        "fast-agent-acp-load-missing",
    ]

    client = TestClient()
    missing_session_id = "missing-session"
    environment_dir = tmp_path / ".fast-agent"
    async with spawn_agent_process(
        lambda _: client,
        *cmd,
        cwd=tmp_path,
        env={"ENVIRONMENT_DIR": str(environment_dir)},
    ) as (connection, _process):
        await _initialize_connection(connection)
        with pytest.raises(RequestError) as exc_info:
            await connection.load_session(
                session_id=missing_session_id,
                cwd=str(tmp_path),
                mcp_servers=[],
            )

    assert exc_info.value.code == -32002
    data = exc_info.value.data
    assert isinstance(data, dict)
    assert data["reason"] == "Session not found"
    assert data["uri"] == missing_session_id
    assert (
        data["details"]
        == f"Session {missing_session_id} could not be resolved from {tmp_path}"
    )


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
            fs=FileSystemCapabilities(read_text_file=True, write_text_file=True),
            terminal=False,
        ),
        client_info=Implementation(name="pytest-client", version="0.0.1"),
    )
