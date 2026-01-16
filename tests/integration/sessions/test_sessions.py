import json
import re
import sys
from pathlib import Path

import pytest

from fast_agent import FastAgent
from fast_agent.mcp.prompt_serialization import load_messages
from fast_agent.session import get_session_manager


def _create_fast_agent(config_path: Path) -> FastAgent:
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        return FastAgent(
            "Test Agent",
            config_path=str(config_path),
            ignore_unknown_args=True,
        )
    finally:
        sys.argv = original_argv


def _reset_session_manager() -> None:
    from fast_agent.session import session_manager as session_manager_module

    session_manager_module._session_manager = None


def _write_config(path: Path, *, session_history: bool | None = None) -> None:
    lines = ["default_model: passthrough"]
    if session_history is not None:
        value = "true" if session_history else "false"
        lines.append(f"session_history: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_history_autosave_default_on(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_session_manager()
    config_path = tmp_path / "fastagent.config.yaml"
    _write_config(config_path)

    fast = _create_fast_agent(config_path)
    agent_name = None

    @fast.agent(model="passthrough")
    async def agent_function():
        nonlocal agent_name
        async with fast.run() as agent:
            agent_name = agent.default.name
            await agent.send("Hello session")

    await agent_function()
    assert agent_name

    sessions_root = tmp_path / ".fast-agent" / "sessions"
    session_dirs = [path for path in sessions_root.iterdir() if path.is_dir()]
    assert len(session_dirs) == 1

    session_id = session_dirs[0].name
    assert re.fullmatch(r"[A-Za-z0-9]{6}", session_id)

    metadata_path = session_dirs[0] / "session.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    session_meta = metadata["metadata"]
    history_map = session_meta["last_history_by_agent"]
    history_filename = history_map[agent_name]
    assert agent_name in history_filename
    assert session_meta["first_user_preview"] == "Hello session"

    history_files = list(session_dirs[0].glob("history_*.json"))
    assert history_files

    messages = load_messages(str(history_files[0]))
    user_messages = [msg for msg in messages if msg.role == "user"]
    assert user_messages
    assert "Hello session" in user_messages[-1].all_text()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_latest_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_session_manager()
    config_path = tmp_path / "fastagent.config.yaml"
    _write_config(config_path)

    fast = _create_fast_agent(config_path)

    @fast.agent(model="passthrough")
    async def agent_function():
        async with fast.run() as agent:
            agent_obj = agent.default
            await agent.send("Resume me")

            manager = get_session_manager()
            assert manager.current_session is not None
            history_path = manager.current_session.latest_history_path(agent_obj.name)
            assert history_path is not None

            saved_messages = load_messages(str(history_path))
            agent_obj.clear(clear_prompts=True)

            result = manager.resume_session_agents(
                agent._agents,
                None,
                default_agent_name=agent_obj.name,
            )
            assert result is not None
            _, loaded, _ = result
            assert agent_obj.name in loaded
            assert len(agent_obj.message_history) == len(saved_messages)
            assert agent_obj.message_history[-1].all_text() == saved_messages[-1].all_text()

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_warns_on_missing_agents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_session_manager()
    config_path = tmp_path / "fastagent.config.yaml"
    _write_config(config_path)

    fast = _create_fast_agent(config_path)

    @fast.agent(name="foo", model="passthrough", default=True)
    async def foo_agent():
        pass

    @fast.agent(name="bar", model="passthrough")
    async def bar_agent():
        pass

    async with fast.run() as agent:
        await agent.send("Hello foo")
        await agent.send("Hello bar", agent_name="bar")

    _reset_session_manager()
    fast2 = _create_fast_agent(config_path)

    @fast2.agent(name="foo", model="passthrough", default=True)
    async def foo_only():
        async with fast2.run() as agent:
            manager = get_session_manager()
            result = manager.resume_session_agents(
                agent._agents,
                None,
                default_agent_name="foo",
            )
            assert result is not None
            _, loaded, missing = result
            assert "foo" in loaded
            assert "bar" in missing


@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_history_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_session_manager()
    config_path = tmp_path / "fastagent.config.yaml"
    _write_config(config_path, session_history=False)

    fast = _create_fast_agent(config_path)

    @fast.agent(model="passthrough")
    async def agent_function():
        async with fast.run() as agent:
            await agent.send("No save")

    await agent_function()

    sessions_root = tmp_path / ".fast-agent" / "sessions"
    assert not sessions_root.exists()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_title_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_session_manager()
    config_path = tmp_path / "fastagent.config.yaml"
    _write_config(config_path)

    fast = _create_fast_agent(config_path)

    @fast.agent(model="passthrough")
    async def agent_function():
        async with fast.run() as agent:
            await agent.send("Title me")
            manager = get_session_manager()
            assert manager.current_session is not None
            manager.current_session.set_title("My Session Title")

    await agent_function()

    sessions_root = tmp_path / ".fast-agent" / "sessions"
    session_dir = next(path for path in sessions_root.iterdir() if path.is_dir())
    metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert metadata["metadata"]["title"] == "My Session Title"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_fork_copies_latest_histories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset_session_manager()
    config_path = tmp_path / "fastagent.config.yaml"
    _write_config(config_path)

    fast = _create_fast_agent(config_path)

    @fast.agent(name="foo", model="passthrough", default=True)
    async def foo_agent():
        pass

    @fast.agent(name="bar", model="passthrough")
    async def bar_agent():
        pass

    async with fast.run() as agent:
        await agent.send("Hello foo")
        await agent.send("Hello bar", agent_name="bar")

        manager = get_session_manager()
        source = manager.current_session
        assert source is not None
        forked = manager.fork_current_session(title="Forked")
        assert forked is not None
        assert forked.info.name != source.info.name
        assert forked.info.metadata["forked_from"] == source.info.name
        history_map = forked.info.metadata.get("last_history_by_agent")
        assert isinstance(history_map, dict)
        assert set(history_map.keys()) == {"foo", "bar"}
        for filename in history_map.values():
            assert (forked.directory / filename).exists()
