from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal, cast

import pytest
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.config import get_settings, update_global_settings
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.session import (
    SessionManager,
    apply_session_window,
    get_session_manager,
    load_session_snapshot,
    reset_session_manager,
)

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    def __init__(
        self,
        *,
        name: str,
        instruction: str,
        history: list[PromptMessageExtended] | None = None,
    ) -> None:
        self.name = name
        self.instruction = instruction
        self.config = AgentConfig(name, instruction=instruction, model="passthrough")
        self.llm = None
        self.usage_accumulator = None
        self.message_history = list(history or [])

    def clear(self, *, clear_prompts: bool = False) -> None:
        del clear_prompts
        self.message_history.clear()

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction

    def set_model(self, model: str | None) -> None:
        self.config.model = model


def _message(role: Literal["user", "assistant"], text: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role=role,
        content=[TextContent(type="text", text=text)],
    )


def _message_texts(agent: _Agent) -> list[str]:
    return [
        content.text
        for message in agent.message_history
        for content in message.content
        if isinstance(content, TextContent)
    ]


def test_prune_sessions_skips_pinned(tmp_path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    override = old_settings.model_copy(
        update={
            "environment_dir": str(env_dir),
            "session_history_window": 1,
        }
    )
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = get_session_manager()
        first = manager.create_session()
        first.set_pinned(True)

        second = manager.create_session()
        third = manager.create_session()

        sessions = manager.list_sessions()
        names = {session.name for session in sessions}

        assert first.info.name in names
        assert third.info.name in names
        assert second.info.name not in names
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


def test_get_session_manager_normalizes_relative_environment_dir(tmp_path) -> None:
    original_env = os.environ.get("ENVIRONMENT_DIR")
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir(parents=True)
    second_cwd.mkdir(parents=True)

    os.environ["ENVIRONMENT_DIR"] = ".dev"
    reset_session_manager()

    try:
        manager_first = get_session_manager(cwd=first_cwd)
        normalized_env = os.environ.get("ENVIRONMENT_DIR")
        assert normalized_env is not None
        assert normalized_env == str((first_cwd / ".dev").resolve())

        manager_second = get_session_manager(cwd=second_cwd)
        assert manager_second is manager_first
        assert manager_second.base_dir == (first_cwd / ".dev" / "sessions").resolve()
        assert manager_second.workspace_dir == second_cwd.resolve()
    finally:
        reset_session_manager()
        if original_env is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = original_env


def test_get_session_manager_refreshes_workspace_dir_for_shared_environment(tmp_path) -> None:
    original_env = os.environ.get("ENVIRONMENT_DIR")
    env_dir = tmp_path / "shared-env"
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir(parents=True)
    second_cwd.mkdir(parents=True)

    os.environ["ENVIRONMENT_DIR"] = str(env_dir)
    reset_session_manager()

    try:
        manager_first = get_session_manager(cwd=first_cwd)
        assert manager_first.workspace_dir == first_cwd.resolve()

        manager_second = get_session_manager(cwd=second_cwd)
        assert manager_second is manager_first
        assert manager_second.base_dir == (env_dir / "sessions").resolve()
        assert manager_second.workspace_dir == second_cwd.resolve()
    finally:
        reset_session_manager()
        if original_env is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = original_env


def test_apply_session_window_appends_pinned_overflow(tmp_path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    override = old_settings.model_copy(
        update={
            "environment_dir": str(env_dir),
            "session_history_window": 2,
        }
    )
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = get_session_manager()
        oldest = manager.create_session()
        oldest.set_pinned(True)
        middle = manager.create_session()
        newest = manager.create_session()

        visible = apply_session_window(manager.list_sessions())
        names = [session.name for session in visible]

        assert names == [newest.info.name, middle.info.name, oldest.info.name]
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


def test_resolve_session_name_ordinal_includes_pinned_overflow(tmp_path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    override = old_settings.model_copy(
        update={
            "environment_dir": str(env_dir),
            "session_history_window": 2,
        }
    )
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = get_session_manager()
        oldest = manager.create_session()
        oldest.set_pinned(True)
        manager.create_session()
        manager.create_session()

        assert manager.resolve_session_name("3") == oldest.info.name
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


def test_load_session_marks_loaded_session_as_latest(tmp_path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    override = old_settings.model_copy(update={"environment_dir": str(env_dir)})
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = get_session_manager()
        older = manager.create_session()
        newer = manager.create_session()

        base_time = datetime(2026, 4, 15, 12, 0, 0)
        older.info.last_activity = base_time
        older._save_metadata()
        newer.info.last_activity = base_time + timedelta(minutes=1)
        newer._save_metadata()

        loaded = manager.load_session(older.info.name)
        assert loaded is not None

        latest = manager.load_latest_session()
        assert latest is not None
        assert latest.info.name == older.info.name
        assert loaded.info.last_activity > newer.info.last_activity
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


@pytest.mark.asyncio
async def test_resume_session_agents_uses_hydrator_active_agent_and_prompt_restore(
    tmp_path,
) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()

    saved_foo = _Agent(
        name="foo",
        instruction="Stored foo prompt",
        history=[_message("user", "hello foo"), _message("assistant", "done foo")],
    )
    saved_bar = _Agent(
        name="bar",
        instruction="Stored bar prompt",
        history=[_message("user", "hello bar"), _message("assistant", "done bar")],
    )

    await session.save_history(
        cast("AgentProtocol", saved_foo),
        agent_registry={
            "foo": cast("AgentProtocol", saved_foo),
            "bar": cast("AgentProtocol", saved_bar),
        },
    )
    await session.save_history(
        cast("AgentProtocol", saved_bar),
        agent_registry={
            "foo": cast("AgentProtocol", saved_foo),
            "bar": cast("AgentProtocol", saved_bar),
        },
    )

    runtime_foo = _Agent(name="foo", instruction="Changed foo prompt")
    runtime_bar = _Agent(name="bar", instruction="Changed bar prompt")

    result = await manager.resume_session_agents_async(
        {
            "foo": cast("AgentProtocol", runtime_foo),
            "bar": cast("AgentProtocol", runtime_bar),
        },
        session.info.name,
        fallback_agent_name="foo",
    )

    assert result is not None
    assert set(result.loaded) == {"foo", "bar"}
    assert result.active_agent == "bar"
    assert result.warnings == []
    assert runtime_foo.instruction == "Stored foo prompt"
    assert runtime_bar.instruction == "Stored bar prompt"
    assert _message_texts(runtime_foo) == ["hello foo", "done foo"]
    assert _message_texts(runtime_bar) == ["hello bar", "done bar"]


def test_resume_session_includes_hydrator_warnings_in_notices(tmp_path) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    payload = {
        "schema_version": 2,
        "session_id": session.info.name,
        "created_at": session.info.created_at.isoformat(),
        "last_activity": session.info.last_activity.isoformat(),
        "metadata": {},
        "continuation": {
            "active_agent": "missing",
            "cwd": None,
            "lineage": {},
            "agents": {
                "foo": {
                    "history_file": "missing.json",
                    "resolved_prompt": "Stored foo prompt",
                    "card_provenance": [],
                    "attachment_refs": [],
                    "model_overlay_refs": [],
                }
            },
        },
        "analysis": {},
    }
    (session.directory / "session.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    runtime_foo = _Agent(name="foo", instruction="Runtime foo prompt")
    resumed = manager.resume_session(cast("AgentProtocol", runtime_foo), session.info.name)

    assert resumed is not None
    _resumed_session, history_path, notices = resumed
    assert history_path is None
    assert runtime_foo.instruction == "Stored foo prompt"
    assert any("Persisted active agent 'missing'" in notice for notice in notices)
    assert any("Persisted history file 'missing.json' is missing" in notice for notice in notices)


@pytest.mark.asyncio
async def test_fork_current_session_clones_typed_snapshot_state(tmp_path) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    source = manager.create_session(metadata={"acp_session_id": "acp-123"})

    saved_foo = _Agent(
        name="foo",
        instruction="Stored foo prompt",
        history=[_message("user", "hello foo"), _message("assistant", "done foo")],
    )
    saved_bar = _Agent(
        name="bar",
        instruction="Stored bar prompt",
        history=[_message("user", "hello bar"), _message("assistant", "done bar")],
    )

    await source.save_history(
        cast("AgentProtocol", saved_foo),
        agent_registry={
            "foo": cast("AgentProtocol", saved_foo),
            "bar": cast("AgentProtocol", saved_bar),
        },
    )
    await source.save_history(
        cast("AgentProtocol", saved_bar),
        agent_registry={
            "foo": cast("AgentProtocol", saved_foo),
            "bar": cast("AgentProtocol", saved_bar),
        },
    )

    forked = manager.fork_current_session(title="Forked title")

    assert forked is not None
    assert forked.info.name != source.info.name

    forked_snapshot = load_session_snapshot(
        json.loads((forked.directory / "session.json").read_text(encoding="utf-8"))
    )

    assert forked_snapshot.session_id == forked.info.name
    assert forked_snapshot.metadata.title == "Forked title"
    assert forked_snapshot.continuation.active_agent == "bar"
    assert forked_snapshot.continuation.lineage.forked_from == source.info.name
    assert forked_snapshot.continuation.lineage.acp_session_id is None
    assert set(forked_snapshot.continuation.agents) == {"foo", "bar"}
    assert forked_snapshot.continuation.agents["foo"].resolved_prompt == "Stored foo prompt"
    assert forked_snapshot.continuation.agents["bar"].resolved_prompt == "Stored bar prompt"

    foo_history = forked_snapshot.continuation.agents["foo"].history_file
    bar_history = forked_snapshot.continuation.agents["bar"].history_file
    assert foo_history is not None
    assert bar_history is not None
    assert (forked.directory / foo_history).exists()
    assert (forked.directory / bar_history).exists()
