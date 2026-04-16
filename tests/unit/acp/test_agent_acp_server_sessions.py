from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from acp.exceptions import RequestError
from acp.schema import SessionModeState

from fast_agent.acp.server.agent_acp_server import ACPSessionState, AgentACPServer
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.session import (
    SessionAgentSnapshot,
    SessionContinuationSnapshot,
    SessionHydrationResult,
    SessionSnapshot,
)
from fast_agent.session.session_manager import SessionInfo

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    instruction = ""
    acp_commands: dict[str, object] = {}

    def __init__(self) -> None:
        self.config = SimpleNamespace(default=False)
        self.message_history: list[object] = []

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction


def _build_instance(agent_names: list[str]) -> AgentInstance:
    agents = {
        name: cast("AgentProtocol", _Agent())
        for name in agent_names
    }
    return AgentInstance(
        app=AgentApp(agents),
        agents=agents,
        registry_version=0,
    )


def _build_server(
    instance: AgentInstance,
    *,
    created_instance: AgentInstance | None = None,
) -> AgentACPServer:
    async def create_instance() -> AgentInstance:
        return created_instance or instance

    async def dispose_instance(_instance: AgentInstance) -> None:
        return None

    return AgentACPServer(
        bootstrap_instance=instance,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="test",
        permissions_enabled=False,
    )


def _hydration_result(
    *,
    session: object,
    session_id: str,
    active_agent: str | None,
    loaded_agents: dict[str, Path] | None = None,
    restored_prompts: dict[str, str] | None = None,
    prompts: dict[str, str] | None = None,
) -> SessionHydrationResult:
    now = datetime.now()
    snapshot_agents = {
        agent_name: SessionAgentSnapshot(resolved_prompt=prompt)
        for agent_name, prompt in (prompts or {}).items()
    }
    return SessionHydrationResult(
        session=cast("Any", session),
        snapshot=SessionSnapshot(
            session_id=session_id,
            created_at=now,
            last_activity=now,
            continuation=SessionContinuationSnapshot(agents=snapshot_agents),
        ),
        loaded_agents=loaded_agents or {},
        restored_prompts=restored_prompts or {},
        skipped_agents=[],
        missing_history_files=[],
        active_agent=active_agent,
    )


@pytest.mark.asyncio
async def test_load_session_falls_back_when_primary_agent_was_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_instance = _build_instance(["main"])
    server = _build_server(primary_instance)
    refreshed_instance = _build_instance(["renamed"])
    session_state = ACPSessionState(
        session_id="s-1",
        instance=refreshed_instance,
    )

    async def fake_initialize_session_state(
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[Any],
    ) -> tuple[ACPSessionState, SessionModeState]:
        del session_id, cwd, mcp_servers
        return session_state, SessionModeState(available_modes=[], current_mode_id="main")

    hydrate_calls: list[str | None] = []

    def fake_get_session_manager(
        *,
        cwd: Any = None,
        environment_override: Any = None,
        respect_env_override: bool = True,
    ) -> Any:
        del cwd, environment_override, respect_env_override

        class _Manager:
            workspace_dir = Path("/workspace")
            base_dir = workspace_dir / ".fast-agent" / "sessions"

            def resolve_session_name(self, name: str) -> str:
                return name

            def get_session(self, name: str) -> Any:
                del name
                return SimpleNamespace(info=SimpleNamespace(metadata={}))

            def load_session(self, name: str) -> Any:
                return self.get_session(name)

        return _Manager()

    class _Hydrator:
        def hydrate_session(
            self,
            *,
            session: Any,
            agents: Any,
            fallback_agent_name: str | None,
        ) -> SessionHydrationResult:
            del agents
            hydrate_calls.append(fallback_agent_name)
            return _hydration_result(
                session=session,
                session_id="s-1",
                active_agent="renamed",
            )

    monkeypatch.setattr(server, "_initialize_session_state", fake_initialize_session_state)
    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        fake_get_session_manager,
    )
    monkeypatch.setattr("fast_agent.acp.server.session_store.SessionHydrator", _Hydrator)

    response = await server.load_session(
        cwd="/workspace",
        session_id="s-1",
        mcp_servers=[],
    )

    assert response is not None
    assert response.modes is not None
    assert hydrate_calls == ["renamed"]
    assert session_state.current_agent_name == "renamed"
    assert response.modes.current_mode_id == "renamed"


@pytest.mark.asyncio
async def test_new_session_recomputes_primary_agent_for_new_instances(tmp_path: Path) -> None:
    primary_instance = _build_instance(["main"])
    refreshed_instance = _build_instance(["renamed"])
    server = _build_server(primary_instance, created_instance=refreshed_instance)

    response = await server.new_session(
        cwd=str(tmp_path.resolve()),
        mcp_servers=[],
    )

    assert response.modes is not None
    assert response.modes.current_mode_id == "renamed"
    assert server.primary_agent_name == "renamed"
    session_state = server._session_state[response.session_id]
    assert session_state.current_agent_name == "renamed"


@pytest.mark.asyncio
async def test_load_session_uses_request_cwd_for_session_manager(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    primary_instance = _build_instance(["main"])
    server = _build_server(primary_instance)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_state = ACPSessionState(
        session_id="s-1",
        instance=primary_instance,
    )
    manager_cwds: list[Any] = []

    async def fake_initialize_session_state(
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[Any],
    ) -> tuple[ACPSessionState, SessionModeState]:
        del session_id, mcp_servers
        assert cwd == str(workspace.resolve())
        return session_state, SessionModeState(available_modes=[], current_mode_id="main")

    def fake_get_session_manager(
        *,
        cwd: Any = None,
        environment_override: Any = None,
        respect_env_override: bool = True,
    ) -> Any:
        manager_cwds.append((cwd, environment_override, respect_env_override))

        class _Manager:
            workspace_dir = workspace
            base_dir = workspace / ".fast-agent" / "sessions"

            def resolve_session_name(self, name: str) -> str:
                return name

            def get_session(self, name: str) -> Any:
                assert name == "s-1"
                return SimpleNamespace(
                    info=SimpleNamespace(metadata={"cwd": str(workspace.resolve())})
                )

            def load_session(self, name: str) -> Any:
                return self.get_session(name)

        return _Manager()

    class _Hydrator:
        def hydrate_session(
            self,
            *,
            session: Any,
            agents: Any,
            fallback_agent_name: str | None,
        ) -> SessionHydrationResult:
            del agents, fallback_agent_name
            return _hydration_result(
                session=session,
                session_id="s-1",
                active_agent="main",
            )

    monkeypatch.setattr(server, "_initialize_session_state", fake_initialize_session_state)
    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        fake_get_session_manager,
    )
    monkeypatch.setattr("fast_agent.acp.server.session_store.SessionHydrator", _Hydrator)

    response = await server.load_session(
        cwd=str(workspace),
        session_id="s-1",
        mcp_servers=[],
    )

    assert response is not None
    assert manager_cwds == [
        (workspace.resolve(), None, True),
        (None, None, True),
    ]


@pytest.mark.asyncio
async def test_load_session_applies_restored_prompt_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_instance = _build_instance(["main"])
    server = _build_server(primary_instance)
    session_state = ACPSessionState(
        session_id="s-1",
        instance=primary_instance,
        resolved_instructions={"main": "Current runtime prompt"},
    )

    async def fake_initialize_session_state(
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[Any],
    ) -> tuple[ACPSessionState, SessionModeState]:
        del session_id, cwd, mcp_servers
        return session_state, SessionModeState(available_modes=[], current_mode_id="main")

    def fake_get_session_manager(
        *,
        cwd: Any = None,
        environment_override: Any = None,
        respect_env_override: bool = True,
    ) -> Any:
        del cwd, environment_override, respect_env_override

        class _Manager:
            workspace_dir = Path("/workspace")
            base_dir = workspace_dir / ".fast-agent" / "sessions"

            def resolve_session_name(self, name: str) -> str:
                return name

            def get_session(self, name: str) -> Any:
                del name
                return SimpleNamespace(info=SimpleNamespace(metadata={}))

            def load_session(self, name: str) -> Any:
                return self.get_session(name)

        return _Manager()

    class _Hydrator:
        def hydrate_session(
            self,
            *,
            session: Any,
            agents: Any,
            fallback_agent_name: str | None,
        ) -> SessionHydrationResult:
            del agents, fallback_agent_name
            return _hydration_result(
                session=session,
                session_id="s-1",
                active_agent="main",
                restored_prompts={"main": "Persisted session prompt"},
                prompts={"main": "Persisted session prompt"},
            )

    monkeypatch.setattr(server, "_initialize_session_state", fake_initialize_session_state)
    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        fake_get_session_manager,
    )
    monkeypatch.setattr("fast_agent.acp.server.session_store.SessionHydrator", _Hydrator)

    response = await server.load_session(
        cwd="/workspace",
        session_id="s-1",
        mcp_servers=[],
    )

    assert response is not None
    assert session_state.resolved_instructions["main"] == "Persisted session prompt"


@pytest.mark.asyncio
async def test_load_session_does_not_cache_prompt_when_hydrator_did_not_restore_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_instance = _build_instance(["main"])
    server = _build_server(primary_instance)
    session_state = ACPSessionState(
        session_id="s-1",
        instance=primary_instance,
        resolved_instructions={"main": "Current runtime prompt"},
    )

    async def fake_initialize_session_state(
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[Any],
    ) -> tuple[ACPSessionState, SessionModeState]:
        del session_id, cwd, mcp_servers
        return session_state, SessionModeState(available_modes=[], current_mode_id="main")

    def fake_get_session_manager(
        *,
        cwd: Any = None,
        environment_override: Any = None,
        respect_env_override: bool = True,
    ) -> Any:
        del cwd, environment_override, respect_env_override

        class _Manager:
            workspace_dir = Path("/workspace")
            base_dir = workspace_dir / ".fast-agent" / "sessions"

            def resolve_session_name(self, name: str) -> str:
                return name

            def get_session(self, name: str) -> Any:
                del name
                return SimpleNamespace(info=SimpleNamespace(metadata={}))

            def load_session(self, name: str) -> Any:
                return self.get_session(name)

        return _Manager()

    class _Hydrator:
        def hydrate_session(
            self,
            *,
            session: Any,
            agents: Any,
            fallback_agent_name: str | None,
        ) -> SessionHydrationResult:
            del agents, fallback_agent_name
            return _hydration_result(
                session=session,
                session_id="s-1",
                active_agent="main",
                prompts={"main": "Persisted session prompt"},
            )

    monkeypatch.setattr(server, "_initialize_session_state", fake_initialize_session_state)
    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        fake_get_session_manager,
    )
    monkeypatch.setattr("fast_agent.acp.server.session_store.SessionHydrator", _Hydrator)

    response = await server.load_session(
        cwd="/workspace",
        session_id="s-1",
        mcp_servers=[],
    )

    assert response is not None
    assert session_state.resolved_instructions["main"] == "Current runtime prompt"


@pytest.mark.asyncio
async def test_list_sessions_keeps_legacy_sessions_when_cwd_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    server = _build_server(_build_instance(["main"]))
    workspace = tmp_path / "workspace"
    legacy_sessions_dir = workspace / ".fast-agent" / "sessions"
    now = datetime.now()

    legacy_session = SessionInfo(
        name="legacy",
        created_at=now,
        last_activity=now,
        metadata={},
    )
    explicit_session = SessionInfo(
        name="explicit",
        created_at=now,
        last_activity=now,
        metadata={"cwd": str(workspace.resolve())},
    )
    other_session = SessionInfo(
        name="other",
        created_at=now,
        last_activity=now,
        metadata={"cwd": str((tmp_path / "other").resolve())},
    )

    class _Manager:
        base_dir = legacy_sessions_dir

        def list_sessions(self) -> list[SessionInfo]:
            return [legacy_session, explicit_session, other_session]

    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        lambda *, cwd=None, environment_override=None, respect_env_override=True: _Manager(),
    )

    response = await server.list_sessions(cwd=str(workspace))

    assert [session.session_id for session in response.sessions] == ["legacy", "explicit"]
    assert [session.cwd for session in response.sessions] == [
        str(workspace.resolve()),
        str(workspace.resolve()),
    ]


@pytest.mark.asyncio
async def test_list_sessions_uses_manager_workspace_for_legacy_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    server = _build_server(_build_instance(["main"]))
    workspace = tmp_path / "workspace"
    env_root = tmp_path / "custom-env"
    custom_sessions_dir = env_root / "sessions"
    now = datetime.now()

    legacy_session = SessionInfo(
        name="legacy",
        created_at=now,
        last_activity=now,
        metadata={},
    )
    other_session = SessionInfo(
        name="other",
        created_at=now,
        last_activity=now,
        metadata={"cwd": str((tmp_path / "other").resolve())},
    )

    class _Manager:
        workspace_dir = workspace
        base_dir = custom_sessions_dir

        def list_sessions(self) -> list[SessionInfo]:
            return [legacy_session, other_session]

    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        lambda *, cwd=None, environment_override=None, respect_env_override=True: _Manager(),
    )

    response = await server.list_sessions(cwd=str(workspace))

    assert [session.session_id for session in response.sessions] == ["legacy"]
    assert [session.cwd for session in response.sessions] == [str(workspace.resolve())]


@pytest.mark.asyncio
async def test_list_sessions_uses_request_cwd_for_session_manager(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    server = _build_server(_build_instance(["main"]))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    now = datetime.now()
    manager_cwds: list[Any] = []

    class _Manager:
        workspace_dir = workspace
        base_dir = workspace / ".fast-agent" / "sessions"

        def list_sessions(self) -> list[SessionInfo]:
            return [
                SessionInfo(
                    name="workspace-session",
                    created_at=now,
                    last_activity=now,
                    metadata={},
                )
            ]

    def fake_get_session_manager(
        *,
        cwd: Any = None,
        environment_override: Any = None,
        respect_env_override: bool = True,
    ) -> Any:
        manager_cwds.append((cwd, environment_override, respect_env_override))
        return _Manager()

    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        fake_get_session_manager,
    )

    response = await server.list_sessions(cwd=str(workspace))

    assert [session.session_id for session in response.sessions] == ["workspace-session"]
    assert manager_cwds == [
        (workspace.resolve(), None, True),
        (None, None, True),
    ]


@pytest.mark.asyncio
async def test_load_session_prefers_workspace_duplicate_session_across_stores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    primary_instance = _build_instance(["main"])
    server = _build_server(primary_instance)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_state = ACPSessionState(
        session_id="s-1",
        instance=primary_instance,
    )
    hydrate_managers: list[str] = []

    async def fake_initialize_session_state(
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[Any],
    ) -> tuple[ACPSessionState, SessionModeState]:
        del session_id, mcp_servers
        assert cwd == str(workspace.resolve())
        return session_state, SessionModeState(available_modes=[], current_mode_id="main")

    class _Manager:
        def __init__(self, label: str, last_activity: datetime, *, metadata: dict[str, Any]) -> None:
            self.label = label
            self.base_dir = workspace / f"{label}-sessions"
            self.workspace_dir = workspace
            self._session = SimpleNamespace(
                label=label,
                info=SimpleNamespace(
                    metadata=metadata,
                    last_activity=last_activity,
                )
            )

        def resolve_session_name(self, name: str) -> str:
            return name

        def get_session(self, name: str) -> Any:
            assert name == "s-1"
            return self._session

        def load_session(self, name: str) -> Any:
            return self.get_session(name)

    request_manager = _Manager(
        "workspace",
        datetime(2024, 1, 1, 10, 0, 0),
        metadata={"cwd": str(workspace.resolve())},
    )
    app_manager = _Manager(
        "app",
        datetime(2024, 1, 1, 11, 0, 0),
        metadata={},
    )

    def fake_get_session_manager(
        *,
        cwd: Any = None,
        environment_override: Any = None,
        respect_env_override: bool = True,
    ) -> Any:
        del environment_override, respect_env_override
        if cwd is not None:
            return request_manager
        return app_manager

    monkeypatch.setattr(server, "_initialize_session_state", fake_initialize_session_state)
    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        fake_get_session_manager,
    )

    class _Hydrator:
        def hydrate_session(
            self,
            *,
            session: Any,
            agents: Any,
            fallback_agent_name: str | None,
        ) -> SessionHydrationResult:
            del agents, fallback_agent_name
            hydrate_managers.append(session.label)
            return _hydration_result(
                session=session,
                session_id="s-1",
                active_agent="main",
            )

    monkeypatch.setattr("fast_agent.acp.server.session_store.SessionHydrator", _Hydrator)

    response = await server.load_session(
        cwd=str(workspace),
        session_id="s-1",
        mcp_servers=[],
    )

    assert response is not None
    assert hydrate_managers == ["workspace"]


@pytest.mark.asyncio
async def test_load_session_marks_selected_manager_session_current(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary_instance = _build_instance(["main"])
    server = _build_server(primary_instance)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_state = ACPSessionState(
        session_id="s-1",
        instance=primary_instance,
    )
    now = datetime.now()
    candidate_session = SimpleNamespace(
        info=SessionInfo(
            name="s-1",
            created_at=now,
            last_activity=now,
            metadata={"cwd": str(workspace.resolve())},
        ),
        label="candidate",
    )
    loaded_session = SimpleNamespace(info=candidate_session.info, label="loaded")
    hydrate_labels: list[str] = []

    async def fake_initialize_session_state(
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[Any],
    ) -> tuple[ACPSessionState, SessionModeState]:
        del session_id, cwd, mcp_servers
        return session_state, SessionModeState(available_modes=[], current_mode_id="main")

    class _Manager:
        def __init__(self) -> None:
            self.base_dir = workspace / ".fast-agent" / "sessions"
            self.workspace_dir = workspace
            self.current_session = None

        def resolve_session_name(self, name: str) -> str:
            return name

        def get_session(self, name: str) -> Any:
            assert name == "s-1"
            return candidate_session

        def load_session(self, name: str) -> Any:
            assert name == "s-1"
            self.current_session = loaded_session
            return loaded_session

    manager = _Manager()

    def fake_get_session_manager(
        *,
        cwd: Any = None,
        environment_override: Any = None,
        respect_env_override: bool = True,
    ) -> Any:
        del cwd, environment_override, respect_env_override
        return manager

    class _Hydrator:
        def hydrate_session(
            self,
            *,
            session: Any,
            agents: Any,
            fallback_agent_name: str | None,
        ) -> SessionHydrationResult:
            del agents, fallback_agent_name
            hydrate_labels.append(session.label)
            return _hydration_result(
                session=session,
                session_id="s-1",
                active_agent="main",
            )

    monkeypatch.setattr(server, "_initialize_session_state", fake_initialize_session_state)
    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        fake_get_session_manager,
    )
    monkeypatch.setattr("fast_agent.acp.server.session_store.SessionHydrator", _Hydrator)

    response = await server.load_session(
        cwd=str(workspace.resolve()),
        session_id="s-1",
        mcp_servers=[],
    )

    assert response is not None
    assert manager.current_session is loaded_session
    assert hydrate_labels == ["loaded"]


@pytest.mark.asyncio
async def test_list_sessions_prefers_workspace_duplicate_session_across_stores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    server = _build_server(_build_instance(["main"]))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    now = datetime.now()

    workspace_session = SessionInfo(
        name="s-1",
        created_at=now,
        last_activity=datetime(2024, 1, 1, 10, 0, 0),
        metadata={
            "cwd": str(workspace.resolve()),
            "title": "Workspace copy",
        },
    )
    app_session = SessionInfo(
        name="s-1",
        created_at=now,
        last_activity=datetime(2024, 1, 1, 11, 0, 0),
        metadata={
            "cwd": str(workspace.resolve()),
            "title": "App copy",
        },
    )

    class _Manager:
        def __init__(self, label: str, sessions: list[SessionInfo]) -> None:
            self.base_dir = workspace / f"{label}-sessions"
            self.workspace_dir = workspace
            self._sessions = sessions

        def list_sessions(self) -> list[SessionInfo]:
            return self._sessions

    request_manager = _Manager("workspace", [workspace_session])
    app_manager = _Manager("app", [app_session])

    def fake_get_session_manager(
        *,
        cwd: Any = None,
        environment_override: Any = None,
        respect_env_override: bool = True,
    ) -> Any:
        del environment_override, respect_env_override
        if cwd is not None:
            return request_manager
        return app_manager

    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        fake_get_session_manager,
    )

    response = await server.list_sessions(cwd=str(workspace))

    assert len(response.sessions) == 1
    assert response.sessions[0].session_id == "s-1"
    assert response.sessions[0].title == "Workspace copy"
    assert response.sessions[0].updated_at == workspace_session.last_activity.isoformat()


@pytest.mark.asyncio
async def test_load_session_falls_back_to_app_store_when_workspace_store_misses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    primary_instance = _build_instance(["main"])
    server = _build_server(primary_instance)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_state = ACPSessionState(
        session_id="s-1",
        instance=primary_instance,
    )
    hydrate_managers: list[str] = []

    async def fake_initialize_session_state(
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[Any],
    ) -> tuple[ACPSessionState, SessionModeState]:
        del session_id, mcp_servers
        assert cwd == str(workspace.resolve())
        return session_state, SessionModeState(available_modes=[], current_mode_id="main")

    class _WorkspaceManager:
        workspace_dir = workspace
        base_dir = workspace / ".fast-agent" / "sessions"

        def resolve_session_name(self, name: str) -> str:
            return name

        def get_session(self, name: str) -> Any:
            assert name == "s-1"
            return None

    class _AppManager:
        workspace_dir = tmp_path / "server"
        base_dir = workspace_dir / ".fast-agent" / "sessions"

        def resolve_session_name(self, name: str) -> str:
            return name

        def get_session(self, name: str) -> Any:
            assert name == "s-1"
            return SimpleNamespace(label="app", info=SimpleNamespace(metadata={}))

        def load_session(self, name: str) -> Any:
            return self.get_session(name)

    workspace_manager = _WorkspaceManager()
    app_manager = _AppManager()

    def fake_get_session_manager(
        *,
        cwd: Any = None,
        environment_override: Any = None,
        respect_env_override: bool = True,
    ) -> Any:
        del environment_override, respect_env_override
        if cwd is not None:
            return workspace_manager
        return app_manager

    monkeypatch.setattr(server, "_initialize_session_state", fake_initialize_session_state)
    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        fake_get_session_manager,
    )

    class _Hydrator:
        def hydrate_session(
            self,
            *,
            session: Any,
            agents: Any,
            fallback_agent_name: str | None,
        ) -> SessionHydrationResult:
            del agents, fallback_agent_name
            hydrate_managers.append(session.label)
            return _hydration_result(
                session=session,
                session_id="s-1",
                active_agent="main",
            )

    monkeypatch.setattr("fast_agent.acp.server.session_store.SessionHydrator", _Hydrator)

    response = await server.load_session(
        cwd=str(workspace),
        session_id="s-1",
        mcp_servers=[],
    )

    assert response is not None
    assert hydrate_managers == ["app"]
    assert session_state.session_store_scope == "app"
    assert session_state.session_store_cwd is None


@pytest.mark.asyncio
async def test_load_session_rejects_noncanonical_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_instance = _build_instance(["main"])
    server = _build_server(primary_instance)
    requested_ids: list[str] = []

    class _Manager:
        workspace_dir = Path("/workspace")
        base_dir = workspace_dir / ".fast-agent" / "sessions"

        def resolve_session_name(self, name: str) -> str:
            requested_ids.append(name)
            return "s-1"

        def get_session(self, name: str) -> Any:
            assert name == "1"
            return None

    def fake_get_session_manager(
        *,
        cwd: Any = None,
        environment_override: Any = None,
        respect_env_override: bool = True,
    ) -> Any:
        del cwd, environment_override, respect_env_override
        return _Manager()

    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        fake_get_session_manager,
    )

    with pytest.raises(RequestError, match="Session not found: 1"):
        await server.load_session(
            cwd="/workspace",
            session_id="1",
            mcp_servers=[],
        )

    assert requested_ids == []


@pytest.mark.asyncio
async def test_load_session_skips_workspace_duplicate_when_cwd_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    primary_instance = _build_instance(["main"])
    server = _build_server(primary_instance)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    session_state = ACPSessionState(
        session_id="s-1",
        instance=primary_instance,
    )
    hydrate_managers: list[str] = []

    async def fake_initialize_session_state(
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[Any],
    ) -> tuple[ACPSessionState, SessionModeState]:
        del session_id, mcp_servers
        assert cwd == str(workspace.resolve())
        return session_state, SessionModeState(available_modes=[], current_mode_id="main")

    class _WorkspaceManager:
        workspace_dir = workspace
        base_dir = workspace / ".fast-agent" / "sessions"

        def resolve_session_name(self, name: str) -> str:
            return name

        def get_session(self, name: str) -> Any:
            assert name == "s-1"
            return SimpleNamespace(
                info=SimpleNamespace(metadata={"cwd": str(other_workspace.resolve())})
            )

    class _AppManager:
        workspace_dir = tmp_path / "server"
        base_dir = workspace_dir / ".fast-agent" / "sessions"

        def resolve_session_name(self, name: str) -> str:
            return name

        def get_session(self, name: str) -> Any:
            assert name == "s-1"
            return SimpleNamespace(
                label="app",
                info=SimpleNamespace(metadata={"cwd": str(workspace.resolve())})
            )

        def load_session(self, name: str) -> Any:
            return self.get_session(name)

    workspace_manager = _WorkspaceManager()
    app_manager = _AppManager()

    def fake_get_session_manager(
        *,
        cwd: Any = None,
        environment_override: Any = None,
        respect_env_override: bool = True,
    ) -> Any:
        del environment_override, respect_env_override
        if cwd is not None:
            return workspace_manager
        return app_manager

    monkeypatch.setattr(server, "_initialize_session_state", fake_initialize_session_state)
    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        fake_get_session_manager,
    )

    class _Hydrator:
        def hydrate_session(
            self,
            *,
            session: Any,
            agents: Any,
            fallback_agent_name: str | None,
        ) -> SessionHydrationResult:
            del agents, fallback_agent_name
            hydrate_managers.append(session.label)
            return _hydration_result(
                session=session,
                session_id="s-1",
                active_agent="main",
            )

    monkeypatch.setattr("fast_agent.acp.server.session_store.SessionHydrator", _Hydrator)

    response = await server.load_session(
        cwd=str(workspace),
        session_id="s-1",
        mcp_servers=[],
    )

    assert response is not None
    assert hydrate_managers == ["app"]
    assert session_state.session_store_scope == "app"
    assert session_state.session_store_cwd is None


@pytest.mark.asyncio
async def test_list_sessions_rejects_relative_cwd() -> None:
    server = _build_server(_build_instance(["main"]))

    with pytest.raises(RequestError) as exc_info:
        await server.list_sessions(cwd="relative/path")

    assert exc_info.value.code == -32602


@pytest.mark.asyncio
async def test_list_sessions_rejects_invalid_cursor() -> None:
    server = _build_server(_build_instance(["main"]))

    with pytest.raises(RequestError) as exc_info:
        await server.list_sessions(cursor="not-a-valid-cursor")

    assert exc_info.value.code == -32602


@pytest.mark.asyncio
async def test_reload_agent_cards_rehydrates_persisted_session_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary_instance = _build_instance(["main"])
    refreshed_instance = _build_instance(["main", "other"])
    server = _build_server(primary_instance, created_instance=refreshed_instance)
    session_state = ACPSessionState(
        session_id="s-1",
        instance=primary_instance,
        session_store_scope="workspace",
        session_store_cwd=str(tmp_path.resolve()),
    )
    server._session_state["s-1"] = session_state
    server.sessions["s-1"] = primary_instance

    async def fake_replace_instance_for_session(
        session_state_arg: ACPSessionState,
        *,
        dispose_error_name: str,
        await_refresh_session_state: bool,
    ) -> AgentInstance:
        del dispose_error_name, await_refresh_session_state
        session_state_arg.instance = refreshed_instance
        server.sessions[session_state_arg.session_id] = refreshed_instance
        return refreshed_instance

    async def fake_reload_callback() -> bool:
        return True

    class _Manager:
        workspace_dir = tmp_path
        base_dir = tmp_path / ".fast-agent" / "sessions"

        def load_session(self, name: str) -> Any:
            assert name == "s-1"
            return SimpleNamespace(label="loaded", info=SimpleNamespace(metadata={}))

    class _Hydrator:
        def hydrate_session(
            self,
            *,
            session: Any,
            agents: Any,
            fallback_agent_name: str | None,
        ) -> SessionHydrationResult:
            del fallback_agent_name
            assert session.label == "loaded"
            agent = agents["main"]
            assert isinstance(agent, _Agent)
            agent.message_history = ["persisted transcript"]
            agent.set_instruction("Persisted session prompt")
            return _hydration_result(
                session=session,
                session_id="s-1",
                active_agent="main",
                loaded_agents={"main": tmp_path / "history_main.json"},
                restored_prompts={"main": "Persisted session prompt"},
                prompts={"main": "Persisted session prompt"},
            )

    monkeypatch.setattr(server, "_replace_instance_for_session", fake_replace_instance_for_session)
    monkeypatch.setattr(server, "_reload_callback", fake_reload_callback)
    monkeypatch.setattr(server, "_get_session_manager", lambda *, cwd=None: _Manager())
    monkeypatch.setattr("fast_agent.acp.server.session_store.SessionHydrator", _Hydrator)

    changed = await server._reload_agent_cards_for_session("s-1")

    assert changed is True
    assert session_state.instance is refreshed_instance
    assert session_state.current_agent_name == "main"
    assert session_state.resolved_instructions["main"] == "Persisted session prompt"
    refreshed_agent = session_state.instance.agents["main"]
    assert isinstance(refreshed_agent, _Agent)
    assert refreshed_agent.instruction == "Persisted session prompt"
    assert refreshed_agent.message_history == ["persisted transcript"]


@pytest.mark.asyncio
async def test_load_session_rejects_relative_cwd() -> None:
    server = _build_server(_build_instance(["main"]))

    with pytest.raises(RequestError) as exc_info:
        await server.load_session(
            cwd="relative/path",
            session_id="s-1",
            mcp_servers=[],
        )

    assert exc_info.value.code == -32602
