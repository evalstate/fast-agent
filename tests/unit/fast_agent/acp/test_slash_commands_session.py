from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.acp.slash.handlers import session as session_slash_handlers
from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.core.fastagent import AgentInstance

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    acp_commands: dict[str, object] = {}


class _App:
    def _agent(self, _name: str):
        return _Agent()

    def visible_agent_names(self, *, force_include: str | None = None):
        del force_include
        return ["main"]

    def registered_agent_names(self):
        return ["main"]

    def registered_agents(self):
        return {"main": _Agent()}

    def resolve_target_agent_name(self, agent_name: str | None = None):
        return agent_name or "main"

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):
        del namespace, agent_name
        return {}


@pytest.mark.asyncio
async def test_render_session_list_uses_acp_session_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager_calls: list[Path | None] = []

    class _Manager:
        current_session = None

        def list_sessions(self) -> list[object]:
            return []

    def fake_get_session_manager(
        *,
        cwd: Path | None = None,
        environment_override=None,
        respect_env_override: bool = True,
    ) -> object:
        del environment_override, respect_env_override
        manager_calls.append(cwd)
        return _Manager()

    monkeypatch.setattr("fast_agent.session.get_session_manager", fake_get_session_manager)

    handler = SlashCommandHandler(
        session_id="s1",
        instance=instance,
        primary_agent_name="main",
    )
    handler._acp_context = cast(
        "Any",
        SimpleNamespace(
            session_cwd=str(workspace.resolve()),
            session_store_scope="workspace",
            session_store_cwd=None,
        ),
    )

    output = session_slash_handlers.render_session_list(handler)

    assert "# sessions" in output
    assert manager_calls == [workspace.resolve()]


@pytest.mark.asyncio
async def test_render_session_list_uses_app_session_store_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager_calls: list[Path | None] = []

    class _Manager:
        current_session = None

        def list_sessions(self) -> list[object]:
            return []

    def fake_get_session_manager(
        *,
        cwd: Path | None = None,
        environment_override=None,
        respect_env_override: bool = True,
    ) -> object:
        del environment_override, respect_env_override
        manager_calls.append(cwd)
        return _Manager()

    monkeypatch.setattr("fast_agent.session.get_session_manager", fake_get_session_manager)

    handler = SlashCommandHandler(
        session_id="s1",
        instance=instance,
        primary_agent_name="main",
    )
    handler._acp_context = cast(
        "Any",
        SimpleNamespace(
            session_cwd=str(workspace.resolve()),
            session_store_scope="app",
            session_store_cwd=None,
        ),
    )

    output = session_slash_handlers.render_session_list(handler)

    assert "# sessions" in output
    assert manager_calls == [None]
