from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from acp.schema import SessionModeState

from fast_agent.acp.server.agent_acp_server import ACPSessionState, AgentACPServer
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    instruction = ""
    acp_commands: dict[str, object] = {}

    def __init__(self) -> None:
        self.config = SimpleNamespace(default=False)


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


def _build_server(instance: AgentInstance) -> AgentACPServer:
    async def create_instance() -> AgentInstance:
        return instance

    async def dispose_instance(_instance: AgentInstance) -> None:
        return None

    return AgentACPServer(
        primary_instance=instance,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        instance_scope="shared",
        server_name="test",
        permissions_enabled=False,
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

    resume_calls: list[str | None] = []

    def fake_get_session_manager(*, cwd: Any) -> Any:
        del cwd

        class _Manager:
            def resume_session_agents(
                self,
                agents: Any,
                name: str | None = None,
                fallback_agent_name: str | None = None,
            ) -> Any:
                del agents, name
                resume_calls.append(fallback_agent_name)
                return SimpleNamespace(
                    session=SimpleNamespace(),
                    loaded={},
                    missing_agents=["main"],
                    usage_notices=[],
                )

        return _Manager()

    monkeypatch.setattr(server, "_initialize_session_state", fake_initialize_session_state)
    monkeypatch.setattr(
        "fast_agent.acp.server.agent_acp_server.get_session_manager",
        fake_get_session_manager,
    )

    response = await server.load_session(
        cwd=".",
        session_id="s-1",
        mcp_servers=[],
    )

    assert response is not None
    assert response.modes is not None
    assert resume_calls == ["renamed"]
    assert session_state.current_agent_name == "renamed"
    assert response.modes.current_mode_id == "renamed"
