from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from acp.schema import McpServerStdio

from fast_agent.acp.server.agent_acp_server import AgentACPServer
from fast_agent.acp.server.models import ACPSessionState, SessionMCPServerState
from fast_agent.config import MCPServerSettings
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.mcp_aggregator import MCPAttachResult, MCPDetachResult

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.interfaces import AgentProtocol


class _Agent:
    instruction = ""
    acp_commands: dict[str, object] = {}

    def __init__(self, name: str) -> None:
        self.name = name
        self.config = SimpleNamespace(default=False)


def _build_instance(label: str) -> AgentInstance:
    agent = cast("AgentProtocol", _Agent(f"{label}-main"))
    return AgentInstance(
        app=AgentApp({"main": agent}),
        agents={"main": agent},
        registry_version=0,
    )


def _install_runtime_mcp_callbacks(
    instance: AgentInstance,
    *,
    attach=None,
    detach=None,
    list_attached=None,
    list_configured_detached=None,
) -> None:
    if attach is not None:
        instance.app.set_attach_mcp_server_callback(attach)
    if detach is not None:
        instance.app.set_detach_mcp_server_callback(detach)
    if list_attached is not None:
        instance.app.set_list_attached_mcp_servers_callback(list_attached)
    if list_configured_detached is not None:
        instance.app.set_list_configured_detached_mcp_servers_callback(list_configured_detached)


def _build_session_server(
    primary_instance: AgentInstance,
    created_instances: list[AgentInstance],
) -> AgentACPServer:
    async def create_instance() -> AgentInstance:
        assert created_instances, "Test did not supply enough instances"
        return created_instances.pop(0)

    async def dispose_instance(_instance: AgentInstance) -> None:
        return None

    return AgentACPServer(
        bootstrap_instance=primary_instance,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        instance_scope="connection",
        server_name="test",
        permissions_enabled=False,
    )


@pytest.mark.asyncio
async def test_initialize_session_state_applies_session_mcp_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary_instance = _build_instance("primary")
    session_instance = _build_instance("session")
    server = _build_session_server(primary_instance, [session_instance])

    attach_calls: list[tuple[AgentInstance, list[str]]] = []

    async def fake_apply_session_mcp_overlay(
        session_state: ACPSessionState,
        instance: AgentInstance,
        *,
        force_reconnect_targets: set[tuple[str, str]] | None = None,
    ) -> None:
        del force_reconnect_targets
        attach_calls.append((instance, sorted(session_state.session_mcp_servers)))

    monkeypatch.setattr(
        server._session_runtime,
        "_apply_session_mcp_overlay",
        fake_apply_session_mcp_overlay,
    )

    session_state, _ = await server._initialize_session_state(
        "session-1",
        cwd=str(tmp_path),
        mcp_servers=[
            McpServerStdio(
                name="session-tools",
                command="/bin/echo",
                args=["hello"],
                env=[],
            )
        ],
    )

    assert session_state.instance is session_instance
    assert server.sessions["session-1"] is session_instance
    assert list(session_state.session_mcp_servers) == ["session-tools"]
    assert session_state.session_mcp_servers["session-tools"].server_config is not None
    assert attach_calls == [(session_instance, ["session-tools"])]


@pytest.mark.asyncio
async def test_replace_instance_for_session_reapplies_session_mcp_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary_instance = _build_instance("primary")
    first_session_instance = _build_instance("session-initial")
    refreshed_session_instance = _build_instance("session-refreshed")
    server = _build_session_server(
        primary_instance,
        [first_session_instance, refreshed_session_instance],
    )

    attach_calls: list[tuple[AgentInstance, list[str]]] = []

    async def fake_apply_session_mcp_overlay(
        session_state: ACPSessionState,
        instance: AgentInstance,
        *,
        force_reconnect_targets: set[tuple[str, str]] | None = None,
    ) -> None:
        del force_reconnect_targets
        attach_calls.append((instance, sorted(session_state.session_mcp_servers)))

    monkeypatch.setattr(
        server._session_runtime,
        "_apply_session_mcp_overlay",
        fake_apply_session_mcp_overlay,
    )

    session_state, _ = await server._initialize_session_state(
        "session-1",
        cwd=str(tmp_path),
        mcp_servers=[
            McpServerStdio(
                name="session-tools",
                command="/bin/echo",
                args=["hello"],
                env=[],
            )
        ],
    )

    assert session_state.instance is first_session_instance

    replaced = await server._replace_instance_for_session(
        session_state,
        dispose_error_name="test_dispose_error",
        await_refresh_session_state=True,
    )

    assert replaced is refreshed_session_instance
    assert session_state.instance is refreshed_session_instance
    assert server.sessions["session-1"] is refreshed_session_instance
    assert attach_calls == [
        (first_session_instance, ["session-tools"]),
        (refreshed_session_instance, ["session-tools"]),
    ]


@pytest.mark.asyncio
async def test_initialize_session_state_detaches_removed_session_mcp_servers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary_instance = _build_instance("primary")
    session_instance = _build_instance("session")
    server = _build_session_server(primary_instance, [session_instance])

    apply_calls: list[tuple[AgentInstance, list[str]]] = []
    detach_calls: list[tuple[AgentInstance, str, str]] = []

    async def fake_apply_session_mcp_overlay(
        session_state: ACPSessionState,
        instance: AgentInstance,
        *,
        force_reconnect_targets: set[tuple[str, str]] | None = None,
    ) -> None:
        del force_reconnect_targets
        apply_calls.append((instance, sorted(session_state.session_mcp_servers)))

    async def fake_detach_server_from_agent(
        instance: AgentInstance,
        *,
        agent_name: str,
        server_name: str,
    ) -> MCPDetachResult:
        detach_calls.append((instance, agent_name, server_name))
        return MCPDetachResult(
            server_name=server_name,
            detached=True,
            tools_removed=[f"{server_name}.echo"],
            prompts_removed=[],
        )

    monkeypatch.setattr(
        server._session_runtime,
        "_apply_session_mcp_overlay",
        fake_apply_session_mcp_overlay,
    )
    monkeypatch.setattr(
        server._session_runtime,
        "_detach_server_from_agent",
        fake_detach_server_from_agent,
    )
    monkeypatch.setattr(
        server._session_runtime,
        "_mcp_capable_agents",
        lambda instance: [("main", cast("AgentProtocol", instance.agents["main"]))],
    )

    session_state, _ = await server._initialize_session_state(
        "session-1",
        cwd=str(tmp_path),
        mcp_servers=[
            McpServerStdio(
                name="keep-tools",
                command="/bin/echo",
                args=["keep"],
                env=[],
            ),
            McpServerStdio(
                name="remove-tools",
                command="/bin/echo",
                args=["remove"],
                env=[],
            ),
        ],
    )

    session_state.agent_mcp_servers = {
        "main": {
            "agent-only": SessionMCPServerState(
                server_name="agent-only",
                server_config=MCPServerSettings(
                    name="agent-only",
                    transport="stdio",
                    command="echo",
                ),
                attached=True,
            )
        }
    }

    session_state, _ = await server._initialize_session_state(
        "session-1",
        cwd=str(tmp_path),
        mcp_servers=[
            McpServerStdio(
                name="keep-tools",
                command="/bin/echo",
                args=["keep"],
                env=[],
            )
        ],
    )

    assert session_state.session_mcp_servers.keys() == {"keep-tools"}
    assert session_state.agent_mcp_servers == {
        "main": {
            "agent-only": SessionMCPServerState(
                server_name="agent-only",
                server_config=MCPServerSettings(
                    name="agent-only",
                    transport="stdio",
                    command="echo",
                ),
                attached=True,
            )
        }
    }
    assert detach_calls == [
        (session_instance, "main", "remove-tools"),
    ]
    assert apply_calls == [
        (session_instance, ["keep-tools", "remove-tools"]),
        (session_instance, ["keep-tools"]),
    ]


@pytest.mark.asyncio
async def test_initialize_session_state_reconnects_same_name_when_config_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary_instance = _build_instance("primary")
    session_instance = _build_instance("session")
    server = _build_session_server(primary_instance, [session_instance])

    apply_calls: list[tuple[AgentInstance, set[tuple[str, str]]]] = []

    async def fake_apply_session_mcp_overlay(
        session_state: ACPSessionState,
        instance: AgentInstance,
        *,
        force_reconnect_targets: set[tuple[str, str]] | None = None,
    ) -> None:
        del session_state
        apply_calls.append((instance, force_reconnect_targets or set()))

    monkeypatch.setattr(
        server._session_runtime,
        "_apply_session_mcp_overlay",
        fake_apply_session_mcp_overlay,
    )
    monkeypatch.setattr(
        server._session_runtime,
        "_mcp_capable_agents",
        lambda instance: [("main", cast("AgentProtocol", instance.agents["main"]))],
    )

    await server._initialize_session_state(
        "session-1",
        cwd=str(tmp_path),
        mcp_servers=[
            McpServerStdio(
                name="session-tools",
                command="/bin/echo",
                args=["hello"],
                env=[],
            )
        ],
    )

    await server._initialize_session_state(
        "session-1",
        cwd=str(tmp_path),
        mcp_servers=[
            McpServerStdio(
                name="session-tools",
                command="/bin/echo",
                args=["updated"],
                env=[],
            )
        ],
    )

    assert apply_calls == [
        (session_instance, set()),
        (session_instance, {("main", "session-tools")}),
    ]


@pytest.mark.asyncio
async def test_attach_session_mcp_server_uses_existing_session_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_instance = _build_instance("primary")
    session_instance = _build_instance("session")
    server = _build_session_server(primary_instance, [])
    session_state = ACPSessionState(session_id="session-1", instance=session_instance)
    server.sessions["session-1"] = session_instance

    async def fake_attach_server_to_agent(
        instance: AgentInstance,
        *,
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None,
        options=None,
    ):
        assert instance is session_instance
        assert agent_name == "main"
        assert server_name == "demo"
        assert server_config is not None
        assert options is None
        return MCPAttachResult(
            server_name="demo",
            transport="stdio",
            attached=True,
            already_attached=False,
            tools_added=["demo.echo"],
            prompts_added=[],
            warnings=[],
        )

    monkeypatch.setattr(
        server._session_runtime,
        "_attach_server_to_agent",
        fake_attach_server_to_agent,
    )

    result = await server._attach_mcp_server_for_session(
        session_state,
        agent_name="main",
        server_name="demo",
        server_config=MCPServerSettings(name="demo", transport="stdio", command="echo"),
    )

    assert result.server_name == "demo"
    assert session_state.instance is session_instance
    assert session_state.agent_mcp_servers["main"]["demo"].attached is True
    assert session_state.agent_mcp_servers["main"]["demo"].server_config is not None


@pytest.mark.asyncio
async def test_attach_session_mcp_server_persists_inherited_session_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_instance = _build_instance("primary")
    session_instance = _build_instance("session")
    server = _build_session_server(primary_instance, [])
    session_state = ACPSessionState(session_id="session-1", instance=session_instance)
    session_state.session_mcp_servers = {
        "demo": SessionMCPServerState(
            server_name="demo",
            server_config=MCPServerSettings(
                name="demo",
                transport="stdio",
                command="echo",
                args=["session"],
            ),
            attached=True,
        )
    }
    server.sessions["session-1"] = session_instance

    attached_configs: list[MCPServerSettings | None] = []

    async def fake_attach_server_to_agent(
        instance: AgentInstance,
        *,
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None,
        options=None,
    ):
        assert instance is session_instance
        assert agent_name == "main"
        assert server_name == "demo"
        assert options is None
        attached_configs.append(server_config)
        return MCPAttachResult(
            server_name="demo",
            transport="stdio",
            attached=True,
            already_attached=False,
            tools_added=["demo.echo"],
            prompts_added=[],
            warnings=[],
        )

    monkeypatch.setattr(
        server._session_runtime,
        "_attach_server_to_agent",
        fake_attach_server_to_agent,
    )

    result = await server._attach_mcp_server_for_session(
        session_state,
        agent_name="main",
        server_name="demo",
        server_config=None,
    )

    assert result.server_name == "demo"
    assert session_state.instance is session_instance
    assert session_state.agent_mcp_servers["main"]["demo"].attached is True
    assert session_state.agent_mcp_servers["main"]["demo"].server_config is not None
    assert attached_configs == [session_state.session_mcp_servers["demo"].server_config]


@pytest.mark.asyncio
async def test_attach_session_mcp_server_force_reconnects_same_name_config_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_instance = _build_instance("primary")
    session_instance = _build_instance("session")
    server = _build_session_server(primary_instance, [])
    session_state = ACPSessionState(session_id="session-1", instance=session_instance)
    session_state.agent_mcp_servers = {
        "main": {
            "demo": SessionMCPServerState(
                server_name="demo",
                server_config=MCPServerSettings(
                    name="demo",
                    transport="stdio",
                    command="echo",
                    args=["old"],
                ),
                attached=True,
            )
        }
    }
    server.sessions["session-1"] = session_instance

    attach_options: list[Any] = []

    async def fake_attach_server_to_agent(
        instance: AgentInstance,
        *,
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None,
        options=None,
    ):
        assert instance is session_instance
        assert agent_name == "main"
        assert server_name == "demo"
        assert server_config is not None
        attach_options.append(options)
        return MCPAttachResult(
            server_name="demo",
            transport="stdio",
            attached=True,
            already_attached=True,
            tools_added=[],
            prompts_added=[],
            warnings=[],
        )

    monkeypatch.setattr(
        server._session_runtime,
        "_attach_server_to_agent",
        fake_attach_server_to_agent,
    )

    await server._attach_mcp_server_for_session(
        session_state,
        agent_name="main",
        server_name="demo",
        server_config=MCPServerSettings(
            name="demo",
            transport="stdio",
            command="echo",
            args=["new"],
        ),
    )

    assert len(attach_options) == 1
    assert attach_options[0] is not None
    assert attach_options[0].force_reconnect is True


@pytest.mark.asyncio
async def test_attach_session_mcp_server_uses_runtime_manager_error_for_non_mcp_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_agent = cast("AgentProtocol", _Agent("default"))
    helper_agent = cast("AgentProtocol", _Agent("helper"))
    session_instance = AgentInstance(
        app=AgentApp({"default": default_agent, "helper": helper_agent}),
        agents={"default": default_agent, "helper": helper_agent},
        registry_version=0,
    )
    primary_instance = _build_instance("primary")
    server = _build_session_server(primary_instance, [])
    session_state = ACPSessionState(session_id="session-1", instance=session_instance)
    server.sessions["session-1"] = session_instance

    async def attach_mcp_server(
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options=None,
    ) -> MCPAttachResult:
        del server_name, server_config, options
        raise RuntimeError(f"Agent '{agent_name}' does not support MCP server management")

    _install_runtime_mcp_callbacks(session_instance, attach=attach_mcp_server)
    monkeypatch.setattr(
        server._session_runtime,
        "_mcp_capable_agents",
        lambda instance: [("helper", cast("Any", instance.agents["helper"]))],
    )

    with pytest.raises(RuntimeError, match="Agent 'default' does not support MCP server management"):
        await server._attach_mcp_server_for_session(
            session_state,
            agent_name="default",
            server_name="demo",
            server_config=MCPServerSettings(name="demo", transport="stdio", command="echo"),
        )

    assert session_state.agent_mcp_servers == {}


@pytest.mark.asyncio
async def test_detach_session_mcp_server_records_agent_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_instance = _build_instance("primary")
    session_instance = _build_instance("session")
    server = _build_session_server(primary_instance, [])
    session_state = ACPSessionState(session_id="session-1", instance=session_instance)
    existing_config = MCPServerSettings(name="local", transport="stdio", command="echo")
    session_state.agent_mcp_servers = {
        "main": {
            "local": SessionMCPServerState(
                server_name="local",
                server_config=existing_config,
                attached=True,
            )
        }
    }
    server.sessions["session-1"] = session_instance

    async def fake_detach_server_from_agent(
        instance: AgentInstance,
        *,
        agent_name: str,
        server_name: str,
    ) -> MCPDetachResult:
        assert instance is session_instance
        assert agent_name == "main"
        assert server_name == "local"
        return MCPDetachResult(
            server_name=server_name,
            detached=True,
            tools_removed=["local.echo"],
            prompts_removed=[],
        )

    monkeypatch.setattr(
        server._session_runtime,
        "_detach_server_from_agent",
        fake_detach_server_from_agent,
    )

    result = await server._detach_mcp_server_for_session(
        session_state,
        agent_name="main",
        server_name="local",
    )

    assert result.detached is True
    assert session_state.instance is session_instance
    assert session_state.agent_mcp_servers["main"]["local"].attached is False
    assert session_state.agent_mcp_servers["main"]["local"].server_config == existing_config


@pytest.mark.asyncio
async def test_attach_session_mcp_server_uses_detached_overlay_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_instance = _build_instance("primary")
    session_instance = _build_instance("session")
    server = _build_session_server(primary_instance, [])
    existing_config = MCPServerSettings(
        name="local",
        transport="stdio",
        command="echo",
        args=["hello"],
    )
    session_state = ACPSessionState(
        session_id="session-1",
        instance=session_instance,
        agent_mcp_servers={
            "main": {
                "local": SessionMCPServerState(
                    server_name="local",
                    server_config=existing_config,
                    attached=False,
                )
            }
        },
    )
    server.sessions["session-1"] = session_instance

    attached_configs: list[MCPServerSettings | None] = []

    async def fake_attach_server_to_agent(
        instance: AgentInstance,
        *,
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None,
        options=None,
    ) -> MCPAttachResult:
        assert instance is session_instance
        assert agent_name == "main"
        assert server_name == "local"
        assert options is None
        attached_configs.append(server_config)
        return MCPAttachResult(
            server_name=server_name,
            transport="stdio",
            attached=True,
            already_attached=False,
            tools_added=["local.echo"],
            prompts_added=[],
            warnings=[],
        )

    monkeypatch.setattr(
        server._session_runtime,
        "_attach_server_to_agent",
        fake_attach_server_to_agent,
    )

    result = await server._attach_mcp_server_for_session(
        session_state,
        agent_name="main",
        server_name="local",
        server_config=None,
    )

    assert result.attached is True
    assert attached_configs == [existing_config]
    assert session_state.agent_mcp_servers["main"]["local"].attached is True
    assert session_state.agent_mcp_servers["main"]["local"].server_config == existing_config


@pytest.mark.asyncio
async def test_list_configured_detached_mcp_servers_includes_session_overlay_config() -> None:
    primary_instance = _build_instance("primary")
    session_instance = _build_instance("session")
    server = _build_session_server(primary_instance, [])
    session_state = ACPSessionState(
        session_id="session-1",
        instance=session_instance,
        agent_mcp_servers={
            "main": {
                "local": SessionMCPServerState(
                    server_name="local",
                    server_config=MCPServerSettings(
                        name="local",
                        transport="stdio",
                        command="echo",
                    ),
                    attached=False,
                )
            }
        },
    )
    server.sessions["session-1"] = session_instance

    async def fake_list_configured_detached(_agent_name: str) -> list[str]:
        return []

    _install_runtime_mcp_callbacks(
        session_instance,
        list_configured_detached=fake_list_configured_detached,
    )

    detached = await server._list_configured_detached_mcp_servers_for_session(
        session_state,
        agent_name="main",
    )

    assert detached == ["local"]
