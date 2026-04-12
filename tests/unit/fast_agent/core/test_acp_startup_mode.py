from __future__ import annotations

import argparse
import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import (
    AgentInstance,
    FastAgent,
    ManagedRunState,
    RunRuntime,
    RunSettings,
)
from fast_agent.mcp.mcp_aggregator import MCPAttachResult, MCPDetachResult

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.config = SimpleNamespace(default=True)


def test_is_acp_server_mode_requires_server_flag_and_acp_transport() -> None:
    agent = FastAgent("TestAgent", parse_cli_args=False)

    agent.args = argparse.Namespace(server=True, transport="acp")
    assert agent._is_acp_server_mode() is True

    agent.args = argparse.Namespace(server=False, transport="acp")
    assert agent._is_acp_server_mode() is False

    agent.args = argparse.Namespace(server=True, transport="stdio")
    assert agent._is_acp_server_mode() is False


def test_resolve_server_instance_scope_defaults_acp_to_connection() -> None:
    assert (
        FastAgent._resolve_server_instance_scope(
            transport="acp",
            instance_scope=None,
        )
        == "connection"
    )


def test_resolve_server_instance_scope_rejects_explicit_shared_for_acp() -> None:
    with pytest.raises(ValueError, match="ACP is always connection-scoped"):
        FastAgent._resolve_server_instance_scope(
            transport="acp",
            instance_scope="shared",
        )


def test_resolve_server_instance_scope_rejects_explicit_request_for_acp() -> None:
    with pytest.raises(ValueError, match="ACP is always connection-scoped"):
        FastAgent._resolve_server_instance_scope(
            transport="acp",
            instance_scope="request",
        )


@pytest.mark.asyncio
async def test_runtime_callback_instances_inherit_mcp_runtime_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = FastAgent("TestAgent", parse_cli_args=False)
    agent = cast("AgentProtocol", _Agent("main"))
    wrapper = AgentApp({"main": agent})
    state = ManagedRunState(
        runtime=RunRuntime(
            model_factory_func=lambda model=None, request_params=None: None,
            global_prompt_context=None,
            is_acp_server_mode=True,
            noenv_mode=False,
            managed_instances=[],
            instance_lock=asyncio.Lock(),
        ),
        primary_instance=AgentInstance(wrapper, {"main": agent}),
        wrapper=wrapper,
        active_agents={"main": agent},
    )
    settings = RunSettings(
        quiet_mode=True,
        cli_model_override=None,
        noenv_mode=False,
        server_mode=True,
        transport="acp",
        is_acp_server_mode=True,
        reload_enabled=False,
    )

    created_instance = AgentInstance(AgentApp({"main": agent}), {"main": agent})

    async def fake_instantiate(runtime: RunRuntime) -> AgentInstance:
        del runtime
        return created_instance

    attach_calls: list[tuple[str, str]] = []

    async def fake_attach(
        active_agents: dict[str, object],
        agent_name: str,
        server_name: str,
        server_config=None,
        options=None,
    ) -> MCPAttachResult:
        del active_agents, server_config, options
        attach_calls.append((agent_name, server_name))
        return MCPAttachResult(
            server_name=server_name,
            transport="stdio",
            attached=True,
            already_attached=False,
            tools_added=[],
            prompts_added=[],
            warnings=[],
        )

    async def fake_detach(
        active_agents: dict[str, object],
        agent_name: str,
        server_name: str,
    ) -> MCPDetachResult:
        del active_agents, agent_name
        return MCPDetachResult(
            server_name=server_name,
            detached=True,
            tools_removed=[],
            prompts_removed=[],
        )

    async def fake_list_attached(active_agents: dict[str, object], agent_name: str) -> list[str]:
        del active_agents, agent_name
        return ["demo"]

    async def fake_list_configured(active_agents: dict[str, object], agent_name: str) -> list[str]:
        del active_agents, agent_name
        return ["docs"]

    monkeypatch.setattr(fast, "_instantiate_agent_instance", fake_instantiate)
    monkeypatch.setattr(fast, "_attach_mcp_server_and_refresh", fake_attach)
    monkeypatch.setattr(fast, "_detach_mcp_server_and_refresh", fake_detach)
    monkeypatch.setattr(fast, "_list_attached_mcp_servers", fake_list_attached)
    monkeypatch.setattr(
        fast,
        "_list_configured_detached_mcp_servers",
        fake_list_configured,
    )

    callbacks = fast._build_runtime_callbacks(state, settings)
    instance = await callbacks.create_instance()

    assert instance.app.can_attach_mcp_servers() is True
    assert instance.app.can_detach_mcp_servers() is True
    assert await instance.app.list_attached_mcp_servers("main") == ["demo"]
    assert await instance.app.list_configured_detached_mcp_servers("main") == ["docs"]
    attach_result = await instance.app.attach_mcp_server("main", "runtime-demo")

    assert attach_result.server_name == "runtime-demo"
    assert attach_calls == [("main", "runtime-demo")]


@pytest.mark.asyncio
async def test_runtime_mcp_callbacks_bind_to_instance_agents_not_primary_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = FastAgent("TestAgent", parse_cli_args=False)
    primary_agent = cast("AgentProtocol", _Agent("primary"))
    session_agent = cast("AgentProtocol", _Agent("session"))
    wrapper = AgentApp({"main": primary_agent})
    state = ManagedRunState(
        runtime=RunRuntime(
            model_factory_func=lambda model=None, request_params=None: None,
            global_prompt_context=None,
            is_acp_server_mode=True,
            noenv_mode=False,
            managed_instances=[],
            instance_lock=asyncio.Lock(),
        ),
        primary_instance=AgentInstance(wrapper, {"main": primary_agent}),
        wrapper=wrapper,
        active_agents={"main": primary_agent},
    )
    settings = RunSettings(
        quiet_mode=True,
        cli_model_override=None,
        noenv_mode=False,
        server_mode=True,
        transport="acp",
        is_acp_server_mode=True,
        reload_enabled=False,
    )

    created_instance = AgentInstance(AgentApp({"main": session_agent}), {"main": session_agent})

    async def fake_instantiate(runtime: RunRuntime) -> AgentInstance:
        del runtime
        return created_instance

    async def fake_attach(
        active_agents: dict[str, object],
        agent_name: str,
        server_name: str,
        server_config=None,
        options=None,
    ) -> MCPAttachResult:
        del server_config, options
        assert active_agents is not state.active_agents
        assert active_agents["main"] is session_agent
        assert active_agents["main"] is not primary_agent
        assert agent_name == "main"
        return MCPAttachResult(
            server_name=server_name,
            transport="stdio",
            attached=True,
            already_attached=False,
            tools_added=[],
            prompts_added=[],
            warnings=[],
        )

    monkeypatch.setattr(fast, "_instantiate_agent_instance", fake_instantiate)
    monkeypatch.setattr(fast, "_attach_mcp_server_and_refresh", fake_attach)

    callbacks = fast._build_runtime_callbacks(state, settings)
    instance = await callbacks.create_instance()
    result = await instance.app.attach_mcp_server("main", "runtime-demo")

    assert result.server_name == "runtime-demo"
