from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.core.agent_app import AgentApp, AgentRefreshResult
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.fastagent import (
    AgentInstance,
    FastAgent,
    ManagedRunState,
    RunRuntime,
    RunSettings,
)
from fast_agent.core.server_runtime import ServerRuntimeContext, run_mcp_server
from fast_agent.mcp.mcp_aggregator import MCPAttachResult, MCPDetachResult
from fast_agent.session.session_manager import ResumeSessionAgentsResult, Session, SessionInfo
from fast_agent.tools.session_environment import ShellExecutionResult

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from fast_agent.core.fastagent import RuntimeCallbacks
    from fast_agent.interfaces import AgentProtocol, FastAgentLLMProtocol, LLMFactoryProtocol
    from fast_agent.mcp.server.harness_app_server import HarnessMCPAppRuntimeOptions


class _Agent:
    def __init__(self, name: str, *, default: bool = True, instruction: str = "") -> None:
        self.name = name
        self.config = SimpleNamespace(default=default)
        self.instruction = instruction

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction

    async def shutdown(self) -> None:
        pass


def _unused_llm_factory(agent: AgentProtocol, **kwargs: object) -> FastAgentLLMProtocol:
    del agent, kwargs
    raise AssertionError("LLM factory should not be called by this test")


def _unused_model_factory(model: str | None = None) -> LLMFactoryProtocol:
    del model
    return _unused_llm_factory


class _FakeShellExecutor:
    async def execute_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        del command, cwd, env, timeout
        return ShellExecutionResult(stdout="", stderr="", exit_code=0)


def _run_runtime(
    *,
    global_prompt_context: dict[str, str] | None = None,
    is_acp_server_mode: bool = True,
    resume_requested: bool = False,
    resume_session_id: str | None = None,
    target_agent_name: str | None = None,
) -> RunRuntime:
    return RunRuntime(
        model_factory_func=_unused_model_factory,
        global_prompt_context=global_prompt_context,
        is_acp_server_mode=is_acp_server_mode,
        noenv_mode=False,
        managed_instances=[],
        instance_lock=asyncio.Lock(),
        shell_executor=_FakeShellExecutor(),
        resume_requested=resume_requested,
        resume_session_id=resume_session_id,
        target_agent_name=target_agent_name,
    )


def _resume_result(tmp_path: Path, *, active_agent: str | None = None) -> ResumeSessionAgentsResult:
    now = datetime.now()
    session = Session(
        SessionInfo(
            name="session-1",
            created_at=now,
            last_activity=now,
        ),
        tmp_path,
    )
    return ResumeSessionAgentsResult(
        session=session,
        loaded={},
        missing_agents=[],
        active_agent=active_agent,
    )


@pytest.mark.asyncio
async def test_main_returns_false_when_args_lacks_server_flag() -> None:
    agent = FastAgent("TestAgent", parse_cli_args=False)
    agent.args = argparse.Namespace(transport="stdio")

    assert await agent.main() is False


@pytest.mark.parametrize(
    ("argv", "expected_server"),
    [
        (["agent.py", "--transport", "http"], True),
        (["agent.py"], False),
    ],
)
def test_constructor_cli_server_mode_is_transport_driven(
    argv: list[str],
    expected_server: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    agent = FastAgent("TestAgent", parse_cli_args=False)
    parser = FastAgent._constructor_arg_parser()
    agent.args, _unknown = parser.parse_known_args()

    agent._normalize_constructor_cli_server_flags()

    assert agent.args.server is expected_server


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
async def test_run_mcp_server_forwards_instance_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_options: HarnessMCPAppRuntimeOptions | None = None

    async def fake_run_harness_mcp_app_server(**kwargs: object) -> None:
        nonlocal captured_options
        captured_options = cast("HarnessMCPAppRuntimeOptions", kwargs["options"])

    monkeypatch.setattr(
        "fast_agent.mcp.server.harness_app_server.run_harness_mcp_app_server",
        fake_run_harness_mcp_app_server,
    )

    context = ServerRuntimeContext(
        app_name="TestAgent",
        args=argparse.Namespace(
            agent=None,
            host="127.0.0.1",
            instance_scope="request",
            port=8000,
            server_description=None,
            server_name=None,
            tool_description=None,
            transport="http",
        ),
        config=None,
        skills_directory_override=None,
        state=cast("ManagedRunState", SimpleNamespace(runtime=SimpleNamespace(shell_executor=None))),
        callbacks=cast(
            "RuntimeCallbacks",
            SimpleNamespace(instance_factory=lambda: object()),
        ),
        settings=cast("RunSettings", SimpleNamespace()),
        acp_server_factory=lambda: object,
    )

    await run_mcp_server(context)

    assert captured_options is not None
    assert captured_options.instance_scope == "request"


def test_resume_request_uses_app_default_for_missing_cli_placeholder() -> None:
    fast = FastAgent("TestAgent", parse_cli_args=False)
    agent = cast("AgentProtocol", _Agent("card_default"))
    app = AgentApp({"card_default": agent})

    request = fast._session_restore_request(
        app,
        _run_runtime(
            resume_requested=True,
            resume_session_id="session-1",
            target_agent_name="agent",
        ),
    )

    assert request.fallback_agent_name == "card_default"


@pytest.mark.asyncio
async def test_finalize_resume_preserves_restored_prompt_after_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fast = FastAgent("TestAgent", parse_cli_args=False)
    agent = cast("AgentProtocol", _Agent("main", instruction="Template {{workspaceRoot}}"))
    app = AgentApp({"main": agent})
    instance = AgentInstance(app, {"main": agent})
    runtime = _run_runtime(
        global_prompt_context={"workspaceRoot": "/current"},
        is_acp_server_mode=False,
        resume_requested=True,
        resume_session_id="session-1",
        target_agent_name=None,
    )

    async def fake_apply_instruction_context(
        target_instance: AgentInstance,
        context: dict[str, str],
    ) -> None:
        assert target_instance is instance
        assert context == {"workspaceRoot": "/current"}
        agent.set_instruction("Template /current")

    async def fake_restore_requested_session(
        agents: dict[str, AgentProtocol],
        request,
    ) -> ResumeSessionAgentsResult:
        assert agents["main"].instruction == "Template /current"
        assert request.fallback_agent_name == "main"
        agents["main"].set_instruction("Stored session prompt")
        return _resume_result(tmp_path, active_agent="main")

    monkeypatch.setattr(fast, "_apply_instruction_context", fake_apply_instruction_context)
    monkeypatch.setattr(
        "fast_agent.core.managed_runtime.restore_requested_session",
        fake_restore_requested_session,
    )
    monkeypatch.setattr(
        "fast_agent.core.managed_runtime.validate_final_provider_state",
        lambda agents: None,
    )

    result = await fast._finalize_initial_agent_instance(runtime, instance)

    assert result.active_agent == "main"
    assert agent.instruction == "Stored session prompt"


@pytest.mark.asyncio
async def test_refresh_finalizes_full_rebuild_before_session_hydration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = FastAgent("TestAgent", parse_cli_args=False)
    old_agent = cast("AgentProtocol", _Agent("main"))
    new_agent = cast("AgentProtocol", _Agent("main"))
    wrapper = AgentApp({"main": old_agent})
    state = ManagedRunState(
        runtime=_run_runtime(global_prompt_context={"workspaceRoot": "/current"}),
        primary_instance=AgentInstance(wrapper, {"main": old_agent}),
        wrapper=wrapper,
        active_agents={"main": old_agent},
    )
    fast._agent_registry_version = 1
    calls: list[str] = []

    async def fake_instantiate(runtime: RunRuntime, app_override: AgentApp | None = None):
        del runtime, app_override
        calls.append("instantiate")
        return AgentInstance(wrapper, {"main": new_agent}, registry_version=1)

    async def fake_finalize(agents: dict[str, AgentProtocol], runtime: RunRuntime) -> None:
        del runtime
        assert agents == {"main": new_agent}
        calls.append("finalize")

    async def fake_refresh_result(
        agents: dict[str, AgentProtocol],
        updated_agents: dict[str, AgentProtocol] | None = None,
    ) -> AgentRefreshResult:
        assert agents == {"main": new_agent}
        assert updated_agents is None
        calls.append("hydrate")
        return AgentRefreshResult(changed=True)

    async def fake_dispose(runtime: RunRuntime, instance: AgentInstance) -> None:
        del runtime, instance
        calls.append("dispose")

    monkeypatch.setattr(fast, "_instantiate_agent_instance", fake_instantiate)
    monkeypatch.setattr(fast, "_finalize_updated_agents", fake_finalize)
    monkeypatch.setattr(fast, "_refresh_result_from_session_restore", fake_refresh_result)
    monkeypatch.setattr(fast, "_dispose_agent_instance", fake_dispose)
    monkeypatch.setattr(fast, "_sync_agent_card_mcp_servers", lambda: None)

    await fast._refresh_shared_instance(state)

    assert calls == ["instantiate", "finalize", "hydrate", "dispose"]


@pytest.mark.asyncio
async def test_refresh_finalizes_partial_rebuild_before_session_hydration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = FastAgent("TestAgent", parse_cli_args=False)
    changed_agent = cast("AgentProtocol", _Agent("changed"))
    unchanged_agent = cast("AgentProtocol", _Agent("unchanged"))
    rebuilt_agent = cast("AgentProtocol", _Agent("changed"))
    wrapper = AgentApp({"changed": changed_agent, "unchanged": unchanged_agent})
    state = ManagedRunState(
        runtime=_run_runtime(global_prompt_context={"workspaceRoot": "/current"}),
        primary_instance=AgentInstance(
            wrapper,
            {"changed": changed_agent, "unchanged": unchanged_agent},
        ),
        wrapper=wrapper,
        active_agents={"changed": changed_agent, "unchanged": unchanged_agent},
    )
    fast._agent_registry_version = 1
    fast._agent_card_last_changed.add("changed")
    calls: list[str] = []

    async def fake_rebuild(
        active_agents: dict[str, AgentProtocol],
        impacted: set[str],
        model_factory_func,
    ) -> None:
        del model_factory_func
        assert impacted == {"changed"}
        active_agents["changed"] = rebuilt_agent
        calls.append("rebuild")

    async def fake_finalize(agents: dict[str, AgentProtocol], runtime: RunRuntime) -> None:
        del runtime
        assert agents == {"changed": rebuilt_agent}
        calls.append("finalize")

    async def fake_refresh_result(
        agents: dict[str, AgentProtocol],
        updated_agents: dict[str, AgentProtocol] | None = None,
    ) -> AgentRefreshResult:
        assert agents == {"changed": rebuilt_agent, "unchanged": unchanged_agent}
        assert updated_agents == {"changed": rebuilt_agent}
        calls.append("hydrate")
        return AgentRefreshResult(changed=True)

    monkeypatch.setattr(fast, "_rebuild_impacted_agents", fake_rebuild)
    monkeypatch.setattr(fast, "_finalize_updated_agents", fake_finalize)
    monkeypatch.setattr(fast, "_refresh_result_from_session_restore", fake_refresh_result)
    monkeypatch.setattr(fast, "_sync_agent_card_mcp_servers", lambda: None)

    await fast._refresh_shared_instance(state)

    assert calls == ["rebuild", "finalize", "hydrate"]


@pytest.mark.asyncio
async def test_runtime_callback_instances_inherit_mcp_runtime_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = FastAgent("TestAgent", parse_cli_args=False)
    agent = cast("AgentProtocol", _Agent("main"))
    wrapper = AgentApp({"main": agent})
    state = ManagedRunState(
        runtime=RunRuntime(
            model_factory_func=_unused_model_factory,
            global_prompt_context=None,
            is_acp_server_mode=True,
            noenv_mode=False,
            managed_instances=[],
            instance_lock=asyncio.Lock(),
            shell_executor=_FakeShellExecutor(),
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
            model_factory_func=_unused_model_factory,
            global_prompt_context=None,
            is_acp_server_mode=True,
            noenv_mode=False,
            managed_instances=[],
            instance_lock=asyncio.Lock(),
            shell_executor=_FakeShellExecutor(),
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


@pytest.mark.asyncio
async def test_load_card_tools_rejects_default_agent_without_agent_tool_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = FastAgent("TestAgent", parse_cli_args=False)
    fast.args = argparse.Namespace(card_tools=["tool-cards"], agent="main")
    main_agent = cast("AgentProtocol", _Agent("main"))
    wrapper = AgentApp({"main": main_agent})
    state = ManagedRunState(
        runtime=RunRuntime(
            model_factory_func=_unused_model_factory,
            global_prompt_context=None,
            is_acp_server_mode=False,
            noenv_mode=False,
            managed_instances=[],
            instance_lock=asyncio.Lock(),
            shell_executor=_FakeShellExecutor(),
        ),
        primary_instance=AgentInstance(wrapper, {"main": main_agent}),
        wrapper=wrapper,
        active_agents={"main": main_agent},
    )

    monkeypatch.setattr(fast, "load_agents", lambda _source: ["tool"])

    async def refresh() -> AgentRefreshResult:
        state.active_agents["tool"] = cast("AgentProtocol", _Agent("tool"))
        return AgentRefreshResult(changed=True)

    with pytest.raises(AgentConfigError, match="does not support agents-as-tools"):
        await fast._apply_card_tool_cli_option(state, refresh)


@pytest.mark.asyncio
async def test_start_server_preserves_existing_optional_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = FastAgent("TestAgent", parse_cli_args=False)
    original_args = argparse.Namespace(
        quiet=False,
        model=None,
        agent="main",
        reload=False,
        watch=False,
        card_tools=None,
    )
    fast.args = original_args
    captured_args: list[argparse.Namespace] = []

    @asynccontextmanager
    async def fake_run():
        captured_args.append(argparse.Namespace(**vars(fast.args)))
        yield None

    monkeypatch.setattr(fast, "run", fake_run)

    await fast.start_server(transport="http")

    assert fast.args is original_args
    assert captured_args
    assert captured_args[0].model is None
    assert captured_args[0].host == "127.0.0.1"
    assert captured_args[0].agent == "main"
    assert captured_args[0].reload is False
    assert captured_args[0].watch is False
    assert captured_args[0].card_tools is None
