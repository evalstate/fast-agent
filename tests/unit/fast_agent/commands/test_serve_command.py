from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import click
import pytest
import typer
from typer.testing import CliRunner

from fast_agent.cli.commands import go as go_command
from fast_agent.cli.commands import serve as serve_command
from fast_agent.core.fastagent import FastAgent

if TYPE_CHECKING:
    from fast_agent.cli.runtime.run_request import AgentRunRequest


def test_run_async_agent_passes_serve_mode() -> None:
    run_kwargs = go_command._build_run_agent_kwargs(
        name="test-agent",
        instruction="test instruction",
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message=None,
        prompt_file=None,
        result_file=None,
        resume=None,
        stdio_commands=None,
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        environment_dir=None,
        shell_enabled=False,
        mode="serve",
        transport="http",
        host="127.0.0.1",
        port=9123,
        tool_description="Send requests to {agent}",
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
    )

    assert run_kwargs["mode"] == "serve"
    assert run_kwargs["transport"] == "http"
    assert run_kwargs["host"] == "127.0.0.1"
    assert run_kwargs["port"] == 9123
    assert run_kwargs["tool_description"] == "Send requests to {agent}"
    assert run_kwargs["instance_scope"] == "shared"


def test_serve_command_builds_run_request() -> None:
    ctx = typer.Context(click.Command("serve"))
    request = serve_command._build_run_request(
        ctx=ctx,
        name="fast-agent",
        instruction=None,
        config_path=None,
        servers=None,
        agent_cards=["./agents"],
        card_tools=["./tool-cards"],
        urls=None,
        auth=None,
        client_metadata_url=None,
        model=None,
        skills_dir=None,
        env_dir=None,
        noenv=False,
        force_smart=False,
        npx=None,
        uvx=None,
        stdio="python tool_server.py",
        description="Chat with {agent}",
        tool_name_template=None,
        transport=serve_command.ServeTransport.STDIO,
        host="127.0.0.1",
        port=7010,
        shell=False,
        instance_scope=serve_command.InstanceScope.CONNECTION,
        no_permissions=False,
        reload=True,
        watch=True,
    )

    assert request.mode == "serve"
    assert request.transport == "stdio"
    assert request.host == "127.0.0.1"
    assert request.port == 7010
    assert request.tool_description == "Chat with {agent}"
    assert request.instance_scope == "connection"
    assert request.agent_cards == ["./agents"]
    assert request.card_tools == ["./tool-cards"]
    assert request.reload is True
    assert request.watch is True
    assert request.stdio_servers is not None
    first_stdio_config = next(iter(request.stdio_servers.values()))
    assert first_stdio_config["command"] == "python"
    assert first_stdio_config["args"] == ["tool_server.py"]


def test_serve_command_noenv_forces_permissions_disabled() -> None:
    ctx = typer.Context(click.Command("serve"))
    request = serve_command._build_run_request(
        ctx=ctx,
        name="fast-agent",
        instruction=None,
        config_path=None,
        servers=None,
        agent_cards=None,
        card_tools=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        model=None,
        skills_dir=None,
        env_dir=None,
        noenv=True,
        force_smart=False,
        npx=None,
        uvx=None,
        stdio=None,
        description=None,
        tool_name_template=None,
        transport=serve_command.ServeTransport.ACP,
        host="127.0.0.1",
        port=7010,
        shell=False,
        instance_scope=serve_command.InstanceScope.CONNECTION,
        no_permissions=False,
        reload=False,
        watch=False,
    )

    assert request.noenv is True
    assert request.permissions_enabled is False


def test_serve_command_builds_a2a_request() -> None:
    ctx = typer.Context(click.Command("serve"))
    request = serve_command._build_run_request(
        ctx=ctx,
        name="fast-agent-a2a",
        instruction=None,
        config_path=None,
        servers=None,
        agent_cards=["./agents"],
        card_tools=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        model=None,
        skills_dir=None,
        env_dir=None,
        noenv=False,
        force_smart=False,
        npx=None,
        uvx=None,
        stdio=None,
        description=None,
        tool_name_template=None,
        transport=serve_command.ServeTransport.A2A,
        host="127.0.0.1",
        port=41241,
        shell=False,
        instance_scope=serve_command.InstanceScope.SHARED,
        no_permissions=False,
        reload=False,
        watch=False,
    )

    assert request.mode == "serve"
    assert request.transport == "a2a"
    assert request.host == "127.0.0.1"
    assert request.port == 41241
    assert request.instance_scope == "shared"


def test_serve_a2a_subcommand_builds_a2a_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_request(request: object) -> None:
        captured["request"] = request

    monkeypatch.setattr(serve_command, "run_request", fake_run_request)

    result = CliRunner().invoke(
        serve_command.app,
        [
            "a2a",
            "--host",
            "127.0.0.1",
            "--port",
            "41241",
            "--agent-cards",
            "./agents",
            "--model",
            "codexresponses.gpt-5.4-mini",
        ],
    )

    assert result.exit_code == 0, result.output
    request = cast("AgentRunRequest", captured["request"])
    assert request.mode == "serve"
    assert request.transport == "a2a"
    assert request.name == "fast-agent-a2a"
    assert request.host == "127.0.0.1"
    assert request.port == 41241
    assert request.agent_cards == ["./agents"]
    assert request.model == "codexresponses.gpt-5.4-mini"


@pytest.mark.asyncio
async def test_fastagent_run_a2a_server_passes_instance_scope(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeA2AServer:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def run_async(self, *, host: str, port: int) -> None:
            captured["run_host"] = host
            captured["run_port"] = port

    monkeypatch.setattr("fast_agent.a2a.AgentA2AServer", FakeA2AServer)

    fast = FastAgent.__new__(FastAgent)
    fast.name = "fast-agent-a2a"
    fast.args = SimpleNamespace(
        server_description=None,
        server_name=None,
        host="127.0.0.1",
        port=41241,
        instance_scope="request",
    )
    state = SimpleNamespace(primary_instance=object())
    callbacks = SimpleNamespace(
        create_instance=object(),
        dispose_instance=object(),
    )

    run_a2a_server = cast("Any", FastAgent._run_a2a_server)
    await run_a2a_server(fast, state, callbacks)

    assert captured["primary_instance"] is state.primary_instance
    assert captured["create_instance"] is callbacks.create_instance
    assert captured["dispose_instance"] is callbacks.dispose_instance
    assert captured["instance_scope"] == "request"
    assert captured["run_host"] == "127.0.0.1"
    assert captured["run_port"] == 41241


def test_serve_command_builds_request_with_missing_shell_cwd_override() -> None:
    ctx = typer.Context(click.Command("serve"))
    request = serve_command._build_run_request(
        ctx=ctx,
        name="fast-agent",
        instruction=None,
        config_path=None,
        servers=None,
        agent_cards=None,
        card_tools=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        model=None,
        skills_dir=None,
        env_dir=None,
        noenv=False,
        force_smart=False,
        npx=None,
        uvx=None,
        stdio=None,
        description=None,
        tool_name_template=None,
        transport=serve_command.ServeTransport.ACP,
        host="127.0.0.1",
        port=7010,
        shell=False,
        instance_scope=serve_command.InstanceScope.CONNECTION,
        no_permissions=False,
        reload=False,
        watch=False,
        missing_shell_cwd=serve_command.MissingShellCwdPolicy.ERROR,
    )

    assert request.missing_shell_cwd_policy == "error"


def test_resolve_instance_scope_defaults_acp_to_connection() -> None:
    ctx = typer.Context(click.Command("serve"))
    ctx.set_parameter_source("instance_scope", click.core.ParameterSource.DEFAULT)

    resolved = serve_command._resolve_instance_scope(
        ctx,
        transport=serve_command.ServeTransport.ACP,
        instance_scope=serve_command.InstanceScope.SHARED,
    )

    assert resolved == serve_command.InstanceScope.CONNECTION


def test_resolve_instance_scope_rejects_explicit_shared_for_acp() -> None:
    ctx = typer.Context(click.Command("serve"))
    ctx.set_parameter_source("instance_scope", click.core.ParameterSource.COMMANDLINE)

    with pytest.raises(typer.BadParameter, match="ACP is always connection-scoped"):
        serve_command._resolve_instance_scope(
            ctx,
            transport=serve_command.ServeTransport.ACP,
            instance_scope=serve_command.InstanceScope.SHARED,
        )


def test_resolve_instance_scope_rejects_explicit_request_for_acp() -> None:
    ctx = typer.Context(click.Command("serve"))
    ctx.set_parameter_source("instance_scope", click.core.ParameterSource.COMMANDLINE)

    with pytest.raises(typer.BadParameter, match="ACP is always connection-scoped"):
        serve_command._resolve_instance_scope(
            ctx,
            transport=serve_command.ServeTransport.ACP,
            instance_scope=serve_command.InstanceScope.REQUEST,
        )
