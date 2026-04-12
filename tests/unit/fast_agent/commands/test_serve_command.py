import click
import pytest
import typer

from fast_agent.cli.commands import go as go_command
from fast_agent.cli.commands import serve as serve_command


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
