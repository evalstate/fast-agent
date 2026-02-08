import click
import typer

from fast_agent.cli.commands import acp as acp_command


def test_acp_command_builds_request_with_watch() -> None:
    ctx = typer.Context(click.Command("acp"))
    request = acp_command._build_run_request(
        ctx=ctx,
        name="fast-agent-acp",
        instruction=None,
        config_path=None,
        servers=None,
        agent_cards=["./agents"],
        card_tools=["./tool-cards"],
        urls=None,
        auth=None,
        model=None,
        env_dir=None,
        noenv=False,
        skills_dir=None,
        npx=None,
        uvx=None,
        stdio=None,
        description="Chat with {agent}",
        host="127.0.0.1",
        port=8010,
        shell=False,
        instance_scope=acp_command.serve.InstanceScope.CONNECTION,
        no_permissions=False,
        resume=None,
        reload=True,
        watch=True,
    )

    assert request.mode == "serve"
    assert request.transport == "acp"
    assert request.host == "127.0.0.1"
    assert request.port == 8010
    assert request.agent_cards == ["./agents"]
    assert request.card_tools == ["./tool-cards"]
    assert request.reload is True
    assert request.watch is True


def test_acp_command_noenv_forces_permissions_disabled() -> None:
    ctx = typer.Context(click.Command("acp"))
    request = acp_command._build_run_request(
        ctx=ctx,
        name="fast-agent-acp",
        instruction=None,
        config_path=None,
        servers=None,
        agent_cards=None,
        card_tools=None,
        urls=None,
        auth=None,
        model=None,
        env_dir=None,
        noenv=True,
        skills_dir=None,
        npx=None,
        uvx=None,
        stdio=None,
        description=None,
        host="127.0.0.1",
        port=8010,
        shell=False,
        instance_scope=acp_command.serve.InstanceScope.CONNECTION,
        no_permissions=False,
        resume=None,
        reload=False,
        watch=False,
    )

    assert request.noenv is True
    assert request.permissions_enabled is False
