import click
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
        agent_cards=None,
        card_tools=None,
        model=None,
        message=None,
        prompt_file=None,
        resume=None,
        stdio_commands=None,
        agent_name="agent",
        skills_directory=None,
        environment_dir=None,
        shell_enabled=False,
        mode="serve",
        transport="sse",
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
    assert run_kwargs["transport"] == "sse"
    assert run_kwargs["host"] == "127.0.0.1"
    assert run_kwargs["port"] == 9123
    assert run_kwargs["tool_description"] == "Send requests to {agent}"
    assert run_kwargs["instance_scope"] == "shared"


def test_serve_command_invokes_run_async_agent() -> None:
    ctx = typer.Context(click.Command("serve"))
    captured = serve_command._build_run_async_agent_kwargs(
        ctx=ctx,
        name="fast-agent",
        instruction=None,
        config_path=None,
        servers=None,
        agent_cards=["./agents"],
        card_tools=["./tool-cards"],
        urls=None,
        auth=None,
        model=None,
        skills_dir=None,
        env_dir=None,
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

    assert captured["mode"] == "serve"
    assert captured["transport"] == "stdio"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 7010
    assert captured["stdio_commands"] == ["python tool_server.py"]
    assert captured["tool_description"] == "Chat with {agent}"
    assert captured["instance_scope"] == "connection"
    assert captured["agent_cards"] == ["./agents"]
    assert captured["card_tools"] == ["./tool-cards"]
    assert captured["reload"] is True
    assert captured["watch"] is True
