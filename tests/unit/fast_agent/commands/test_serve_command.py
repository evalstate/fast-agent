import click
import typer

from fast_agent.cli.commands import go as go_command
from fast_agent.cli.commands import serve as serve_command


def test_run_async_agent_passes_serve_mode(monkeypatch):
    captured: dict = {}

    async def fake_run_agent(**kwargs):
        captured.update(kwargs)

    # Avoid altering whichever loop pytest might have prepared
    monkeypatch.setattr(go_command, "_run_agent", fake_run_agent)
    monkeypatch.setattr(go_command, "_set_asyncio_exception_handler", lambda loop: None)

    go_command.run_async_agent(
        name="test-agent",
        instruction="test instruction",
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        model=None,
        message=None,
        prompt_file=None,
        stdio_commands=None,
        agent_name="agent",
        skills_directory=None,
        shell_enabled=False,
        mode="serve",
        transport="sse",
        host="127.0.0.1",
        port=9123,
        tool_description="Send requests to {agent}",
    )

    assert captured["mode"] == "serve"
    assert captured["transport"] == "sse"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9123
    assert captured["tool_description"] == "Send requests to {agent}"


def test_serve_command_invokes_run_async_agent(monkeypatch):
    captured: dict = {}

    def fake_run_async_agent(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(serve_command, "run_async_agent", fake_run_async_agent)

    ctx = typer.Context(click.Command("serve"))
    serve_command.serve(
        ctx=ctx,
        name="fast-agent",
        instruction=None,
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        model=None,
        skills_dir=None,
        npx=None,
        uvx=None,
        stdio="python tool_server.py",
        description="Chat with {agent}",
        transport=serve_command.ServeTransport.STDIO,
        host="127.0.0.1",
        port=7010,
        shell=False,
    )

    assert captured["mode"] == "serve"
    assert captured["transport"] == "stdio"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 7010
    assert captured["stdio_commands"] == ["python tool_server.py"]
    assert captured["tool_description"] == "Chat with {agent}"
