import click
import typer

from fast_agent.cli.commands import acp as acp_command


def test_acp_command_passes_watch(monkeypatch):
    captured: dict = {}

    def fake_run_async_agent(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(acp_command, "run_async_agent", fake_run_async_agent)

    ctx = typer.Context(click.Command("acp"))
    acp_command.run_acp(
        ctx=ctx,
        name="fast-agent-acp",
        instruction=None,
        config_path=None,
        servers=None,
        agent_cards=["./agents"],
        urls=None,
        auth=None,
        model=None,
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
        reload=True,
        watch=True,
    )

    assert captured["mode"] == "serve"
    assert captured["transport"] == "acp"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8010
    assert captured["agent_cards"] == ["./agents"]
    assert captured["reload"] is True
    assert captured["watch"] is True
