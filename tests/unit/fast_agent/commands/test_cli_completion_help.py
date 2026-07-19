from typer.testing import CliRunner

from fast_agent.cli.commands import (
    acp,
    auth,
    batch,
    cards,
    check_config,
    config,
    demo,
    go,
    model,
    quickstart,
    serve,
    session,
    setup,
)


def test_command_help_hides_typer_completion_options():
    runner = CliRunner()
    command_apps = [
        go.app,
        serve.app,
        acp.app,
        cards.app,
        batch.app,
        auth.app,
        config.app,
        demo.app,
        model.app,
        setup.app,
        check_config.app,
        quickstart.app,
        session.app,
    ]

    for app in command_apps:
        result = runner.invoke(app, ["--help"], terminal_width=160)
        assert result.exit_code == 0
        assert "--install-completion" not in result.output
        assert "--show-completion" not in result.output
