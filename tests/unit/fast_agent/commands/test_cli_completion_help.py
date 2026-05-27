from typer.testing import CliRunner

from fast_agent.cli import main as cli_main
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
    setup,
)
from fast_agent.cli.main import app as root_app


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
    ]

    for command_app in command_apps:
        result = runner.invoke(command_app, ["--help"], terminal_width=160)
        assert result.exit_code == 0
        assert "--install-completion" not in result.output
        assert "--show-completion" not in result.output


def test_root_help_uses_lazy_command_metadata(monkeypatch):
    imported_modules: list[str] = []

    def fail_import(name: str):
        imported_modules.append(name)
        raise AssertionError(f"root help should not import lazy command module: {name}")

    monkeypatch.setattr(cli_main.importlib, "import_module", fail_import)

    result = CliRunner().invoke(root_app, ["--help"], terminal_width=160)

    assert result.exit_code == 0
    assert "go" in result.output
    assert "Run an interactive agent" in result.output
    assert imported_modules == []
