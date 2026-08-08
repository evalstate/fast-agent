from __future__ import annotations

import subprocess
import sys

import click
import typer.main

from fast_agent.cli.main import LAZY_SUBCOMMAND_HELP, LAZY_SUBCOMMANDS, LazyGroup, app


def test_root_help_does_not_import_lazy_subcommands() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from typer.testing import CliRunner; "
                "from fast_agent.cli.main import LAZY_SUBCOMMANDS, app; "
                "result = CliRunner().invoke(app, ['--help']); "
                "assert result.exit_code == 0, result.exception; "
                "modules = {target.split(':', 1)[0] for target in LAZY_SUBCOMMANDS.values()}; "
                "assert modules.isdisjoint(sys.modules), modules & sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_root_help_metadata_matches_subcommand_help() -> None:
    root_command = typer.main.get_command(app)
    assert isinstance(root_command, LazyGroup)
    context = click.Context(root_command)

    assert LAZY_SUBCOMMAND_HELP.keys() == LAZY_SUBCOMMANDS.keys()
    for command_name, expected_help in LAZY_SUBCOMMAND_HELP.items():
        command = root_command.get_command(context, command_name)
        assert command is not None
        assert (command.short_help or command.help) == expected_help
