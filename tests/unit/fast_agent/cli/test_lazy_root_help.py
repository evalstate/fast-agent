from __future__ import annotations

import subprocess
import sys

import pytest
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


@pytest.mark.skipif(sys.version_info < (3, 15), reason="__lazy_modules__ requires Python 3.15+")
def test_help_defers_lazy_modules() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from typer.testing import CliRunner; "
                "from fast_agent.cli.main import app; "
                "deferred = {'fast_agent.cli.update_check', 'fast_agent.cli.display', "
                "'fast_agent.mcp.connect_targets'}; "
                "results = [CliRunner().invoke(app, args) for args in (['--help'], ['go', '--help'])]; "
                "assert all(r.exit_code == 0 for r in results), [r.exception for r in results]; "
                "assert deferred.isdisjoint(sys.modules), deferred & sys.modules.keys()"
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
    context = typer.Context(root_command)

    assert LAZY_SUBCOMMAND_HELP.keys() == LAZY_SUBCOMMANDS.keys()
    for command_name, expected_help in LAZY_SUBCOMMAND_HELP.items():
        command = root_command.get_command(context, command_name)
        assert command is not None
        assert (command.short_help or command.help) == expected_help
