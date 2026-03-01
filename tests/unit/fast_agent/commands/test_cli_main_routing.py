from __future__ import annotations

import subprocess
import sys

import pytest
from click.utils import strip_ansi

from fast_agent.cli.__main__ import _first_positional_argument


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--env", "demo", "demo", "--help"], "demo"),
        (["--env", "demo", "--message", "hi"], None),
        (["-m", "serve", "--help"], None),
        (["--env=demo", "cards", "--help"], "cards"),
        (["--", "demo"], "demo"),
    ],
)
def test_first_positional_argument_skips_option_values(
    arguments: list[str],
    expected: str | None,
) -> None:
    assert _first_positional_argument(arguments) == expected


def _run_fast_agent_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "fast_agent.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_auto_routes_to_go_when_env_value_matches_subcommand() -> None:
    result = _run_fast_agent_cli("--env", "demo", "--message", "hi", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "go [OPTIONS] COMMAND" in output
    assert "--message" in output


def test_auto_routes_to_go_when_message_matches_subcommand() -> None:
    result = _run_fast_agent_cli("-m", "serve", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "go [OPTIONS] COMMAND" in output


def test_demo_subcommand_still_detected_after_env_option_value() -> None:
    result = _run_fast_agent_cli("--env", "demo", "demo", "--help")
    output = strip_ansi(result.stdout)

    assert result.returncode == 0, result.stderr
    assert "demo [OPTIONS] COMMAND" in output
    assert "Demo commands for UI features." in output
