from __future__ import annotations

import re

from typer.testing import CliRunner

from fast_agent.cli.main import app
from fast_agent.cli.prompt_argument_hint import prompt_argument_message

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """Error panels are colored and wrapped to the terminal width (CI: 80 columns)."""
    text = _ANSI.sub("", output)
    text = re.sub(r"[│╭╮╰╯─]", " ", text)
    return " ".join(text.split())


def test_prompt_argument_message_points_to_working_alternatives() -> None:
    message = prompt_argument_message("review the login bug")
    assert "does not take a prompt as an argument" in message
    assert "hc send" in message
    assert "fast-agent go --message 'review the login bug'" in message


def test_prompt_argument_explains_how_to_send_a_first_message() -> None:
    result = CliRunner().invoke(app, ["review the login bug"])
    assert result.exit_code == 2
    output = _plain(result.output)
    assert "does not take a prompt as an argument" in output
    assert "--message" in output


def test_go_prompt_argument_gets_the_same_explanation() -> None:
    result = CliRunner().invoke(app, ["go", "review the login bug"])
    assert result.exit_code == 2
    assert "does not take a prompt as an argument" in _plain(result.output)


def test_unknown_single_word_command_keeps_the_usual_error() -> None:
    result = CliRunner().invoke(app, ["gp"])
    assert result.exit_code == 2
    assert "No such command" in _plain(result.output)
