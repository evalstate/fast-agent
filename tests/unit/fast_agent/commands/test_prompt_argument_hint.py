from __future__ import annotations

from typer.testing import CliRunner

from fast_agent.cli.main import app


def test_prompt_argument_explains_how_to_send_a_first_message() -> None:
    result = CliRunner().invoke(app, ["review the login bug"])
    assert result.exit_code == 2
    assert "does not take a prompt as an argument" in result.output
    assert "--message" in result.output


def test_go_prompt_argument_gets_the_same_explanation() -> None:
    result = CliRunner().invoke(app, ["go", "review the login bug"])
    assert result.exit_code == 2
    assert "does not take a prompt as an argument" in result.output


def test_unknown_single_word_command_keeps_the_usual_error() -> None:
    result = CliRunner().invoke(app, ["gp"])
    assert result.exit_code == 2
    assert "No such command" in result.output
