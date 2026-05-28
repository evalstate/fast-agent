import json

from typer.testing import CliRunner

from fast_agent.cli.commands import model as model_command


def test_model_presets_filters_by_provider_json() -> None:
    result = CliRunner().invoke(
        model_command.app,
        ["presets", "--provider", "anthropic", "--json"],
    )

    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)

    assert rows
    assert {row["provider"] for row in rows} == {"anthropic"}
    assert any(row["alias"] == "opus" and row["model"] == "claude-opus-4-7" for row in rows)


def test_model_presets_text_shows_downstream_model() -> None:
    result = CliRunner().invoke(
        model_command.app,
        ["presets", "--provider", "responses"],
    )

    assert result.exit_code == 0, result.output
    assert "Model presets (responses)" in result.output
    assert "gpt55" in result.output
    assert "responses.gpt-5.5" in result.output
