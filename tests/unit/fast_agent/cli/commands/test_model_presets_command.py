import json

import pytest
from typer.testing import CliRunner

from fast_agent.cli.commands import model as model_command
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider_types import Provider


def test_model_presets_filters_by_provider_json() -> None:
    result = CliRunner().invoke(
        model_command.app,
        ["presets", "--provider", "anthropic", "--json"],
    )

    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)

    assert rows
    assert {row["provider"] for row in rows} == {"anthropic"}
    opus = next(row for row in rows if row["alias"] == "opus")
    assert opus["model"] == ModelFactory.parse_model_string("opus").model_name


def test_model_presets_provider_filter_normalizes_aliases() -> None:
    assert model_command._model_presets_provider_filter("  HuggingFace  ") == Provider.HUGGINGFACE


def test_model_presets_text_shows_downstream_model() -> None:
    result = CliRunner().invoke(
        model_command.app,
        ["presets", "--provider", "responses"],
    )

    assert result.exit_code == 0, result.output
    assert "Model presets (responses)" in result.output
    assert "gpt55" in result.output
    assert "responses.gpt-5.5" in result.output


def test_model_presets_json_preserves_query_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ModelFactory,
        "get_runtime_presets",
        lambda: {"planned": "responses.gpt-5.5?reasoning=medium"},
    )

    result = CliRunner().invoke(
        model_command.app,
        ["presets", "--provider", "responses", "--json"],
    )

    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert len(rows) == 1
    assert rows[0]["alias"] == "planned"
    assert rows[0]["provider"] == "responses"
    assert rows[0]["model"] == "gpt-5.5"
    assert rows[0]["expanded"] == "responses.gpt-5.5?reasoning=medium"
