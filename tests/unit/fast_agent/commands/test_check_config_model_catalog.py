import os
from pathlib import Path

import pytest

from fast_agent.cli.commands.check_config import (
    show_check_summary,
    show_provider_model_catalog,
)


def test_show_check_summary_includes_current_model_suggestions(tmp_path: Path, capsys) -> None:
    env_dir = tmp_path / ".fast-agent"
    env_dir.mkdir(parents=True)

    original_openai_key = os.environ.get("OPENAI_API_KEY")
    cwd = Path.cwd()
    try:
        os.environ["OPENAI_API_KEY"] = "sk-openai-env"
        os.chdir(tmp_path)
        show_check_summary(env_dir=env_dir)
    finally:
        os.chdir(cwd)
        if original_openai_key is not None:
            os.environ["OPENAI_API_KEY"] = original_openai_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    output = capsys.readouterr().out
    assert "Current Model Suggestions" in output
    assert "All Known Models (" not in output


def test_show_provider_model_catalog_openai_defaults_to_curated_aliases(capsys) -> None:
    show_provider_model_catalog("openai")

    output = capsys.readouterr().out
    assert "OpenAI model catalog (curated aliases)" in output
    assert "OpenAI" in output
    assert "Responses" in output
    assert "Codex Responses" in output
    assert "gpt-4.1-mini" in output
    assert "gpt-5-mini" in output
    assert "codexspark" in output


def test_show_provider_model_catalog_openai_all_includes_openai_family(capsys) -> None:
    show_provider_model_catalog("openai", show_all=True)

    output = capsys.readouterr().out
    assert "OpenAI model catalog (all models)" in output
    assert "OpenAI (" in output
    assert "Responses (" in output
    assert "Codex Responses (" in output
    assert "gpt-4.1" in output
    assert "o1" in output
    assert "gpt-5.3-codex-spark" in output


def test_show_provider_model_catalog_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        show_provider_model_catalog("not-a-provider")
