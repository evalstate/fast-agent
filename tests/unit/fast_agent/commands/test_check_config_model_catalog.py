import os
from pathlib import Path

import pytest

from fast_agent.cli.commands.check_config import (
    show_check_summary,
    show_model_secret_requirements,
    show_models_overview,
    show_provider_model_catalog,
)
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider_types import Provider

OPENAI_FAMILY_PROVIDERS = (
    Provider.OPENAI,
    Provider.RESPONSES,
    Provider.CODEX_RESPONSES,
)


def _collapse(text: str) -> str:
    return "".join(text.split())


def _current_catalog_models(providers: tuple[Provider, ...]) -> set[str]:
    models: set[str] = set()
    for provider in providers:
        models.update(ModelSelectionCatalog.list_current_models(provider))
    return models


def _first_additional_model(providers: tuple[Provider, ...]) -> str:
    current_models = _current_catalog_models(providers)
    for provider in providers:
        for model in ModelSelectionCatalog.list_all_models(provider):
            if model not in current_models:
                return model
    raise AssertionError("expected at least one non-curated model in provider catalog")


def test_show_check_summary_points_to_check_models_command(tmp_path: Path, capsys) -> None:
    home = tmp_path / ".fast-agent"
    home.mkdir(parents=True)

    original_openai_key = os.environ.get("OPENAI_API_KEY")
    cwd = Path.cwd()
    try:
        os.environ["OPENAI_API_KEY"] = "sk-openai-env"
        os.chdir(tmp_path)
        show_check_summary(home=home)
    finally:
        os.chdir(cwd)
        if original_openai_key is not None:
            os.environ["OPENAI_API_KEY"] = original_openai_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    output = capsys.readouterr().out
    assert "Current Model Suggestions" not in output
    assert "Use fast-agent check models" in output


def test_show_provider_model_catalog_openai_defaults_to_curated_aliases(capsys) -> None:
    show_provider_model_catalog("openai")

    output = capsys.readouterr().out
    collapsed_output = _collapse(output)
    assert "OpenAI model catalog (curated)" in output
    assert "Tags" in output
    assert "fast" in output
    assert "OpenAI" in output
    assert "Responses" in output
    assert "Codex Responses" in output
    assert "All known models" not in output

    for provider in OPENAI_FAMILY_PROVIDERS:
        current_entry = ModelSelectionCatalog.list_current_entries(provider)[0]
        fast_entry = next(
            entry for entry in ModelSelectionCatalog.list_current_entries(provider) if entry.fast
        )
        assert current_entry.alias in output
        assert _collapse(current_entry.model) in collapsed_output
        assert fast_entry.alias in output
        assert _collapse(fast_entry.model) in collapsed_output

    assert "More models are available" in output


def test_show_provider_model_catalog_openai_all_includes_openai_family(capsys) -> None:
    show_provider_model_catalog("openai", show_all=True)

    output = capsys.readouterr().out
    collapsed_output = _collapse(output)
    additional_model = _first_additional_model(OPENAI_FAMILY_PROVIDERS)

    assert "OpenAI model catalog (curated + all models)" in output
    assert "All known models" in output
    assert "catalog" in output
    assert _collapse(additional_model) in collapsed_output


def test_show_models_overview_includes_provider_args_and_named_aliases(
    tmp_path: Path,
    capsys,
) -> None:
    home = tmp_path / ".fast-agent"
    home.mkdir(parents=True)
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_references:",
                "  system:",
                "    fast: responses.gpt-5-mini?reasoning=low",
            ]
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        show_models_overview(home=home)
    finally:
        os.chdir(cwd)

    output = capsys.readouterr().out
    assert "fast-agent check models <provider>" in output
    assert "Provider Arg" in output
    assert "Active" in output
    assert "openai" in output
    assert "Named Model Aliases" in output
    assert "$system.fast" in output
    assert "responses.gpt-5-mini?reasoning=low" in output


def test_show_models_overview_uses_default_env_config_aliases(tmp_path: Path, capsys) -> None:
    home = tmp_path / ".fast-agent"
    home.mkdir(parents=True)
    (home / "fastagent.config.yaml").write_text(
        "\n".join(
            [
                "model_references:",
                "  system:",
                "    envfast: responses.gpt-5-mini?reasoning=low",
            ]
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    previous_home = os.environ.get("FAST_AGENT_HOME")
    try:
        os.environ.pop("FAST_AGENT_HOME", None)
        os.chdir(tmp_path)
        show_models_overview(home=None)
    finally:
        os.chdir(cwd)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home

    output = capsys.readouterr().out
    assert "$system.envfast" in output


def test_show_provider_model_catalog_scopes_overlays_to_requested_env(
    tmp_path: Path,
    capsys,
) -> None:
    home = tmp_path / "project-env"
    ambient_home = tmp_path / "ambient-env"
    (home / "model-overlays").mkdir(parents=True)
    (ambient_home / "model-overlays").mkdir(parents=True)
    (home / "model-overlays" / "projectoverlay.yaml").write_text(
        "name: projectoverlay\nprovider: responses\nmodel: overlay-tests/project\n",
        encoding="utf-8",
    )
    (ambient_home / "model-overlays" / "ambientoverlay.yaml").write_text(
        "name: ambientoverlay\nprovider: responses\nmodel: overlay-tests/ambient\n",
        encoding="utf-8",
    )

    cwd = Path.cwd()
    previous_home = os.environ.get("FAST_AGENT_HOME")
    try:
        os.environ["FAST_AGENT_HOME"] = str(ambient_home)
        os.chdir(tmp_path)
        show_provider_model_catalog("responses", home=home)
    finally:
        os.chdir(cwd)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home

    output = capsys.readouterr().out
    assert "projectoverlay" in output
    assert "responses.overlay-tests/project" in output
    assert "ambientoverlay" not in output
    assert "responses.overlay-tests/ambient" not in output


def test_show_provider_model_catalog_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        show_provider_model_catalog("not-a-provider")


def test_show_check_summary_reports_invalid_effective_model_references(
    tmp_path: Path,
    capsys,
) -> None:
    home = tmp_path / ".fast-agent"
    home.mkdir(parents=True)

    (tmp_path / "fastagent.config.yaml").write_text("logger:\n  level: warning\n", encoding="utf-8")
    (home / "fastagent.config.yaml").write_text(
        "\n".join(
            [
                "model_references:",
                "  system: foo=bar",
            ]
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        show_check_summary(home=home)
    finally:
        os.chdir(cwd)

    output = " ".join(capsys.readouterr().out.split())
    assert "Effective Config Errors" in output
    assert "model_references" in output
    assert "Input should be a valid dictionary" in output


def test_show_check_summary_reports_malformed_yaml_as_effective_config_error(
    tmp_path: Path,
    capsys,
) -> None:
    home = tmp_path / ".fast-agent"
    home.mkdir(parents=True)

    (tmp_path / "fastagent.config.yaml").write_text("logger:\n  level: warning\n", encoding="utf-8")
    (home / "fastagent.config.yaml").write_text(
        "\n".join(
            [
                "model_references:",
                "  system.code=codexplan?transport=ws",
            ]
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        show_check_summary(home=home)
    finally:
        os.chdir(cwd)

    output = " ".join(capsys.readouterr().out.split())
    assert "Config File Issues" in output
    assert "Fix the YAML syntax errors in your configuration files" in output


def test_show_check_summary_resolves_relative_home_from_cwd(
    tmp_path: Path,
    capsys,
) -> None:
    home = tmp_path / ".fast-agent-alt"
    (home / "agent-cards").mkdir(parents=True)
    (home / "agent-cards" / "demo_agent.yaml").write_text(
        "\n".join(
            [
                "name: demo_agent",
                "model: gpt-4.1-mini",
            ]
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        show_check_summary(home=Path(".fast-agent-alt"))
    finally:
        os.chdir(cwd)

    output = capsys.readouterr().out
    assert "demo_agent" in output


def test_show_model_secret_requirements_plain_output_includes_safety_instruction(capsys) -> None:
    show_model_secret_requirements("sonnet")

    output = capsys.readouterr().out
    assert "Model secret requirements" in output
    assert "ANTHROPIC_API_KEY" in output
    assert "IMPORTANT:" in output
    assert "Never pass secret values" in output


def test_show_model_secret_requirements_json_output_lists_candidate_env_vars(capsys) -> None:
    show_model_secret_requirements("sonnet,kimi", json_output=True)

    output = capsys.readouterr().out
    assert '"candidate_secret_env_vars"' in output
    assert "ANTHROPIC_API_KEY" in output
    assert "HF_TOKEN" in output
    assert '"safety_rule"' in output
