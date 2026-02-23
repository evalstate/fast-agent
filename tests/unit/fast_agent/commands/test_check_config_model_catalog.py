import os
from pathlib import Path

import pytest

from fast_agent.cli.commands.check_config import (
    show_check_summary,
    show_models_overview,
    show_provider_model_catalog,
)


def test_show_check_summary_points_to_check_models_command(tmp_path: Path, capsys) -> None:
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
    assert "Current Model Suggestions" not in output
    assert "Use fast-agent check models" in output


def test_show_provider_model_catalog_openai_defaults_to_curated_aliases(capsys) -> None:
    show_provider_model_catalog("openai")

    output = capsys.readouterr().out
    assert "OpenAI model catalog (curated)" in output
    assert "Tags" in output
    assert "fast" in output
    assert "OpenAI" in output
    assert "Responses" in output
    assert "Codex Responses" in output
    assert "gpt-4.1-mini" in output
    assert "openai.gpt-4.1-mini" in output
    assert "gpt-5-mini" in output
    assert "responses.gpt-5-mini" in output
    assert "codexspark" in output
    assert "More models are available" in output


def test_show_provider_model_catalog_openai_all_includes_openai_family(capsys) -> None:
    show_provider_model_catalog("openai", show_all=True)

    output = capsys.readouterr().out
    assert "OpenAI model catalog (curated + all models)" in output
    assert "All known models" in output
    assert "catalog" in output
    assert "gpt-4.1" in output
    assert "o1" in output
    assert "gpt-5.3-codex-spark" in output


def test_show_models_overview_includes_provider_args_and_named_aliases(
    tmp_path: Path,
    capsys,
) -> None:
    env_dir = tmp_path / ".fast-agent"
    env_dir.mkdir(parents=True)
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_aliases:",
                "  system:",
                "    fast: responses.gpt-5-mini?reasoning=low",
            ]
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        show_models_overview(env_dir=env_dir)
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


def test_show_provider_model_catalog_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        show_provider_model_catalog("not-a-provider")


def test_show_check_summary_reports_invalid_effective_model_aliases(
    tmp_path: Path,
    capsys,
) -> None:
    env_dir = tmp_path / ".fast-agent"
    env_dir.mkdir(parents=True)

    (tmp_path / "fastagent.config.yaml").write_text("logger:\n  level: warning\n", encoding="utf-8")
    (env_dir / "fastagent.config.yaml").write_text(
        "\n".join(
            [
                "model_aliases:",
                "  system: foo=bar",
            ]
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        show_check_summary(env_dir=env_dir)
    finally:
        os.chdir(cwd)

    output = " ".join(capsys.readouterr().out.split())
    assert "Effective Config Errors" in output
    assert "model_aliases" in output
    assert "Input should be a valid dictionary" in output


def test_show_check_summary_reports_malformed_yaml_as_effective_config_error(
    tmp_path: Path,
    capsys,
) -> None:
    env_dir = tmp_path / ".fast-agent"
    env_dir.mkdir(parents=True)

    (tmp_path / "fastagent.config.yaml").write_text("logger:\n  level: warning\n", encoding="utf-8")
    (env_dir / "fastagent.config.yaml").write_text(
        "\n".join(
            [
                "model_aliases:",
                "  system.code=codexplan?transport=ws",
            ]
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        show_check_summary(env_dir=env_dir)
    finally:
        os.chdir(cwd)

    output = " ".join(capsys.readouterr().out.split())
    assert "Config File Issues" in output
    assert "Fix the YAML syntax errors in your configuration files" in output


def test_show_check_summary_resolves_relative_env_dir_from_cwd(
    tmp_path: Path,
    capsys,
) -> None:
    env_dir = tmp_path / ".fast-agent-alt"
    (env_dir / "agent-cards").mkdir(parents=True)
    (env_dir / "agent-cards" / "demo_agent.yaml").write_text(
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
        show_check_summary(env_dir=Path(".fast-agent-alt"))
    finally:
        os.chdir(cwd)

    output = capsys.readouterr().out
    assert "demo_agent" in output
