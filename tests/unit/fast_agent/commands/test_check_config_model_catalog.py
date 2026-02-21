import os
from pathlib import Path

from fast_agent.cli.commands.check_config import show_check_summary, show_model_catalog_summary


def test_show_check_summary_curated_mode_hides_all_models(tmp_path: Path, capsys) -> None:
    env_dir = tmp_path / ".fast-agent"
    env_dir.mkdir(parents=True)

    original_openai_key = os.environ.get("OPENAI_API_KEY")
    cwd = Path.cwd()
    try:
        os.environ["OPENAI_API_KEY"] = "sk-openai-env"
        os.chdir(tmp_path)
        show_check_summary(env_dir=env_dir, model_catalog="curated")
    finally:
        os.chdir(cwd)
        if original_openai_key is not None:
            os.environ["OPENAI_API_KEY"] = original_openai_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    output = capsys.readouterr().out
    assert "Current Model Suggestions" in output
    assert "All Known Models (" not in output


def test_show_check_summary_all_mode_hides_curated_models(tmp_path: Path, capsys) -> None:
    env_dir = tmp_path / ".fast-agent"
    env_dir.mkdir(parents=True)

    original_openai_key = os.environ.get("OPENAI_API_KEY")
    cwd = Path.cwd()
    try:
        os.environ["OPENAI_API_KEY"] = "sk-openai-env"
        os.chdir(tmp_path)
        show_check_summary(env_dir=env_dir, model_catalog="all")
    finally:
        os.chdir(cwd)
        if original_openai_key is not None:
            os.environ["OPENAI_API_KEY"] = original_openai_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    output = capsys.readouterr().out
    assert "All Known Models (" in output
    assert "Current Model Suggestions" not in output


def test_show_model_catalog_summary_hides_standard_sections(tmp_path: Path, capsys) -> None:
    env_dir = tmp_path / ".fast-agent"
    env_dir.mkdir(parents=True)

    original_openai_key = os.environ.get("OPENAI_API_KEY")
    cwd = Path.cwd()
    try:
        os.environ["OPENAI_API_KEY"] = "sk-openai-env"
        os.chdir(tmp_path)
        show_model_catalog_summary(env_dir=env_dir, model_catalog="all")
    finally:
        os.chdir(cwd)
        if original_openai_key is not None:
            os.environ["OPENAI_API_KEY"] = original_openai_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    output = capsys.readouterr().out
    assert "All Known Models (" in output
    assert "API Keys" not in output
    assert "Application Settings" not in output
