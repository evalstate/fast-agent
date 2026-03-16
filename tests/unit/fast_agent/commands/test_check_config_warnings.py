import os
from pathlib import Path

from fast_agent.cli.commands.check_config import show_check_summary
from fast_agent.llm.model_overlays import load_model_overlay_registry


def test_check_config_warns_missing_api_key(tmp_path: Path, capsys) -> None:
    env_dir = tmp_path / ".fast-agent"
    agent_cards_dir = env_dir / "agent-cards"
    agent_cards_dir.mkdir(parents=True)

    card_path = agent_cards_dir / "data_cleaner.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: data_cleaner",
                "model: gpt-4.1",
            ]
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    original_openai_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        os.chdir(tmp_path)
        show_check_summary(env_dir=env_dir)
    finally:
        os.chdir(cwd)
        if original_openai_key is not None:
            os.environ["OPENAI_API_KEY"] = original_openai_key

    captured = capsys.readouterr()
    normalized = " ".join(captured.out.split())
    assert (
        'Warning: Card "data_cleaner" uses model "gpt-4.1" (OpenAI) '
        "but no API key configured."
    ) in normalized


def test_check_config_does_not_warn_missing_openresponses_api_key(
    tmp_path: Path,
    capsys,
) -> None:
    env_dir = tmp_path / ".fast-agent"
    agent_cards_dir = env_dir / "agent-cards"
    agent_cards_dir.mkdir(parents=True)

    card_path = agent_cards_dir / "local_openresponses.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: local_openresponses",
                "model: openresponses.unsloth/Qwen3.5-9B-GGUF",
            ]
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    original_openresponses_key = os.environ.pop("OPENRESPONSES_API_KEY", None)
    try:
        os.chdir(tmp_path)
        show_check_summary(env_dir=env_dir)
    finally:
        os.chdir(cwd)
        if original_openresponses_key is not None:
            os.environ["OPENRESPONSES_API_KEY"] = original_openresponses_key

    captured = capsys.readouterr()
    normalized = " ".join(captured.out.split())
    assert (
        'Warning: Card "local_openresponses" uses model '
        '"openresponses.unsloth/Qwen3.5-9B-GGUF" (OpenResponses) '
        "but no API key configured."
    ) not in normalized


def test_check_config_warns_missing_openresponses_api_key_for_hosted_endpoint(
    tmp_path: Path,
    capsys,
) -> None:
    env_dir = tmp_path / ".fast-agent"
    agent_cards_dir = env_dir / "agent-cards"
    agent_cards_dir.mkdir(parents=True)
    (env_dir / "fastagent.config.yaml").write_text(
        "\n".join(
            [
                "openresponses:",
                "  base_url: https://gateway.example/v1",
            ]
        ),
        encoding="utf-8",
    )

    card_path = agent_cards_dir / "remote_openresponses.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: remote_openresponses",
                "model: openresponses.unsloth/Qwen3.5-9B-GGUF",
            ]
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    original_openresponses_key = os.environ.pop("OPENRESPONSES_API_KEY", None)
    original_openresponses_base_url = os.environ.pop("OPENRESPONSES_BASE_URL", None)
    try:
        os.chdir(tmp_path)
        show_check_summary(env_dir=env_dir)
    finally:
        os.chdir(cwd)
        if original_openresponses_key is not None:
            os.environ["OPENRESPONSES_API_KEY"] = original_openresponses_key
        if original_openresponses_base_url is not None:
            os.environ["OPENRESPONSES_BASE_URL"] = original_openresponses_base_url

    captured = capsys.readouterr()
    normalized = " ".join(captured.out.split())
    assert (
        'Warning: Card "remote_openresponses" uses model '
        '"openresponses.unsloth/Qwen3.5-9B-GGUF" (OpenResponses) '
        "but no API key configured."
    ) in normalized


def test_check_config_reports_overlay_preset_collision_as_info(
    tmp_path: Path,
    capsys,
) -> None:
    env_dir = tmp_path / ".fast-agent"
    overlays_dir = env_dir / "model-overlays"
    overlays_dir.mkdir(parents=True)
    (overlays_dir / "sonnet.yaml").write_text(
        "\n".join(
            [
                "name: sonnet",
                "provider: openresponses",
                "model: local/sonnet",
                "connection:",
                "  base_url: http://localhost:8080/v1",
                "  auth: none",
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
        empty_env_dir = tmp_path / ".empty-fast-agent"
        empty_env_dir.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, env_dir=empty_env_dir)

    captured = capsys.readouterr()
    normalized = " ".join(captured.out.split())
    assert (
        'Info: Local model overlay "sonnet" overrides existing '
        'built-in model preset "sonnet".'
    ) in normalized
