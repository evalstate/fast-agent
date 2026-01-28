import os
from pathlib import Path

from fast_agent.cli.commands.check_config import show_check_summary


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
