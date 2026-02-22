from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

from click.utils import strip_ansi
from typer.testing import CliRunner

from fast_agent.cli.commands import cards as cards_command
from fast_agent.cli.main import LAZY_SUBCOMMANDS
from fast_agent.config import get_settings, update_global_settings

if TYPE_CHECKING:
    from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "Test User")


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)


def test_cards_lazy_subcommand_registered() -> None:
    assert LAZY_SUBCOMMANDS["cards"] == "fast_agent.cli.commands.cards:app"


def test_cards_add_and_remove_via_cli(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    pack_root = repo / "packs" / "alpha"
    (pack_root / "agent-cards").mkdir(parents=True)
    (pack_root / "agent-cards" / "alpha.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\nhello\n",
        encoding="utf-8",
    )
    (pack_root / "card-pack.yaml").write_text(
        "schema_version: 1\n"
        "name: alpha\n"
        "kind: card\n"
        "install:\n"
        "  agent_cards: ['agent-cards/alpha.md']\n"
        "  tool_cards: []\n"
        "  files: []\n",
        encoding="utf-8",
    )
    _commit_all(repo, "initial")

    marketplace_path = tmp_path / "marketplace.json"
    marketplace_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "alpha",
                        "kind": "card",
                        "repo_url": repo.as_posix(),
                        "repo_path": "packs/alpha",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        f"environment_dir: '{env_root.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        runner = CliRunner()

        add_result = runner.invoke(
            cards_command.app,
            ["--registry", marketplace_path.as_posix(), "add", "alpha"],
        )
        assert add_result.exit_code == 0, add_result.output
        assert "Installed card pack: alpha" in add_result.output
        assert (env_root / "agent-cards" / "alpha.md").exists()

        list_result = runner.invoke(cards_command.app, ["list"])
        assert list_result.exit_code == 0, list_result.output
        assert "alpha" in list_result.output

        remove_result = runner.invoke(cards_command.app, ["remove", "alpha"])
        assert remove_result.exit_code == 0, remove_result.output
        assert "Removed card pack: alpha" in remove_result.output
        assert not (env_root / "agent-cards" / "alpha.md").exists()
    finally:
        update_global_settings(old_settings)


def test_cards_help_has_registry_option_no_registry_subcommand() -> None:
    runner = CliRunner()
    result = runner.invoke(cards_command.app, ["--help"])
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "--registry" in output
    assert "│ registry" not in output


def test_top_level_env_flag_routes_to_cards_subcommand(tmp_path: Path) -> None:
    env_root = tmp_path / "custom-env"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fast_agent.cli",
            "--env",
            str(env_root),
            "cards",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "cards [OPTIONS] COMMAND" in result.stdout


def test_cards_add_uses_configured_marketplace_urls_by_default(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    pack_root = repo / "packs" / "alpha"
    (pack_root / "agent-cards").mkdir(parents=True)
    (pack_root / "agent-cards" / "alpha.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\nhello\n",
        encoding="utf-8",
    )
    (pack_root / "card-pack.yaml").write_text(
        "schema_version: 1\n"
        "name: alpha\n"
        "kind: card\n"
        "install:\n"
        "  agent_cards: ['agent-cards/alpha.md']\n"
        "  tool_cards: []\n"
        "  files: []\n",
        encoding="utf-8",
    )
    _commit_all(repo, "initial")

    marketplace_path = tmp_path / "marketplace.json"
    marketplace_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "alpha",
                        "kind": "card",
                        "repo_url": repo.as_posix(),
                        "repo_path": "packs/alpha",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        f"environment_dir: '{env_root.as_posix()}'\n"
        "cards:\n"
        "  marketplace_urls:\n"
        f"    - '{marketplace_path.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        runner = CliRunner()
        add_result = runner.invoke(cards_command.app, ["add", "alpha"])

        assert add_result.exit_code == 0, add_result.output
        assert "Installed card pack: alpha" in add_result.output
        assert (env_root / "agent-cards" / "alpha.md").exists()
    finally:
        update_global_settings(old_settings)


def test_cards_publish_no_push_commits_locally(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    pack_root = repo / "packs" / "alpha"
    (pack_root / "agent-cards").mkdir(parents=True)
    (pack_root / "agent-cards" / "alpha.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\nhello\n",
        encoding="utf-8",
    )
    (pack_root / "card-pack.yaml").write_text(
        "schema_version: 1\n"
        "name: alpha\n"
        "kind: card\n"
        "install:\n"
        "  agent_cards: ['agent-cards/alpha.md']\n"
        "  tool_cards: []\n"
        "  files: []\n",
        encoding="utf-8",
    )
    _commit_all(repo, "initial")

    marketplace_path = tmp_path / "marketplace.json"
    marketplace_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "alpha",
                        "kind": "card",
                        "repo_url": repo.as_posix(),
                        "repo_path": "packs/alpha",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    env_root = tmp_path / ".fast-agent"
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        f"environment_dir: '{env_root.as_posix()}'\n",
        encoding="utf-8",
    )

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        runner = CliRunner()
        add_result = runner.invoke(
            cards_command.app,
            ["--registry", marketplace_path.as_posix(), "add", "alpha"],
        )
        assert add_result.exit_code == 0, add_result.output

        installed_card = env_root / "agent-cards" / "alpha.md"
        installed_card.write_text(
            installed_card.read_text(encoding="utf-8") + "\nlocal publish edit\n",
            encoding="utf-8",
        )

        publish_result = runner.invoke(
            cards_command.app,
            ["publish", "alpha", "--no-push", "--message", "publish alpha"],
        )
        assert publish_result.exit_code == 0, publish_result.output
        assert "Status: committed" in publish_result.output
        assert "local publish edit" in (repo / "packs" / "alpha" / "agent-cards" / "alpha.md").read_text(
            encoding="utf-8"
        )
    finally:
        update_global_settings(old_settings)


def test_cards_publish_help_lists_temp_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(cards_command.app, ["publish", "--help"])
    output = strip_ansi(result.output)

    assert result.exit_code == 0, output
    assert "--temp-dir" in output
    assert "--keep-temp" in output
