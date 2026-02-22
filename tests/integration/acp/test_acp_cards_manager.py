"""Integration tests for ACP /cards manager commands."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.config import get_settings


@dataclass
class StubAgent:
    name: str
    message_history: list[Any] = field(default_factory=list)
    llm: Any = None


@dataclass
class StubAgentInstance:
    agents: dict[str, Any] = field(default_factory=dict)


def _handler(instance: StubAgentInstance, agent_name: str) -> SlashCommandHandler:
    return SlashCommandHandler(
        "test-session",
        cast("Any", instance),
        agent_name,
    )


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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cards_add_and_remove(tmp_path: Path) -> None:
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
        f"  marketplace_url: '{marketplace_path.as_posix()}'\n",
        encoding="utf-8",
    )

    get_settings(config_path=str(config_path))
    try:
        agent = StubAgent(name="test-agent")
        instance = StubAgentInstance(agents={"test-agent": agent})
        handler = _handler(instance, "test-agent")

        add_response = await handler.execute_command("cards", "add alpha")
        assert "Installed card pack: alpha" in add_response
        assert (env_root / "agent-cards" / "alpha.md").exists()

        list_response = await handler.execute_command("cards", "")
        assert "alpha" in list_response

        remove_response = await handler.execute_command("cards", "remove alpha")
        assert "Removed card pack: alpha" in remove_response
        assert not (env_root / "agent-cards" / "alpha.md").exists()
    finally:
        get_settings(config_path=str(Path(__file__).parent / "fastagent.config.yaml"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cards_registry_numbered_selection(tmp_path: Path) -> None:
    marketplace1 = tmp_path / "marketplace1.json"
    marketplace1.write_text("{\"entries\": []}", encoding="utf-8")
    marketplace2 = tmp_path / "marketplace2.json"
    marketplace2.write_text("{\"entries\": []}", encoding="utf-8")

    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        "cards:\n"
        "  marketplace_urls:\n"
        f"    - '{marketplace1.as_posix()}'\n"
        f"    - '{marketplace2.as_posix()}'\n",
        encoding="utf-8",
    )

    get_settings(config_path=str(config_path))
    try:
        agent = StubAgent(name="test-agent")
        instance = StubAgentInstance(agents={"test-agent": agent})
        handler = _handler(instance, "test-agent")

        registry_list = await handler.execute_command("cards", "registry")
        assert "[ 1]" in registry_list
        assert "[ 2]" in registry_list

        set_response = await handler.execute_command("cards", "registry 2")
        assert "Registry set to" in set_response
        assert get_settings().cards.marketplace_url == marketplace2.as_posix()

        invalid = await handler.execute_command("cards", "registry 99")
        assert "Invalid registry number" in invalid
    finally:
        get_settings(config_path=str(Path(__file__).parent / "fastagent.config.yaml"))
