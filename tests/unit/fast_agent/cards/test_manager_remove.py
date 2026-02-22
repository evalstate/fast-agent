from __future__ import annotations

import subprocess

from fast_agent.cards import manager
from fast_agent.paths import resolve_environment_paths


def _git(repo, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(repo) -> None:
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "Test User")


def _commit_all(repo, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _write_pack(repo) -> None:
    pack_root = repo / "packs" / "alpha"
    (pack_root / "agent-cards").mkdir(parents=True, exist_ok=True)
    (pack_root / "shared").mkdir(parents=True, exist_ok=True)
    (pack_root / "agent-cards" / "alpha.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\nhello\n",
        encoding="utf-8",
    )
    (pack_root / "shared" / "helper.txt").write_text("helper\n", encoding="utf-8")
    (pack_root / "card-pack.yaml").write_text(
        "schema_version: 1\n"
        "name: alpha\n"
        "kind: bundle\n"
        "install:\n"
        "  agent_cards: ['agent-cards/alpha.md']\n"
        "  tool_cards: []\n"
        "  files:\n"
        "    - 'shared/helper.txt'\n",
        encoding="utf-8",
    )


def test_remove_only_owned_files_preserves_unmanaged(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo)
    _commit_all(repo, "initial")

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    manager._install_marketplace_card_pack_sync(
        manager.MarketplaceCardPack(
            name="alpha",
            description="test",
            kind="bundle",
            repo_url=str(repo),
            repo_ref=None,
            repo_path="packs/alpha",
            source_url=None,
        ),
        env_paths,
        False,
        False,
        None,
    )

    unmanaged = env_paths.root / "shared" / "keep.txt"
    unmanaged.parent.mkdir(parents=True, exist_ok=True)
    unmanaged.write_text("keep\n", encoding="utf-8")

    removal = manager.remove_local_card_pack("alpha", environment_paths=env_paths)
    assert "agent-cards/alpha.md" in removal.removed_paths
    assert "shared/helper.txt" in removal.removed_paths

    assert not (env_paths.agent_cards / "alpha.md").exists()
    assert not (env_paths.root / "shared" / "helper.txt").exists()
    assert unmanaged.exists()
    assert not (env_paths.card_packs / "alpha").exists()
