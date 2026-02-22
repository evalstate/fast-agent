from __future__ import annotations

import subprocess
from dataclasses import replace

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


def _write_pack(repo, *, body: str) -> None:
    pack_root = repo / "packs" / "alpha"
    (pack_root / "agent-cards").mkdir(parents=True, exist_ok=True)
    (pack_root / "agent-cards" / "alpha.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\n" + body + "\n",
        encoding="utf-8",
    )
    (pack_root / "card-pack.yaml").write_text(
        "schema_version: 1\n"
        "name: alpha\n"
        "kind: card\n"
        "install:\n"
        "  agent_cards: ['agent-cards/alpha.md']\n"
        "  files: []\n"
        "  tool_cards: []\n",
        encoding="utf-8",
    )


def _pack(repo) -> manager.MarketplaceCardPack:
    return manager.MarketplaceCardPack(
        name="alpha",
        description="test",
        kind="card",
        repo_url=str(repo),
        repo_ref=None,
        repo_path="packs/alpha",
        source_url=None,
    )


def test_update_check_and_apply(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, body="v1")
    _commit_all(repo, "initial")

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    manager._install_marketplace_card_pack_sync(_pack(repo), env_paths, False, False, None)

    _write_pack(repo, body="v2")
    second_commit = _commit_all(repo, "update")

    updates = manager.check_card_pack_updates(environment_paths=env_paths)
    assert len(updates) == 1
    assert updates[0].status == "update_available"

    applied = manager.apply_card_pack_updates(updates, environment_paths=env_paths, force=False)
    assert len(applied) == 1
    assert applied[0].status == "updated"

    installed_card = env_paths.agent_cards / "alpha.md"
    assert "v2" in installed_card.read_text(encoding="utf-8")

    source, error = manager.read_installed_card_pack_source(env_paths.card_packs / "alpha")
    assert error is None
    assert source is not None
    assert source.installed_commit == second_commit


def test_update_skips_dirty_without_force_and_overwrites_with_force(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, body="v1")
    _commit_all(repo, "initial")

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    manager._install_marketplace_card_pack_sync(_pack(repo), env_paths, False, False, None)

    _write_pack(repo, body="v2")
    _commit_all(repo, "update")

    installed_card = env_paths.agent_cards / "alpha.md"
    installed_card.write_text(
        installed_card.read_text(encoding="utf-8") + "\nlocal edit\n",
        encoding="utf-8",
    )

    updates = manager.check_card_pack_updates(environment_paths=env_paths)
    assert updates[0].status == "update_available"

    skipped = manager.apply_card_pack_updates(
        [updates[0]],
        environment_paths=env_paths,
        force=False,
    )
    assert skipped[0].status == "skipped_dirty"
    assert "local edit" in installed_card.read_text(encoding="utf-8")

    forced = manager.apply_card_pack_updates(
        [updates[0]],
        environment_paths=env_paths,
        force=True,
    )
    assert forced[0].status == "updated"
    installed_text = installed_card.read_text(encoding="utf-8")
    assert "v2" in installed_text
    assert "local edit" not in installed_text


def test_publish_commits_local_changes_without_push(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, body="v1")
    _commit_all(repo, "initial")

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    install_result = manager._install_marketplace_card_pack_sync(_pack(repo), env_paths, False, False, None)

    installed_card = env_paths.agent_cards / "alpha.md"
    installed_card.write_text(
        installed_card.read_text(encoding="utf-8") + "\nlocal publish edit\n",
        encoding="utf-8",
    )

    publish_result = manager.publish_local_card_pack(
        install_result.pack_dir,
        environment_paths=env_paths,
        push=False,
        commit_message="publish alpha",
    )

    assert publish_result.status == "committed"
    assert publish_result.commit is not None
    source, error = manager.read_installed_card_pack_source(install_result.pack_dir)
    assert error is None
    assert source is not None
    assert source.installed_commit == publish_result.commit

    repo_card = repo / "packs" / "alpha" / "agent-cards" / "alpha.md"
    repo_text = repo_card.read_text(encoding="utf-8")
    assert "local publish edit" in repo_text


def test_publish_creates_patch_when_push_fails(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, body="v1")
    _commit_all(repo, "initial")

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    install_result = manager._install_marketplace_card_pack_sync(_pack(repo), env_paths, False, False, None)

    installed_card = env_paths.agent_cards / "alpha.md"
    installed_card.write_text(
        installed_card.read_text(encoding="utf-8") + "\nchange for push failure\n",
        encoding="utf-8",
    )

    publish_result = manager.publish_local_card_pack(
        install_result.pack_dir,
        environment_paths=env_paths,
        push=True,
    )

    assert publish_result.status == "publish_failed"
    assert publish_result.commit is not None
    assert publish_result.patch_path is not None
    assert publish_result.patch_path.exists()

    source, error = manager.read_installed_card_pack_source(install_result.pack_dir)
    assert error is None
    assert source is not None
    assert source.installed_commit == publish_result.commit


def test_publish_remote_clone_failure_can_retain_temp_checkout(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, body="v1")
    _commit_all(repo, "initial")

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    install_result = manager._install_marketplace_card_pack_sync(_pack(repo), env_paths, False, False, None)

    source, error = manager.read_installed_card_pack_source(install_result.pack_dir)
    assert error is None
    assert source is not None

    manager.write_installed_card_pack_source(
        install_result.pack_dir,
        replace(source, repo_url="foo://example.invalid/card-packs.git"),
    )

    temp_root = tmp_path / "publish-temp"
    publish_result = manager.publish_local_card_pack(
        install_result.pack_dir,
        environment_paths=env_paths,
        push=False,
        keep_temp=True,
        temp_dir=temp_root,
    )

    assert publish_result.status == "source_unreachable"
    assert publish_result.retained_temp_dir is not None
    assert publish_result.retained_temp_dir.exists()
    assert publish_result.retained_temp_dir.parent == temp_root
