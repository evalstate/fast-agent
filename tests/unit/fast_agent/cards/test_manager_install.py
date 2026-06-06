from __future__ import annotations

import subprocess

import pytest
import yaml

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


def _write_pack(repo, *, pack_subdir: str, pack_name: str, files: list[str]) -> None:
    pack_root = repo / pack_subdir
    (pack_root / "agent-cards").mkdir(parents=True, exist_ok=True)
    (pack_root / "tool-cards").mkdir(parents=True, exist_ok=True)
    (pack_root / "shared").mkdir(parents=True, exist_ok=True)
    (pack_root / "agent-cards" / f"{pack_name}.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\nhello\n",
        encoding="utf-8",
    )
    (pack_root / "tool-cards" / f"{pack_name}-tool.md").write_text(
        "---\nname: tool\nmodel: passthrough\n---\n\nhello\n",
        encoding="utf-8",
    )
    (pack_root / "shared" / "helper.txt").write_text("helper\n", encoding="utf-8")

    manifest_lines = [
        "schema_version: 1",
        f"name: {pack_name}",
        "kind: bundle",
        "install:",
        f"  agent_cards: ['agent-cards/{pack_name}.md']",
        f"  tool_cards: ['tool-cards/{pack_name}-tool.md']",
        "  files:",
    ]
    manifest_lines.extend(f"    - '{entry}'" for entry in files)

    (pack_root / "card-pack.yaml").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


def _pack(
    repo,
    *,
    name: str,
    path: str,
    repo_ref: str | None = None,
) -> manager.MarketplaceCardPack:
    return manager.MarketplaceCardPack(
        name=name,
        description="test pack",
        kind="bundle",
        repo_url=str(repo),
        repo_ref=repo_ref,
        repo_path=path,
        source_url=None,
    )


def test_install_copies_expected_files_and_writes_sidecar(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, pack_subdir="packs/alpha", pack_name="alpha", files=["shared/helper.txt"])
    _commit_all(repo, "initial")

    env_root = tmp_path / ".fast-agent"
    env_paths = resolve_environment_paths(override=env_root, cwd=tmp_path)

    result = manager._install_marketplace_card_pack_sync(
        _pack(repo, name="alpha", path="packs/alpha"),
        env_paths,
        False,
        False,
        None,
    )

    assert (env_paths.agent_cards / "alpha.md").exists()
    assert (env_paths.tool_cards / "alpha-tool.md").exists()
    assert (env_paths.root / "shared" / "helper.txt").exists()
    assert result.source.installed_files == (
        "agent-cards/alpha.md",
        "shared/helper.txt",
        "tool-cards/alpha-tool.md",
    )

    source, error = manager.read_installed_card_pack_source(result.pack_dir)
    assert error is None
    assert source is not None
    assert source.name == "alpha"
    assert source.kind == "bundle"


@pytest.mark.parametrize("install_mode", ["repo_ref", "pinned_revision"])
def test_install_local_card_pack_ref_copies_committed_content(
    tmp_path,
    install_mode: str,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, pack_subdir="packs/alpha", pack_name="alpha", files=["shared/helper.txt"])
    first_commit = _commit_all(repo, "initial")

    (repo / "packs" / "alpha" / "agent-cards" / "alpha.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\nworking tree change\n",
        encoding="utf-8",
    )

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    pack = _pack(repo, name="alpha", path="packs/alpha")
    pinned_revision = None
    if install_mode == "repo_ref":
        pack = _pack(repo, name="alpha", path="packs/alpha", repo_ref=first_commit)
    else:
        pinned_revision = first_commit

    result = manager._install_marketplace_card_pack_sync(
        pack,
        env_paths,
        False,
        False,
        pinned_revision,
    )

    installed_card = (env_paths.agent_cards / "alpha.md").read_text(encoding="utf-8")
    assert "hello" in installed_card
    assert "working tree change" not in installed_card
    assert result.source.installed_commit == first_commit


def test_install_dirty_local_card_pack_records_local_revision(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, pack_subdir="packs/alpha", pack_name="alpha", files=["shared/helper.txt"])
    _commit_all(repo, "initial")

    (repo / "packs" / "alpha" / "agent-cards" / "alpha.md").write_text(
        "---\nname: alpha\nmodel: passthrough\n---\n\ndirty content\n",
        encoding="utf-8",
    )

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    result = manager._install_marketplace_card_pack_sync(
        _pack(repo, name="alpha", path="packs/alpha"),
        env_paths,
        False,
        False,
        None,
    )

    installed_card = (env_paths.agent_cards / "alpha.md").read_text(encoding="utf-8")
    assert "dirty content" in installed_card
    assert result.source.installed_revision == manager.LOCAL_REVISION
    assert result.source.installed_commit is None
    assert result.source.installed_path_oid is None


def test_format_missing_installed_files_detail_uses_singular_label() -> None:
    assert manager._format_missing_installed_files_detail(["agent-cards/alpha.md"]) == (
        "missing installed file in environment: agent-cards/alpha.md"
    )


def test_format_missing_installed_files_detail_uses_plural_label_and_preview_limit() -> None:
    assert (
        manager._format_missing_installed_files_detail(
            [
                "one.md",
                "two.md",
                "three.md",
                "four.md",
            ]
        )
        == "missing installed files in environment: one.md, two.md, three.md, ..."
    )


def test_install_rejects_manifest_path_traversal(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, pack_subdir="packs/invalid", pack_name="invalid", files=["../escape.txt"])
    _commit_all(repo, "initial")

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    with pytest.raises(ValueError, match="Invalid install path"):
        manager._install_marketplace_card_pack_sync(
            _pack(repo, name="invalid", path="packs/invalid"),
            env_paths,
            False,
            False,
            None,
        )


def test_validate_pack_source_dir_rejects_manifest_directory(tmp_path) -> None:
    source_dir = tmp_path / "pack"
    (source_dir / "card-pack.yaml").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="card-pack.yaml not found"):
        manager._validate_pack_source_dir(source_dir, "packs/alpha")


def test_install_detects_ownership_conflicts(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, pack_subdir="packs/one", pack_name="one", files=["shared/common.txt"])
    (repo / "packs" / "one" / "shared" / "common.txt").write_text("one\n", encoding="utf-8")
    _write_pack(repo, pack_subdir="packs/two", pack_name="two", files=["shared/common.txt"])
    (repo / "packs" / "two" / "shared" / "common.txt").write_text("two\n", encoding="utf-8")
    _commit_all(repo, "initial")

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    manager._install_marketplace_card_pack_sync(
        _pack(repo, name="one", path="packs/one"),
        env_paths,
        False,
        False,
        None,
    )

    with pytest.raises(manager.OwnershipConflictError, match="owned by another pack"):
        manager._install_marketplace_card_pack_sync(
            _pack(repo, name="two", path="packs/two"),
            env_paths,
            False,
            False,
            None,
        )


def test_install_maps_legacy_pack_config_to_preferred_filename(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, pack_subdir="packs/codex", pack_name="codex", files=["fastagent.config.yaml"])
    (repo / "packs" / "codex" / "fastagent.config.yaml").write_text(
        'default_model: "$system.default"\n',
        encoding="utf-8",
    )
    _commit_all(repo, "initial")

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)

    result = manager._install_marketplace_card_pack_sync(
        _pack(repo, name="codex", path="packs/codex"),
        env_paths,
        False,
        False,
        None,
    )

    assert (env_paths.root / "fast-agent.yaml").exists()
    assert not (env_paths.root / "fastagent.config.yaml").exists()
    assert "fast-agent.yaml" in result.source.installed_files
    assert "fastagent.config.yaml" not in result.source.installed_files


def test_install_legacy_pack_config_merges_existing_preferred_config(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, pack_subdir="packs/codex", pack_name="codex", files=["fastagent.config.yaml"])
    (repo / "packs" / "codex" / "fastagent.config.yaml").write_text(
        'default_model: "$system.default"\n'
        "model_references:\n"
        "  system:\n"
        "    fast: codexspark\n"
        "    last_used: codexplan\n",
        encoding="utf-8",
    )
    _commit_all(repo, "initial")

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    env_paths.root.mkdir(parents=True, exist_ok=True)
    (env_paths.root / "fast-agent.yaml").write_text(
        "model_references:\n  system:\n    last_used: gpt-4.1-mini\n",
        encoding="utf-8",
    )

    manager._install_marketplace_card_pack_sync(
        _pack(repo, name="codex", path="packs/codex"),
        env_paths,
        False,
        False,
        None,
    )

    assert not (env_paths.root / "fastagent.config.yaml").exists()
    with open(env_paths.root / "fast-agent.yaml", "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["default_model"] == "$system.default"
    assert saved["model_references"]["system"]["fast"] == "codexspark"
    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"


def test_install_merges_unmanaged_env_config_when_it_only_preserves_last_used(tmp_path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, pack_subdir="packs/codex", pack_name="codex", files=["fastagent.config.yaml"])
    (repo / "packs" / "codex" / "fastagent.config.yaml").write_text(
        'default_model: "$system.default"\n'
        "model_references:\n"
        "  system:\n"
        "    fast: codexspark\n"
        "    last_used: codexplan\n"
        "mcp:\n"
        "  targets:\n"
        "    - name: openai\n"
        "      target: https://developers.openai.com/mcp\n",
        encoding="utf-8",
    )
    _commit_all(repo, "initial")

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    env_paths.root.mkdir(parents=True, exist_ok=True)
    (env_paths.root / "fastagent.config.yaml").write_text(
        "model_references:\n  system:\n    last_used: gpt-4.1-mini\n",
        encoding="utf-8",
    )

    manager._install_marketplace_card_pack_sync(
        _pack(repo, name="codex", path="packs/codex"),
        env_paths,
        False,
        False,
        None,
    )

    with open(env_paths.root / "fastagent.config.yaml", "r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)

    assert saved["default_model"] == "$system.default"
    assert saved["model_references"]["system"]["fast"] == "codexspark"
    assert saved["model_references"]["system"]["last_used"] == "gpt-4.1-mini"
    assert saved["mcp"]["targets"][0]["name"] == "openai"


def test_install_rejects_unmanaged_env_config_when_it_contains_more_than_last_used(
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_pack(repo, pack_subdir="packs/codex", pack_name="codex", files=["fastagent.config.yaml"])
    (repo / "packs" / "codex" / "fastagent.config.yaml").write_text(
        'default_model: "$system.default"\n',
        encoding="utf-8",
    )
    _commit_all(repo, "initial")

    env_paths = resolve_environment_paths(override=tmp_path / ".fast-agent", cwd=tmp_path)
    env_paths.root.mkdir(parents=True, exist_ok=True)
    (env_paths.root / "fastagent.config.yaml").write_text(
        "default_model: keep-me\nmodel_references:\n  system:\n    last_used: gpt-4.1-mini\n",
        encoding="utf-8",
    )

    with pytest.raises(
        manager.OwnershipConflictError,
        match="fastagent.config.yaml already exists and is unmanaged",
    ):
        manager._install_marketplace_card_pack_sync(
            _pack(repo, name="codex", path="packs/codex"),
            env_paths,
            False,
            False,
            None,
        )
