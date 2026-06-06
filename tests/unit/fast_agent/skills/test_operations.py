from __future__ import annotations

import asyncio
import json
import subprocess
from typing import TYPE_CHECKING

from fast_agent.skills.models import MarketplaceSkill
from fast_agent.skills.operations import (
    _has_skill_manifest,
    _may_be_implicit_skill_bundle,
    fetch_marketplace_skills_with_source,
    install_marketplace_skill_sync,
    select_skill_by_name_or_index,
)
from fast_agent.skills.provenance import read_installed_skill_source
from fast_agent.skills.service import install_skill_sync

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


def _marketplace_skill(
    name: str,
    repo_path: str,
    *,
    install_dir_name_override: str | None = None,
) -> MarketplaceSkill:
    return MarketplaceSkill(
        name=name,
        description=None,
        repo_url="https://github.com/example/skills",
        repo_ref=None,
        repo_path=repo_path,
        install_dir_name_override=install_dir_name_override,
    )


def test_select_skill_by_name_or_index_accepts_install_dir_name_alias() -> None:
    skill = _marketplace_skill(
        "bundle-entry",
        "plugins/app/SKILL.md",
        install_dir_name_override="canonical-name",
    )

    assert select_skill_by_name_or_index([skill], "canonical-name") is skill


def test_select_skill_by_name_or_index_accepts_lowercase_manifest_parent_alias() -> None:
    skill = _marketplace_skill("bundle-entry", "plugins/app/skill.md")

    assert skill.install_dir_name == "app"
    assert select_skill_by_name_or_index([skill], "app") is skill


def test_manifest_path_checks_normalize_manifest_filename(tmp_path: Path) -> None:
    manifest = tmp_path / "Skill.MD"
    manifest.write_text("---\nname: demo\n---\n", encoding="utf-8")
    skill = _marketplace_skill("demo", "skills/demo/Skill.MD")

    assert not _may_be_implicit_skill_bundle(skill)
    assert _has_skill_manifest(manifest)


def test_select_skill_by_name_or_index_rejects_ambiguous_install_dir_name_alias() -> None:
    skills = [
        _marketplace_skill("first", "skills/first", install_dir_name_override="shared"),
        _marketplace_skill("second", "skills/second", install_dir_name_override="shared"),
    ]

    assert select_skill_by_name_or_index(skills, "shared") is None


def test_operations_scan_local_registry_and_install_into_managed_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    skill_dir = repo / "skills" / "alpha"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Test skill\n---\n\nAlpha body.\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")

    registry_path = tmp_path / "marketplace.json"
    registry_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "alpha",
                        "description": "Alpha skill",
                        "repo_url": repo.as_posix(),
                        "repo_path": "skills/alpha",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    skills, resolved_source = asyncio.run(
        fetch_marketplace_skills_with_source(registry_path.as_posix())
    )

    assert resolved_source == registry_path.as_posix()
    assert [skill.name for skill in skills] == ["alpha"]

    managed_root = tmp_path / "managed"
    install_dir = install_marketplace_skill_sync(skills[0], managed_root)
    read_result = read_installed_skill_source(install_dir)

    assert read_result.error is None
    assert read_result.source is not None
    assert read_result.source.source_origin == "local"
    assert (managed_root / "alpha" / "SKILL.md").exists()


def test_operations_expands_plugin_source_with_multiple_nested_skills(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    plugin_dir = repo / "plugins" / "mcp-apps"
    for name in ("create-mcp-app", "convert-web-app"):
        skill_dir = plugin_dir / "skills" / name
        skill_dir.mkdir(parents=True)
        skill_dir.joinpath("SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name} skill\n---\n\nBody.\n",
            encoding="utf-8",
        )

    registry_path = repo / ".claude-plugin" / "marketplace.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "plugins": [
                    {
                        "name": "mcp-apps",
                        "description": "Skills for MCP Apps development",
                        "source": "./plugins/mcp-apps",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")

    skills, resolved_source = asyncio.run(
        fetch_marketplace_skills_with_source(registry_path.as_posix())
    )

    assert resolved_source == registry_path.as_posix()
    assert [skill.name for skill in skills] == ["convert-web-app", "create-mcp-app"]
    assert [skill.repo_path for skill in skills] == [
        "plugins/mcp-apps/skills/convert-web-app",
        "plugins/mcp-apps/skills/create-mcp-app",
    ]
    assert all(skill.bundle_name == "mcp-apps" for skill in skills)

    managed_root = tmp_path / "managed"
    install_marketplace_skill_sync(skills[0], managed_root)

    assert (managed_root / "convert-web-app" / "SKILL.md").exists()


def test_operations_expands_pinned_local_plugin_source_from_ref(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    plugin_dir = repo / "plugins" / "mcp-apps"

    alpha_dir = plugin_dir / "skills" / "alpha"
    alpha_dir.mkdir(parents=True)
    alpha_dir.joinpath("SKILL.md").write_text(
        "---\nname: alpha\ndescription: Alpha skill\n---\n\nBody.\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    first_commit = _git(repo, "rev-parse", "HEAD")

    beta_dir = plugin_dir / "skills" / "beta"
    beta_dir.mkdir(parents=True)
    beta_dir.joinpath("SKILL.md").write_text(
        "---\nname: beta\ndescription: Beta skill\n---\n\nBody.\n",
        encoding="utf-8",
    )

    registry_path = repo / ".claude-plugin" / "marketplace.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "plugins": [
                    {
                        "name": "mcp-apps",
                        "description": "Skills for MCP Apps development",
                        "source": "./plugins/mcp-apps",
                        "ref": first_commit,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    skills, resolved_source = asyncio.run(
        fetch_marketplace_skills_with_source(registry_path.as_posix())
    )

    assert resolved_source == registry_path.as_posix()
    assert [skill.name for skill in skills] == ["alpha"]
    assert skills[0].repo_ref == first_commit


def test_install_skill_rolls_back_when_installed_skill_cannot_be_reloaded(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    skill_dir = repo / "skills" / "alpha"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\n---\n\nbroken\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")

    registry_path = repo / ".claude-plugin" / "marketplace.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "plugins": [
                    {
                        "name": "alpha",
                        "description": "Broken skill",
                        "source": "./skills/alpha",
                        "skills": "./",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    managed_root = tmp_path / "managed"

    try:
        install_skill_sync(registry_path.as_posix(), "alpha", destination_root=managed_root)
    except RuntimeError as exc:
        assert "Installed skill could not be reloaded" in str(exc)
    else:
        raise AssertionError("expected install to fail")

    assert not (managed_root / "alpha").exists()
