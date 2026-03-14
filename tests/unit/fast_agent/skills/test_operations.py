from __future__ import annotations

import asyncio
import json
import subprocess
from typing import TYPE_CHECKING

from fast_agent.skills.operations import (
    fetch_marketplace_skills_with_source,
    install_marketplace_skill_sync,
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
    source, error = read_installed_skill_source(install_dir)

    assert error is None
    assert source is not None
    assert source.source_origin == "local"
    assert (managed_root / "alpha" / "SKILL.md").exists()


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
