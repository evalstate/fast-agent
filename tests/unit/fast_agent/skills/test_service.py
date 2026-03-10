from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

from fast_agent.skills.service import (
    apply_updates,
    check_updates,
    install_skill_sync,
    list_installed_skills,
    remove_skill,
    scan_marketplace_sync,
)

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


def _write_skill(repo: Path, *, name: str, body: str) -> None:
    skill_dir = repo / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _write_registry(repo: Path, registry_path: Path, *, skill_name: str = "alpha") -> None:
    registry_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": skill_name,
                        "description": "Test skill",
                        "repo_url": repo.as_posix(),
                        "repo_path": f"skills/{skill_name}",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_service_scan_install_list_and_remove(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo, name="alpha", body="Version 1")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")

    registry_path = tmp_path / "marketplace.json"
    _write_registry(repo, registry_path)

    scan_result = scan_marketplace_sync(registry_path.as_posix())
    assert scan_result.source == registry_path.as_posix()
    assert [skill.name for skill in scan_result.skills] == ["alpha"]

    managed_root = tmp_path / "managed"
    installed = install_skill_sync(
        registry_path.as_posix(),
        "alpha",
        destination_root=managed_root,
    )

    assert installed.name == "alpha"
    assert installed.provenance.status == "managed"

    listed = list_installed_skills(managed_root)
    assert [record.name for record in listed] == ["alpha"]

    removed = remove_skill(managed_root, "alpha")
    assert removed.name == "alpha"
    assert not removed.skill_dir.exists()


def test_service_check_and_apply_updates(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write_skill(repo, name="alpha", body="Version 1")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")

    registry_path = tmp_path / "marketplace.json"
    _write_registry(repo, registry_path)
    managed_root = tmp_path / "managed"
    install_skill_sync(registry_path.as_posix(), "alpha", destination_root=managed_root)

    _write_skill(repo, name="alpha", body="Version 2")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "update")

    updates = check_updates(managed_root)
    assert len(updates) == 1
    assert updates[0].status == "update_available"

    applied = apply_updates(managed_root, "alpha", force=False)
    assert len(applied) == 1
    assert applied[0].status == "updated"
    assert "Version 2" in (managed_root / "alpha" / "SKILL.md").read_text(encoding="utf-8")
