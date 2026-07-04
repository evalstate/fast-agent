"""Contract tests for skill discovery through an environment filesystem."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from fast_agent.skills.environment_scan import scan_environment_skills
from fast_agent.tools.local_shell_executor import LocalEnvironment

if TYPE_CHECKING:
    from pathlib import Path


def _write_skill(root: Path, name: str, *, description: str = "A skill") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nUse {name}.\n",
        encoding="utf-8",
    )
    return manifest


def _environment(tmp_path: Path) -> LocalEnvironment:
    return LocalEnvironment(logger=logging.getLogger("test-env-scan"), working_directory=tmp_path)


@pytest.mark.asyncio
async def test_scan_discovers_default_directories_relative_to_environment_cwd(
    tmp_path: Path,
) -> None:
    manifest_path = _write_skill(tmp_path / ".fast-agent" / "skills", "alpha")

    manifests, warnings = await scan_environment_skills(_environment(tmp_path))

    assert warnings == []
    assert [manifest.name for manifest in manifests] == ["alpha"]
    assert manifests[0].path == manifest_path
    assert manifests[0].body == "Use alpha."


@pytest.mark.asyncio
async def test_scan_missing_directories_are_silently_skipped(tmp_path: Path) -> None:
    manifests, warnings = await scan_environment_skills(_environment(tmp_path))

    assert manifests == []
    assert warnings == []


@pytest.mark.asyncio
async def test_scan_custom_directories_resolve_against_environment_cwd(tmp_path: Path) -> None:
    _write_skill(tmp_path / "team-skills", "beta")

    manifests, warnings = await scan_environment_skills(
        _environment(tmp_path), directories=["team-skills"]
    )

    assert warnings == []
    assert [manifest.name for manifest in manifests] == ["beta"]


@pytest.mark.asyncio
async def test_scan_later_directory_overrides_duplicate_with_warning(tmp_path: Path) -> None:
    _write_skill(tmp_path / "first", "gamma", description="First gamma")
    winning = _write_skill(tmp_path / "second", "gamma", description="Second gamma")

    manifests, warnings = await scan_environment_skills(
        _environment(tmp_path), directories=["first", "second"]
    )

    assert [manifest.path for manifest in manifests] == [winning]
    assert len(warnings) == 1
    assert "Duplicate skill 'gamma'" in warnings[0]


@pytest.mark.asyncio
async def test_scan_reports_invalid_manifest_and_continues(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill(skills_root, "good")
    broken_dir = skills_root / "broken"
    broken_dir.mkdir()
    (broken_dir / "SKILL.md").write_text("---\ndescription: no name\n---\nBody\n", encoding="utf-8")
    (skills_root / "notes.txt").write_text("not a skill", encoding="utf-8")

    manifests, warnings = await scan_environment_skills(
        _environment(tmp_path), directories=["skills"]
    )

    assert [manifest.name for manifest in manifests] == ["good"]
    assert len(warnings) == 1
    assert "broken" in warnings[0]
