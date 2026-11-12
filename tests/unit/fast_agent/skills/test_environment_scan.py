"""Contract tests for skill discovery through an environment filesystem."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import pytest

from fast_agent.skills.environment_scan import scan_environment_skills
from fast_agent.skills.registry import SkillRegistry
from fast_agent.tools.local_shell_executor import LocalEnvironment

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.tools.execution_environment import EnvironmentFileEntry


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


class DelayedFilesystem(LocalEnvironment):
    """Real disk storage with cooperative read latency and reversed listings."""

    active_reads = 0
    peak_reads = 0
    completed_reads = 0

    async def list_dir(self, path: str) -> list[EnvironmentFileEntry]:
        return sorted(await super().list_dir(path), key=lambda entry: entry.path, reverse=True)

    async def read_text(self, path: str) -> str:
        self.active_reads += 1
        self.peak_reads = max(self.peak_reads, self.active_reads)
        try:
            await asyncio.sleep(0)
            return await super().read_text(path)
        finally:
            self.active_reads -= 1
            self.completed_reads += 1


@pytest.mark.asyncio
async def test_scan_bounds_concurrency_and_preserves_registry_order(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    for index in range(21):
        _write_skill(root, f"skill-{index:02d}")
    # Two different folders advertise the same skill. The lexical last wins,
    # regardless of backend listing order or concurrent completion order.
    duplicate = _write_skill(root, "zz-duplicate")
    duplicate.write_text("---\nname: skill-00\ndescription: Override\n---\n", encoding="utf-8")
    environment = DelayedFilesystem(
        logger=logging.getLogger("test-env-scan"), working_directory=tmp_path
    )

    manifests, warnings = await scan_environment_skills(environment, directories=["skills"])
    registry = SkillRegistry(base_dir=tmp_path, directories=[root])

    assert manifests == registry.load_manifests()
    assert warnings == registry.warnings
    assert manifests[-1].path == duplicate
    assert environment.completed_reads == 22
    assert environment.active_reads == 0
    assert 1 < environment.peak_reads <= 8


@pytest.mark.asyncio
async def test_scan_read_failure_does_not_discard_other_manifests(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    good = _write_skill(root, "good")
    (root / "missing").mkdir()
    (root / "unreadable" / "SKILL.md").mkdir(parents=True)

    manifests, warnings = await scan_environment_skills(
        _environment(tmp_path), directories=["skills"]
    )

    assert [manifest.path for manifest in manifests] == [good]
    assert len(warnings) == 1
    assert "Failed to read skill manifest" in warnings[0]
    assert "unreadable/SKILL.md" in warnings[0]
