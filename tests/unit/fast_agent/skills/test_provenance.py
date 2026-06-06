from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.skills.models import (
    SKILL_SOURCE_SCHEMA_VERSION,
    InstalledSkillSource,
    MarketplaceSkill,
)
from fast_agent.skills.provenance import (
    build_installed_skill_source,
    read_installed_skill_source,
    write_installed_skill_source,
)

if TYPE_CHECKING:
    from pathlib import Path

_VALID_SHA256_FINGERPRINT = "sha256:" + ("0" * 64)


def test_installed_skill_source_round_trip(tmp_path: Path) -> None:
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()

    source = InstalledSkillSource(
        schema_version=SKILL_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin="remote",
        repo_url="https://github.com/example/skills",
        repo_ref="main",
        repo_path="skills/alpha",
        source_url="https://github.com/example/skills/blob/main/skills/alpha/SKILL.md",
        installed_commit="abcdef1234567890",
        installed_path_oid="feedbeef",
        installed_revision="abcdef1234567890",
        installed_at="2026-03-10T12:00:00Z",
        content_fingerprint=_VALID_SHA256_FINGERPRINT,
    )

    write_installed_skill_source(skill_dir, source)
    read_result = read_installed_skill_source(skill_dir)

    assert read_result.error is None
    assert read_result.source == source


def test_installed_skill_source_accepts_repo_root_path(tmp_path: Path) -> None:
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()

    source = InstalledSkillSource(
        schema_version=SKILL_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin="local",
        repo_url="/tmp/example-skills",
        repo_ref=None,
        repo_path=".",
        source_url="/tmp/example-skills/SKILL.md",
        installed_commit=None,
        installed_path_oid=None,
        installed_revision="local",
        installed_at="2026-03-10T12:00:00Z",
        content_fingerprint=_VALID_SHA256_FINGERPRINT,
    )

    write_installed_skill_source(skill_dir, source)
    read_result = read_installed_skill_source(skill_dir)

    assert read_result.error is None
    assert read_result.source == source


def test_build_installed_skill_source_uses_local_revision_without_commit() -> None:
    source = build_installed_skill_source(
        skill=MarketplaceSkill(
            name="alpha",
            description="demo",
            repo_url="/tmp/example-skills",
            repo_ref=None,
            repo_path="skills/alpha",
        ),
        source_origin="local",
        installed_commit=None,
        installed_path_oid=None,
        fingerprint="sha256:test",
    )

    assert source.schema_version == SKILL_SOURCE_SCHEMA_VERSION
    assert source.installed_revision == "local"
    assert source.installed_at.endswith("Z")
