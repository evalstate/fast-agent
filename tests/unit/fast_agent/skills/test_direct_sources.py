"""Direct skill source coverage.

These tests exercise the new source-adapter and service boundary only; broader
marketplace install/update behavior remains covered by existing skills tests.
"""

from __future__ import annotations

import pytest

from fast_agent.skills.direct_sources import (
    DirectSkillSourceError,
    is_direct_skill_source,
    resolve_direct_skill_source_sync,
)
from fast_agent.skills.service import install_direct_skill_sync


def test_direct_local_source_uses_manifest_name_and_copies_assets(tmp_path):
    source_dir = tmp_path / "repo-name"
    source_dir.mkdir()
    source_dir.joinpath("SKILL.md").write_text(
        "---\nname: canonical-name\ndescription: Test skill\n---\n\nBody.\n",
        encoding="utf-8",
    )
    source_dir.joinpath("references").mkdir()
    source_dir.joinpath("references", "notes.md").write_text("notes", encoding="utf-8")

    direct = resolve_direct_skill_source_sync(source_dir.as_posix())

    assert direct.skill.name == "canonical-name"
    assert direct.skill.description == "Test skill"
    assert direct.skill.install_dir_name == "canonical-name"
    assert direct.skill.repo_path == "."


@pytest.mark.parametrize(
    "name",
    [
        "Uppercase",
        "-leading",
        "trailing-",
        "has_underscore",
        "a" * 65,
    ],
)
def test_direct_source_rejects_skill_names_outside_agent_skills_spec(tmp_path, name):
    source_dir = tmp_path / "skill"
    source_dir.mkdir()
    source_dir.joinpath("SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill\n---\n\nBody.\n",
        encoding="utf-8",
    )

    with pytest.raises(DirectSkillSourceError, match="Does not meet Agent Skills specification"):
        resolve_direct_skill_source_sync(source_dir.as_posix())


def test_direct_source_rejects_missing_frontmatter(tmp_path):
    source_dir = tmp_path / "skill"
    source_dir.mkdir()
    source_dir.joinpath("SKILL.md").write_text("No frontmatter.\n", encoding="utf-8")

    with pytest.raises(DirectSkillSourceError, match="Invalid SKILL.md frontmatter"):
        resolve_direct_skill_source_sync(source_dir.as_posix())


def test_github_skill_url_is_direct_source():
    assert is_direct_skill_source(
        "https://github.com/org/repo/blob/main/skills/example/SKILL.md"
    )
    assert is_direct_skill_source("https://github.com/org/repo/tree/main/skills/example")


def test_install_direct_skill_installs_under_manifest_name_with_assets(tmp_path):
    """Service smoke: direct source -> existing installer -> canonical managed dir."""
    source_dir = tmp_path / "source-name"
    source_dir.mkdir()
    source_dir.joinpath("SKILL.md").write_text(
        "---\nname: canonical-name\ndescription: Test skill\n---\n\nBody.\n",
        encoding="utf-8",
    )
    source_dir.joinpath("assets").mkdir()
    source_dir.joinpath("assets", "example.txt").write_text("asset", encoding="utf-8")

    managed_dir = tmp_path / "managed"
    record = install_direct_skill_sync(source_dir.as_posix(), destination_root=managed_dir)

    assert record.name == "canonical-name"
    assert record.skill_dir == managed_dir / "canonical-name"
    assert (managed_dir / "canonical-name" / "SKILL.md").exists()
    assert (managed_dir / "canonical-name" / "assets" / "example.txt").read_text(
        encoding="utf-8"
    ) == "asset"
    assert not (managed_dir / "source-name").exists()
