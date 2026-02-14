from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.commands.renderers.skills_markdown import render_skills_by_directory
from fast_agent.skills.manager import InstalledSkillSource, write_installed_skill_source
from fast_agent.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from pathlib import Path


def _write_skill(root: Path, name: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Test skill\n"
        "---\n\n"
        "body\n",
        encoding="utf-8",
    )
    return skill_dir


def test_render_skills_by_directory_shows_unmanaged_provenance(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill(skills_root, "alpha")

    manifests = SkillRegistry.load_directory(skills_root)
    rendered = render_skills_by_directory({skills_root: manifests}, heading="skills", cwd=tmp_path)

    assert "**Provenance:**" in rendered
    assert "unmanaged." in rendered


def test_render_skills_by_directory_shows_managed_provenance(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    skill_dir = _write_skill(skills_root, "alpha")

    write_installed_skill_source(
        skill_dir,
        InstalledSkillSource(
            schema_version=1,
            installed_via="marketplace",
            source_origin="remote",
            repo_url="https://github.com/example/skills",
            repo_ref="main",
            repo_path="skills/alpha",
            source_url="https://raw.githubusercontent.com/example/skills/main/marketplace.json",
            installed_commit="abcdef1234567890",
            installed_path_oid="def456",
            installed_revision="abcdef1234567890",
            installed_at="2026-02-13T00:00:00Z",
            content_fingerprint="sha256:deadbeef",
        ),
    )

    manifests = SkillRegistry.load_directory(skills_root)
    rendered = render_skills_by_directory({skills_root: manifests}, heading="skills", cwd=tmp_path)

    assert "https://github.com/example/skills@main (skills/alpha)" in rendered
    assert "**Installed:**" in rendered
    assert "2026-02-13 00:00:00 revision: abcdef1" in rendered
