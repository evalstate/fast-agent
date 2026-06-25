from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.commands.renderers.skills_markdown import (
    render_marketplace_skills,
    render_skills_by_directory,
    render_skills_remove_list,
)
from fast_agent.skills.models import InstalledSkillSource, MarketplaceSkill
from fast_agent.skills.provenance import write_installed_skill_source
from fast_agent.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from pathlib import Path


def _write_skill(root: Path, name: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill\n---\n\nbody\n",
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
            content_fingerprint=f"sha256:{'a' * 64}",
        ),
    )

    manifests = SkillRegistry.load_directory(skills_root)
    rendered = render_skills_by_directory({skills_root: manifests}, heading="skills", cwd=tmp_path)

    assert "https://github.com/example/skills@main (skills/alpha)" in rendered
    assert "**Installed:**" in rendered
    assert "2026-02-13 00:00:00 revision: abcdef1" in rendered


def test_render_skills_by_directory_emits_browse_guidance_once(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill(skills_root, "alpha")

    manifests = SkillRegistry.load_directory(skills_root)
    rendered = render_skills_by_directory({skills_root: manifests}, heading="skills", cwd=tmp_path)

    assert rendered.count("Use `/skills available` to browse marketplace skills.") == 1
    assert rendered.count("Search with `/skills search <query>`.") == 1


def test_render_skills_by_directory_normalizes_heading(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"

    rendered = render_skills_by_directory({skills_root: []}, heading="# skills", cwd=tmp_path)

    assert rendered.startswith("# skills")
    assert not rendered.startswith("# #")


def test_skills_markdown_renderers_escape_headings(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"

    directory_rendered = render_skills_by_directory(
        {skills_root: []},
        heading="skills_[draft]*",
        cwd=tmp_path,
    )
    remove_rendered = render_skills_remove_list(
        manager_dir=skills_root,
        manifests=[],
        heading="remove_[draft]*",
        cwd=tmp_path,
    )
    marketplace_rendered = render_marketplace_skills([], heading="market_[draft]*")

    assert directory_rendered.startswith("# skills\\_\\[draft\\]\\*\n")
    assert remove_rendered.startswith("# remove\\_\\[draft\\]\\*\n")
    assert marketplace_rendered.startswith("# market\\_\\[draft\\]\\*\n")


def test_render_skills_by_directory_code_spans_directory_headings(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills_[draft]"

    rendered = render_skills_by_directory({skills_root: []}, heading="skills", cwd=tmp_path)

    assert "## `skills_[draft]`" in rendered


def test_render_skills_remove_list_code_spans_manager_directory_heading(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills`draft"

    rendered = render_skills_remove_list(
        manager_dir=skills_root,
        manifests=[],
        heading="skills remove",
        cwd=tmp_path,
    )

    assert "## `` skills`draft ``" in rendered


def test_render_marketplace_skills_footer_has_clean_guidance() -> None:
    rendered = render_marketplace_skills(
        [
            MarketplaceSkill(
                name="alpha",
                description=None,
                repo_url="https://github.com/example/skills",
                repo_ref="main",
                repo_path="skills/alpha",
            )
        ],
        heading="skills available",
    )

    assert "Search with `/skills search <query>`.\n" in rendered
    assert "Search with `/skills search <query>`. \n" not in rendered
    assert "Change registry with `/skills registry`." in rendered


def test_render_marketplace_skills_includes_source_link() -> None:
    rendered = render_marketplace_skills(
        [
            MarketplaceSkill(
                name="alpha",
                description=None,
                repo_url="https://github.com/example/skills",
                repo_ref="main",
                repo_path="skills/alpha",
                source_url="https://example.test/alpha",
            )
        ],
        heading="skills available",
    )

    assert "    > **Source:**\n    > `https://example.test/alpha`" in rendered


def test_render_marketplace_skills_code_spans_source_url() -> None:
    rendered = render_marketplace_skills(
        [
            MarketplaceSkill(
                name="alpha",
                description=None,
                repo_url="https://github.com/example/skills",
                repo_ref="main",
                repo_path="skills/alpha",
                source_url="https://example.test/a) [bad](https://evil.test)`x",
            )
        ],
        heading="skills available",
    )

    assert "`` https://example.test/a) [bad](https://evil.test)`x ``" in rendered
    assert "[link](" not in rendered


def test_render_marketplace_skills_escapes_backticks_in_repository() -> None:
    rendered = render_marketplace_skills([], heading="skills available", repository="repo`name")

    assert "Repository: `` repo`name ``" in rendered


def test_render_marketplace_skills_strips_and_omits_blank_repository() -> None:
    assert "Repository: `repo`" in render_marketplace_skills(
        [], heading="skills available", repository=" repo "
    )
    assert "Repository:" not in render_marketplace_skills(
        [],
        heading="skills available",
        repository="   ",
    )


def test_render_marketplace_skills_escapes_markdown_names_and_bundles() -> None:
    rendered = render_marketplace_skills(
        [
            MarketplaceSkill(
                name="alpha_[draft]*",
                description=None,
                repo_url="https://github.com/example/skills",
                repo_ref="main",
                repo_path="skills/alpha",
                bundle_name="Bundle_[draft]*",
            )
        ],
        heading="skills available",
    )

    assert "## Bundle\\_\\[draft\\]\\*" in rendered
    assert "1. **alpha\\_\\[draft\\]\\***" in rendered


def test_render_marketplace_skills_escapes_description_markdown() -> None:
    rendered = render_marketplace_skills(
        [
            MarketplaceSkill(
                name="alpha",
                description="Use [docs](bad) and *care*",
                repo_url="https://github.com/example/skills",
                repo_ref="main",
                repo_path="skills/alpha",
                bundle_name="Examples",
                bundle_description="Bundle [docs](bad) and *care*",
            )
        ],
        heading="skills available",
    )

    assert "> Bundle \\[docs\\](bad) and \\*care\\*" in rendered
    assert "    > Use \\[docs\\](bad) and \\*care\\*" in rendered
    assert "[docs](bad)" not in rendered


def test_render_marketplace_skills_truncates_bundle_description_blockquote() -> None:
    rendered = render_marketplace_skills(
        [
            MarketplaceSkill(
                name="alpha",
                description=None,
                repo_url="https://github.com/example/skills",
                repo_ref="main",
                repo_path="skills/alpha",
                bundle_name="Examples",
                bundle_description=" ".join(["description"] * 80),
            )
        ],
        heading="skills available",
    )

    assert rendered.count("> ") == 5
    assert "> …" in rendered

