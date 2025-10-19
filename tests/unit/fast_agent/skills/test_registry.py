from pathlib import Path

from fast_agent.skills.registry import SkillRegistry


def write_skill(directory: Path, name: str, description: str = "desc", body: str = "Body") -> Path:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    manifest = skill_dir / "SKILLS.md"
    manifest.write_text(
        f"""---
name: {name}
description: {description}
---
{body}
""",
        encoding="utf-8",
    )
    return manifest


def test_default_directory_prefers_fast_agent(tmp_path: Path) -> None:
    default_dir = tmp_path / ".fast-agent" / "skills"
    write_skill(default_dir, "alpha", body="Alpha body")
    write_skill(tmp_path / "claude" / "skills", "beta", body="Beta body")

    registry = SkillRegistry(base_dir=tmp_path)
    assert registry.directory == default_dir.resolve()

    manifests = registry.load_manifests()
    assert [manifest.name for manifest in manifests] == ["alpha"]
    assert manifests[0].body == "Alpha body"


def test_default_directory_falls_back_to_claude(tmp_path: Path) -> None:
    claude_dir = tmp_path / "claude" / "skills"
    write_skill(claude_dir, "alpha", body="Alpha body")

    registry = SkillRegistry(base_dir=tmp_path)
    assert registry.directory == claude_dir.resolve()
    manifests = registry.load_manifests()
    assert len(manifests) == 1 and manifests[0].name == "alpha"


def test_override_directory(tmp_path: Path) -> None:
    override_dir = tmp_path / "custom"
    write_skill(override_dir, "override", body="Override body")

    registry = SkillRegistry(base_dir=tmp_path, override_directory=override_dir)
    assert registry.directory == override_dir.resolve()

    manifests = registry.load_manifests()
    assert len(manifests) == 1
    assert manifests[0].name == "override"
    assert manifests[0].body == "Override body"


def test_load_directory_helper(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    write_skill(skills_dir, "alpha")
    write_skill(skills_dir, "beta")

    manifests = SkillRegistry.load_directory(skills_dir)
    assert {manifest.name for manifest in manifests} == {"alpha", "beta"}


def test_no_default_directory(tmp_path: Path) -> None:
    registry = SkillRegistry(base_dir=tmp_path)
    assert registry.directory is None
    assert registry.load_manifests() == []


def test_registry_reports_errors(tmp_path: Path) -> None:
    invalid_dir = tmp_path / ".fast-agent" / "skills" / "invalid"
    invalid_dir.mkdir(parents=True)
    (invalid_dir / "SKILLS.md").write_text("invalid front matter", encoding="utf-8")

    registry = SkillRegistry(base_dir=tmp_path)
    manifests, errors = registry.load_manifests_with_errors()
    assert manifests == []
    assert errors
    assert "invalid" in errors[0]["path"]


def test_override_missing_directory(tmp_path: Path) -> None:
    override_dir = tmp_path / "missing" / "skills"
    registry = SkillRegistry(base_dir=tmp_path, override_directory=override_dir)
    manifests = registry.load_manifests()
    assert manifests == []
    assert registry.override_failed is True
    assert registry.directory is None
