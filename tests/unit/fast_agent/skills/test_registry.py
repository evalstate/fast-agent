from pathlib import Path
from unittest.mock import patch

from fast_agent.skills.registry import (
    SkillManifest,
    SkillRegistry,
    _validate_skill_name,
    format_skills_for_prompt,
)


def write_skill(directory: Path, name: str, description: str = "desc", body: str = "Body") -> Path:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    manifest = skill_dir / "SKILL.md"
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
    claude_dir = tmp_path / ".claude" / "skills"
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
    (invalid_dir / "SKILL.md").write_text("invalid front matter", encoding="utf-8")

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


def test_validate_skill_name_valid(tmp_path: Path) -> None:
    """Valid skill names should produce no warnings."""
    manifest_path = tmp_path / "my-skill" / "SKILL.md"
    manifest_path.parent.mkdir(parents=True)
    warnings = _validate_skill_name("my-skill", manifest_path)
    assert warnings == []


def test_validate_skill_name_uppercase(tmp_path: Path) -> None:
    """Uppercase letters should produce a warning."""
    manifest_path = tmp_path / "MySkill" / "SKILL.md"
    manifest_path.parent.mkdir(parents=True)
    warnings = _validate_skill_name("MySkill", manifest_path)
    assert any("lowercase" in w for w in warnings)


def test_validate_skill_name_consecutive_hyphens(tmp_path: Path) -> None:
    """Consecutive hyphens should produce a warning."""
    manifest_path = tmp_path / "my--skill" / "SKILL.md"
    manifest_path.parent.mkdir(parents=True)
    warnings = _validate_skill_name("my--skill", manifest_path)
    assert any("consecutive hyphens" in w for w in warnings)


def test_validate_skill_name_mismatch_directory(tmp_path: Path) -> None:
    """Name not matching directory should produce a warning."""
    manifest_path = tmp_path / "different-name" / "SKILL.md"
    manifest_path.parent.mkdir(parents=True)
    warnings = _validate_skill_name("my-skill", manifest_path)
    assert any("does not match parent directory" in w for w in warnings)


def test_validate_skill_name_too_long(tmp_path: Path) -> None:
    """Names over 64 characters should produce a warning."""
    long_name = "a" * 65
    manifest_path = tmp_path / long_name / "SKILL.md"
    manifest_path.parent.mkdir(parents=True)
    warnings = _validate_skill_name(long_name, manifest_path)
    assert any("exceeds max length" in w for w in warnings)


def test_format_skills_for_prompt_uses_absolute_paths(tmp_path: Path) -> None:
    """format_skills_for_prompt should use absolute paths per Agent Skills spec."""
    abs_path = tmp_path / "my-skill" / "SKILL.md"
    manifest = SkillManifest(
        name="my-skill",
        description="Test skill",
        body="Body",
        path=abs_path,
        relative_path=Path("my-skill/SKILL.md"),
    )
    prompt = format_skills_for_prompt([manifest])
    # Should use absolute path, not relative
    assert f'path="{abs_path}"' in prompt
    assert 'path="my-skill/SKILL.md"' not in prompt


def test_format_skills_for_prompt_has_read_tool_false(tmp_path: Path) -> None:
    """Without read tool, preamble should mention execute tool."""
    abs_path = tmp_path / "my-skill" / "SKILL.md"
    manifest = SkillManifest(
        name="my-skill",
        description="Test skill",
        body="Body",
        path=abs_path,
    )
    prompt = format_skills_for_prompt([manifest], has_read_tool=False)
    assert "'execute' tool" in prompt
    assert "read_text_file" not in prompt


def test_format_skills_for_prompt_has_read_tool_true(tmp_path: Path) -> None:
    """With read tool, preamble should mention read_text_file tool."""
    abs_path = tmp_path / "my-skill" / "SKILL.md"
    manifest = SkillManifest(
        name="my-skill",
        description="Test skill",
        body="Body",
        path=abs_path,
    )
    prompt = format_skills_for_prompt([manifest], has_read_tool=True)
    assert "'read_text_file' tool" in prompt
    assert "'execute' tool" not in prompt


def test_parse_manifest_logs_name_validation_warnings(tmp_path: Path) -> None:
    """Parsing a manifest with invalid name should log warnings."""
    skills_dir = tmp_path / "skills"
    # Create skill with uppercase name (violates spec)
    write_skill(skills_dir, "MySkill", description="Test")

    with patch("fast_agent.skills.registry.logger") as mock_logger:
        manifests = SkillRegistry.load_directory(skills_dir)

    # Should still parse successfully
    assert len(manifests) == 1
    assert manifests[0].name == "MySkill"
    # But should have logged a warning about the name format
    warning_calls = [
        call for call in mock_logger.warning.call_args_list if "lowercase" in str(call)
    ]
    assert len(warning_calls) > 0
