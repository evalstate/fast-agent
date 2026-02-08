import os
from contextlib import contextmanager
from pathlib import Path

from fast_agent.skills.registry import SkillRegistry


@contextmanager
def _without_environment_dir():
    import fast_agent.config as config_module

    original_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    original_settings = getattr(config_module, "_settings", None)
    config_module._settings = None
    try:
        yield
    finally:
        config_module._settings = original_settings
        if original_env_dir is not None:
            os.environ["ENVIRONMENT_DIR"] = original_env_dir


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
    claude_dir = tmp_path / ".claude" / "skills"
    write_skill(claude_dir, "beta", body="Beta body")

    with _without_environment_dir():
        registry = SkillRegistry(base_dir=tmp_path)
        assert registry.directories == [default_dir.resolve(), claude_dir.resolve()]

        manifests = registry.load_manifests()
        assert {manifest.name for manifest in manifests} == {"alpha", "beta"}


def test_default_directory_falls_back_to_claude(tmp_path: Path) -> None:
    claude_dir = tmp_path / ".claude" / "skills"
    write_skill(claude_dir, "alpha", body="Alpha body")

    with _without_environment_dir():
        registry = SkillRegistry(base_dir=tmp_path)
        assert registry.directories == [claude_dir.resolve()]
        manifests = registry.load_manifests()
        assert len(manifests) == 1 and manifests[0].name == "alpha"


def test_override_directory(tmp_path: Path) -> None:
    override_dir = tmp_path / "custom"
    write_skill(override_dir, "override", body="Override body")

    registry = SkillRegistry(base_dir=tmp_path, directories=[override_dir])
    assert registry.directories == [override_dir.resolve()]

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
    with _without_environment_dir():
        registry = SkillRegistry(base_dir=tmp_path)
        assert registry.directories == []
        assert registry.load_manifests() == []


def test_registry_reports_errors(tmp_path: Path) -> None:
    invalid_dir = tmp_path / ".fast-agent" / "skills" / "invalid"
    invalid_dir.mkdir(parents=True)
    (invalid_dir / "SKILL.md").write_text("invalid front matter", encoding="utf-8")

    with _without_environment_dir():
        registry = SkillRegistry(base_dir=tmp_path)
        manifests, errors = registry.load_manifests_with_errors()
        assert manifests == []
        assert errors
        assert "invalid" in errors[0]["path"]


def test_override_missing_directory(tmp_path: Path) -> None:
    override_dir = tmp_path / "missing" / "skills"
    registry = SkillRegistry(base_dir=tmp_path, directories=[override_dir])
    manifests = registry.load_manifests()
    assert manifests == []
    assert registry.directories == []
    assert registry.warnings
    assert str(override_dir.resolve()) in registry.warnings[0]


def test_cli_override_propagates_to_global_settings(tmp_path: Path, monkeypatch) -> None:
    """Verify that skills_directory passed to FastAgent updates global settings."""
    import fast_agent.config as config_module
    from fast_agent.skills.manager import resolve_skill_directories

    # Reset global settings
    monkeypatch.setattr(config_module, "_settings", None)

    # Create a custom skills directory with a skill
    custom_skills = tmp_path / "my-skills"
    write_skill(custom_skills, "test-skill", "A test skill")

    # Create a minimal config file
    config_file = tmp_path / "fastagent.config.yaml"
    config_file.write_text("default_model: playback\n", encoding="utf-8")

    # Change to tmp_path so config is found
    monkeypatch.chdir(tmp_path)

    # Import and create FastAgent with skills_directory override
    from fast_agent.core.fastagent import FastAgent

    # Creating FastAgent updates global settings as a side effect
    FastAgent(
        name="test",
        config_path=str(config_file),
        skills_directory=custom_skills,
        ignore_unknown_args=True,
        parse_cli_args=False,
    )

    # Now resolve_skill_directories() should return our custom directory
    directories = resolve_skill_directories()
    directory_strs = [str(d) for d in directories]

    assert str(custom_skills) in directory_strs, (
        f"Expected {custom_skills} in {directory_strs}"
    )
