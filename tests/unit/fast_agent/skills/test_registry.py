import os
from contextlib import contextmanager
from pathlib import Path
from xml.etree import ElementTree

from fast_agent.constants import FAST_AGENT_RUNTIME_HOME
from fast_agent.paths import default_skill_paths
from fast_agent.skills.registry import SkillManifest, SkillRegistry, format_skills_for_prompt


@contextmanager
def _without_home():
    import fast_agent.config as config_module

    original_home = os.environ.pop("FAST_AGENT_HOME", None)
    original_fast_agent_home = os.environ.pop("FAST_AGENT_HOME", None)
    original_runtime_environment = os.environ.pop(FAST_AGENT_RUNTIME_HOME, None)
    original_settings = getattr(config_module, "_settings", None)
    config_module._settings = None
    try:
        yield
    finally:
        config_module._settings = original_settings
        if original_runtime_environment is not None:
            os.environ[FAST_AGENT_RUNTIME_HOME] = original_runtime_environment
        if original_fast_agent_home is not None:
            os.environ["FAST_AGENT_HOME"] = original_fast_agent_home
        if original_home is not None:
            os.environ["FAST_AGENT_HOME"] = original_home


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

    with _without_home():
        registry = SkillRegistry(base_dir=tmp_path)
        assert registry.directories == [default_dir.resolve(), claude_dir.resolve()]

        manifests = registry.load_manifests()
        assert {manifest.name for manifest in manifests} == {"alpha", "beta"}


def test_runtime_environment_takes_precedence_over_fast_agent_home(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime_skills = workspace / ".cdx" / "skills"
    home_skills = tmp_path / "home" / "skills"
    write_skill(runtime_skills, "runtime", body="Runtime body")
    write_skill(home_skills, "home", body="Home body")

    monkeypatch.setenv("FAST_AGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(FAST_AGENT_RUNTIME_HOME, str(workspace / ".cdx"))
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)

    registry = SkillRegistry(base_dir=workspace)

    assert registry.directories == [runtime_skills.resolve()]
    assert [manifest.name for manifest in registry.load_manifests()] == ["runtime"]


def test_default_directory_falls_back_to_claude(tmp_path: Path) -> None:
    claude_dir = tmp_path / ".claude" / "skills"
    write_skill(claude_dir, "alpha", body="Alpha body")

    with _without_home():
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


def test_load_manifests_deduplicates_skill_names_case_insensitively(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    write_skill(first_dir, "Alpha", body="First body")
    write_skill(second_dir, "alpha", body="Second body")

    registry = SkillRegistry(base_dir=tmp_path, directories=[first_dir, second_dir])

    manifests = registry.load_manifests()

    assert [manifest.name for manifest in manifests] == ["alpha"]
    assert manifests[0].body == "Second body"
    assert "Duplicate skill 'alpha'" in registry.warnings[0]


def test_format_skills_for_prompt_escapes_xml_text(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "alpha & beta"
    skill_dir.mkdir(parents=True)
    manifest_path = skill_dir / "SKILL.md"
    manifest_path.write_text("---\nname: alpha\n---\n", encoding="utf-8")
    (skill_dir / "references").mkdir()
    manifest = SkillManifest(
        name="alpha <beta>",
        description="Use A & B > C",
        body="",
        path=manifest_path,
    )

    formatted = format_skills_for_prompt([manifest], include_preamble=False)

    root = ElementTree.fromstring(formatted)
    skill = root.find("skill")
    assert skill is not None
    assert skill.findtext("name") == "alpha <beta>"
    assert skill.findtext("description") == "Use A & B > C"
    assert skill.findtext("location") == str(manifest_path)
    assert skill.findtext("directory") == str(skill_dir)
    assert skill.findtext("references") == str(skill_dir / "references")
    assert "alpha &lt;beta&gt;" in formatted
    assert "Use A &amp; B &gt; C" in formatted


def test_format_skills_for_prompt_omits_blank_description(tmp_path: Path) -> None:
    manifest = SkillManifest(
        name="alpha",
        description="   ",
        body="",
        path=tmp_path / "skills" / "alpha" / "SKILL.md",
    )

    formatted = format_skills_for_prompt([manifest], include_preamble=False)

    root = ElementTree.fromstring(formatted)
    skill = root.find("skill")
    assert skill is not None
    assert skill.find("description") is None


def test_parse_manifest_text_normalizes_optional_fields() -> None:
    manifest, error = SkillRegistry.parse_manifest_text(
        """---
name: "  demo  "
description: "  Demo skill  "
license: "  MIT  "
compatibility: "   "
allowed-tools: "  bash   python  "
metadata:
  retries: 3
  enabled: true
---
Body
"""
    )

    assert error is None
    assert manifest is not None
    assert manifest.name == "demo"
    assert manifest.description == "Demo skill"
    assert manifest.license == "MIT"
    assert manifest.compatibility is None
    assert manifest.allowed_tools == ["bash", "python"]
    assert manifest.metadata == {"retries": "3", "enabled": "True"}


def test_no_default_directory(tmp_path: Path) -> None:
    with _without_home():
        registry = SkillRegistry(base_dir=tmp_path)
        assert registry.directories == []
        assert registry.load_manifests() == []


def test_registry_reports_errors(tmp_path: Path) -> None:
    invalid_dir = tmp_path / ".fast-agent" / "skills" / "invalid"
    invalid_dir.mkdir(parents=True)
    (invalid_dir / "SKILL.md").write_text("invalid front matter", encoding="utf-8")

    with _without_home():
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


def test_explicit_default_directories_are_optional(tmp_path: Path) -> None:
    with _without_home():
        default_dirs = default_skill_paths(cwd=tmp_path)
        registry = SkillRegistry(base_dir=tmp_path, directories=default_dirs)
        manifests = registry.load_manifests()

    assert manifests == []
    assert registry.warnings == []


def test_cli_override_propagates_to_global_settings(tmp_path: Path, monkeypatch) -> None:
    """Verify that skills_directory passed to FastAgent updates global settings."""
    import fast_agent.config as config_module
    from fast_agent.skills.scope import resolve_skill_directories

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

    assert str(custom_skills) in directory_strs, f"Expected {custom_skills} in {directory_strs}"
