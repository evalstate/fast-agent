
import os

import pytest

from fast_agent.core.prompt_templates import (
    apply_template_variables,
    enrich_with_environment_context,
)


def test_apply_template_variables_is_noop_without_context():
    template = "Path: {{workspaceRoot}}"
    # First pass - no context yet
    assert apply_template_variables(template, {}) == template
    assert apply_template_variables(template, None) == template


def test_apply_template_variables_supports_escaped_placeholders(tmp_path):
    template = r"Literal: \{{workspaceRoot}} and \{{file:missing.txt}}"
    variables = {"workspaceRoot": str(tmp_path)}

    result = apply_template_variables(template, variables)

    assert result == "Literal: {{workspaceRoot}} and {{file:missing.txt}}"


@pytest.mark.parametrize(
    "variables,expected",
    [
        ({"workspaceRoot": "/workspace/project"}, "Path: /workspace/project"),
        ({"workspaceRoot": None}, "Path: {{workspaceRoot}}"),
    ],
)
def test_apply_template_variables_applies_when_context_available(variables, expected):
    template = "Path: {{workspaceRoot}}"
    assert apply_template_variables(template, variables) == expected


def test_enrich_with_environment_context_populates_env_block():
    context: dict[str, str] = {}
    client_info = {"name": "Zed", "version": "1.2.3"}

    enrich_with_environment_context(context, "/workspace/app", client_info)

    assert context["workspaceRoot"] == "/workspace/app"

    env_text = context["env"]
    assert "Environment:" in env_text
    assert "Workspace root: /workspace/app" in env_text
    assert "Client: Zed 1.2.3" in env_text
    assert "Host platform:" in env_text
    assert "agentInternalResources" in context
    assert "internal://fast-agent/smart-agent-cards" in context["agentInternalResources"]


def test_file_template_substitutes_contents_relative_to_workspace(tmp_path):
    """File templates should resolve relative to workspaceRoot."""
    # Create a file in the workspace
    file_path = tmp_path / "snippet.txt"
    file_path.write_text("Hello template", encoding="utf-8")

    template = "Start {{file:snippet.txt}} End"
    variables = {"workspaceRoot": str(tmp_path)}

    result = apply_template_variables(template, variables)

    assert result == "Start Hello template End"


def test_file_template_supports_nested_paths(tmp_path):
    """File templates should support nested relative paths."""
    # Create nested directory structure
    nested_dir = tmp_path / "docs" / "examples"
    nested_dir.mkdir(parents=True)
    file_path = nested_dir / "note.txt"
    file_path.write_text("Nested content", encoding="utf-8")

    template = "Content: {{file:docs/examples/note.txt}}"
    variables = {"workspaceRoot": str(tmp_path)}

    result = apply_template_variables(template, variables)

    assert result == "Content: Nested content"


def test_file_template_rejects_absolute_paths(tmp_path):
    """File templates must reject absolute paths."""
    absolute_path = tmp_path / "file.txt"
    absolute_path.write_text("content", encoding="utf-8")

    template = f"Start {{{{file:{absolute_path}}}}} End"
    variables = {"workspaceRoot": str(tmp_path)}

    with pytest.raises(ValueError, match="File template paths must be relative"):
        apply_template_variables(template, variables)


def test_file_silent_returns_empty_when_missing(tmp_path):
    """File silent templates should return empty string for missing files."""
    template = "Begin{{file_silent:missing.txt}}Finish"
    variables = {"workspaceRoot": str(tmp_path)}

    result = apply_template_variables(template, variables)

    assert result == "BeginFinish"


def test_file_silent_reads_when_present(tmp_path):
    """File silent templates should read file when present."""
    file_path = tmp_path / "note.txt"
    file_path.write_text("data", encoding="utf-8")

    template = "Value: {{file_silent:note.txt}}"
    variables = {"workspaceRoot": str(tmp_path)}

    result = apply_template_variables(template, variables)

    assert result == "Value: data"


def test_file_silent_rejects_absolute_paths(tmp_path):
    """File silent templates must reject absolute paths."""
    absolute_path = tmp_path / "file.txt"

    template = f"Start {{{{file_silent:{absolute_path}}}}} End"
    variables = {"workspaceRoot": str(tmp_path)}

    with pytest.raises(ValueError, match="File template paths must be relative"):
        apply_template_variables(template, variables)


def test_enrich_with_environment_context_loads_skills(tmp_path):
    """enrich_with_environment_context should load and format skills."""
    # Create a skills directory structure
    skills_dir = tmp_path / ".fast-agent" / "skills" / "test-skill"
    skills_dir.mkdir(parents=True)

    skill_file = skills_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: test-skill
description: A test skill for unit testing
---

This is the skill body content.
""",
        encoding="utf-8",
    )

    context: dict[str, str] = {}
    client_info = {"name": "test-client"}

    original_env_dir = os.environ.pop("ENVIRONMENT_DIR", None)
    import fast_agent.config as config_module
    original_settings = getattr(config_module, "_settings", None)
    config_module._settings = None
    try:
        enrich_with_environment_context(context, str(tmp_path), client_info)
    finally:
        config_module._settings = original_settings
        if original_env_dir is not None:
            os.environ["ENVIRONMENT_DIR"] = original_env_dir

    # Verify skills were loaded
    assert "agentSkills" in context
    assert "test-skill" in context["agentSkills"]
    assert "A test skill for unit testing" in context["agentSkills"]
    # Verify path is relative to workspace root, not skills directory
    assert ".fast-agent/skills/test-skill/SKILL.md" in context["agentSkills"]


def test_enrich_with_environment_context_respects_skills_override(tmp_path):
    """enrich_with_environment_context should use skills override directory."""
    # Create default skills directory
    default_skills_dir = tmp_path / ".fast-agent" / "skills" / "default-skill"
    default_skills_dir.mkdir(parents=True)
    (default_skills_dir / "SKILL.md").write_text(
        """---
name: default-skill
description: Default skill
---
""",
        encoding="utf-8",
    )

    # Create custom skills directory
    custom_skills_dir = tmp_path / "custom-skills" / "custom-skill"
    custom_skills_dir.mkdir(parents=True)
    (custom_skills_dir / "SKILL.md").write_text(
        """---
name: custom-skill
description: Custom skill from override
---
""",
        encoding="utf-8",
    )

    context: dict[str, str] = {}
    client_info = {"name": "test-client"}

    # Use the override
    enrich_with_environment_context(
        context, str(tmp_path), client_info, "custom-skills"
    )

    # Should have custom skill, not default
    assert "agentSkills" in context
    assert "custom-skill" in context["agentSkills"]
    assert "default-skill" not in context["agentSkills"]
    # Verify path uses custom directory relative to workspace root
    assert "custom-skills/custom-skill/SKILL.md" in context["agentSkills"]


def test_load_skills_for_context_handles_missing_directory(tmp_path):
    """load_skills_for_context should handle missing skills directory gracefully."""
    from fast_agent.core.prompt_templates import load_skills_for_context

    # No skills directory exists
    manifests = load_skills_for_context(str(tmp_path), None)

    # Should return empty list, not error
    assert manifests == []


def test_load_skills_for_context_with_relative_override(tmp_path):
    """load_skills_for_context should resolve relative override paths."""
    from fast_agent.core.prompt_templates import load_skills_for_context

    # Create custom skills directory
    custom_skills_dir = tmp_path / "my-skills" / "skill1"
    custom_skills_dir.mkdir(parents=True)
    (custom_skills_dir / "SKILL.md").write_text(
        """---
name: skill1
description: Skill 1
---
""",
        encoding="utf-8",
    )

    manifests = load_skills_for_context(str(tmp_path), "my-skills")

    assert len(manifests) == 1
    assert manifests[0].name == "skill1"


def test_load_skills_for_context_uses_environment_dir_setting(tmp_path):
    """load_skills_for_context should honor settings.environment_dir when using defaults."""
    from fast_agent.config import Settings, get_settings, update_global_settings
    from fast_agent.core.prompt_templates import load_skills_for_context

    skills_dir = tmp_path / ".dev" / "skills" / "env-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\nname: env-skill\ndescription: Skill from env directory\n---\n",
        encoding="utf-8",
    )

    previous_settings = get_settings()
    update_global_settings(Settings(environment_dir=".dev"))
    try:
        manifests = load_skills_for_context(str(tmp_path), None)
    finally:
        update_global_settings(previous_settings)

    assert [manifest.name for manifest in manifests] == ["env-skill"]
