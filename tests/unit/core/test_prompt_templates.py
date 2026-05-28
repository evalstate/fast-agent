
import os

from fast_agent.core.prompt_templates import enrich_with_environment_context


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
    assert "internal://fast-agent/model-overlays" in context["agentInternalResources"]


def test_enrich_with_environment_context_noenv_omits_environment_paths(tmp_path):
    from fast_agent.config import Settings, get_settings, update_global_settings

    context: dict[str, str] = {}
    settings = Settings()
    settings._fast_agent_noenv = True
    previous_settings = get_settings()

    try:
        update_global_settings(settings)
        enrich_with_environment_context(context, str(tmp_path), {"name": "Zed"}, noenv=True)
    finally:
        update_global_settings(previous_settings)

    assert context["workspaceRoot"] == str(tmp_path)
    assert "environmentDir" not in context
    assert "environmentAgentCardsDir" not in context
    assert "environmentToolCardsDir" not in context
    assert f"Workspace root: {tmp_path}" in context["env"]


def test_enrich_with_environment_context_formats_acp_client_handoff():
    context: dict[str, str] = {}
    client_info = {
        "name": "fast-agent",
        "version": "0.7.1",
        "viaName": "zed",
        "viaTitle": "Zed",
        "viaVersion": "1.2.3",
    }

    enrich_with_environment_context(context, "/workspace/app", client_info)

    assert "Client: fast-agent 0.7.1 via Zed 1.2.3" in context["env"]


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
    original_fast_agent_home = os.environ.pop("FAST_AGENT_HOME", None)
    import fast_agent.config as config_module
    original_settings = getattr(config_module, "_settings", None)
    config_module._settings = None
    try:
        enrich_with_environment_context(context, str(tmp_path), client_info)
    finally:
        config_module._settings = original_settings
        if original_env_dir is not None:
            os.environ["ENVIRONMENT_DIR"] = original_env_dir
        if original_fast_agent_home is not None:
            os.environ["FAST_AGENT_HOME"] = original_fast_agent_home

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
