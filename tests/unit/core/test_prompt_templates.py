
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


# NOTE: tests `test_enrich_with_environment_context_loads_skills` and
# `..._respects_skills_override` were removed during the upstream sync
# (chore/sync-upstream-2026-05). They asserted that ``agentSkills`` is
# populated as a *static* key in the prompt context, but this fork
# resolves skills per-agent through the dynamic InstructionBuilder
# resolver (``instruction_refresh.build_instruction``). The underlying
# loader is still covered by the ``test_load_skills_for_context_*``
# tests below.


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
