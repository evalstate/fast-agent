from pathlib import Path
from unittest.mock import patch

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.skills.registry import SkillRegistry, format_skills_for_prompt


def create_skill(directory: Path, name: str, description: str = "desc", body: str = "Body") -> None:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        f"""---\nname: {name}\ndescription: {description}\n---\n{body}\n""",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_mcp_agent_exposes_skill_tools(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "alpha", body="Alpha body")

    manifests = SkillRegistry.load_directory(skills_root)
    context = Context()

    config = AgentConfig(name="test", instruction="Instruction", servers=[], skills=skills_root)
    config.skill_manifests = manifests

    agent = McpAgent(config=config, context=context)

    tools_result = await agent.list_tools()
    tool_names = {tool.name for tool in tools_result.tools}
    assert "alpha" not in tool_names
    assert manifests[0].relative_path == Path("alpha/SKILL.md")


@pytest.mark.asyncio
async def test_agent_skills_template_substitution(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "beta", description="Beta desc", body="Beta body")

    manifests = SkillRegistry.load_directory(skills_root)
    context = Context()

    config = AgentConfig(
        name="test",
        instruction="Instructions:\n\n{{agentSkills}}\nEnd.",
        servers=[],
        skills=skills_root,
    )
    config.skill_manifests = manifests

    agent = McpAgent(config=config, context=context)
    await agent._apply_instruction_templates()

    assert "{{agentSkills}}" not in agent.instruction
    assert '<agent-skill name="beta"' in agent.instruction
    assert 'path="beta/SKILL.md"' in agent.instruction
    assert "Beta desc" in agent.instruction
    assert "<instructions>" not in agent.instruction
    assert "Beta body" not in agent.instruction


@pytest.mark.asyncio
async def test_agent_skills_missing_placeholder_warns(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "gamma")

    manifests = SkillRegistry.load_directory(skills_root)
    context = Context()

    config = AgentConfig(
        name="test",
        instruction="Instruction without placeholder.",
        servers=[],
        skills=skills_root,
    )
    config.skill_manifests = manifests

    agent = McpAgent(config=config, context=context)

    with patch.object(agent.logger, "warning") as mock_warning:
        await agent._apply_instruction_templates()
        await agent._apply_instruction_templates()

    mock_warning.assert_called_once()
    assert "system prompt does not include {{agentSkills}}" in mock_warning.call_args[0][0]


def test_skills_absolute_dir_outside_cwd(tmp_path: Path) -> None:
    """When skills dir is outside base_dir with absolute override, absolute paths should be used."""
    # Create skills in tmp_path (simulates /tmp/foo)
    skills_root = tmp_path / "external_skills"
    create_skill(skills_root, "external", description="External skill")

    # Use a different base_dir that doesn't contain skills_root
    base_dir = tmp_path / "workspace"
    base_dir.mkdir()

    # Create registry with base_dir different from skills directory (absolute override)
    registry = SkillRegistry(base_dir=base_dir, override_directory=skills_root)
    manifests = registry.load_manifests()

    assert len(manifests) == 1
    manifest = manifests[0]

    # relative_path should be computed from the override directory
    # Since override_directory was absolute, it stays as the absolute path prefix
    assert manifest.relative_path is not None
    assert str(manifest.relative_path).endswith("external/SKILL.md")

    # The absolute path should still be set
    assert manifest.path is not None
    assert manifest.path.is_absolute()

    # format_skills_for_prompt should use the relative path from override
    prompt = format_skills_for_prompt(manifests)
    assert f'path="{manifest.relative_path}"' in prompt


def test_skills_relative_dir_outside_cwd(tmp_path: Path) -> None:
    """When skills dir is specified with relative path like ../skills, preserve that path."""
    # Create workspace and external skills directories as siblings
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "my-skill", description="My skill")

    # Use relative path like ../skills
    override_dir = Path("../skills")

    # Create registry with workspace as base_dir and relative override
    registry = SkillRegistry(base_dir=workspace, override_directory=override_dir)
    manifests = registry.load_manifests()

    assert len(manifests) == 1
    manifest = manifests[0]

    # relative_path should preserve the original relative path prefix
    assert manifest.relative_path is not None
    assert str(manifest.relative_path) == "../skills/my-skill/SKILL.md"

    # format_skills_for_prompt should use the relative path
    prompt = format_skills_for_prompt(manifests)
    assert 'path="../skills/my-skill/SKILL.md"' in prompt
