from pathlib import Path
from unittest.mock import patch

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.skills.registry import SkillRegistry


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
    assert "<description>Beta desc</description>" in agent.instruction
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
