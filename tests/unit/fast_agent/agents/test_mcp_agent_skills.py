from pathlib import Path

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.skills.registry import SkillRegistry


def create_skill(directory: Path, name: str, description: str = "desc", body: str = "Body") -> None:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    manifest = skill_dir / "SKILLS.md"
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
    assert "alpha" in tool_names

    result = await agent.call_tool("alpha", None)
    assert result.isError is False
    assert result.content
    assert result.content[0].text == "Alpha body"
