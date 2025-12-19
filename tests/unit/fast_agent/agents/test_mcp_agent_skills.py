from pathlib import Path
from unittest.mock import patch

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.skills.registry import SkillManifest, SkillRegistry, format_skills_for_prompt
from fast_agent.tools.skill_reader import SkillReader


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
    # Per Agent Skills standard, we now use 'location' attribute with absolute paths by default
    # The location points to the skill directory (not the SKILL.md file)
    assert 'location="' in agent.instruction
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

    # format_skills_for_prompt uses absolute paths by default (location attribute)
    prompt = format_skills_for_prompt(manifests)
    assert f'location="{manifest.path.parent}"' in prompt

    # With use_absolute_paths=False, it uses relative paths
    prompt_relative = format_skills_for_prompt(manifests, use_absolute_paths=False)
    assert f'location="{manifest.relative_path.parent}"' in prompt_relative


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

    # format_skills_for_prompt uses absolute paths by default
    prompt = format_skills_for_prompt(manifests)
    assert f'location="{manifest.path.parent}"' in prompt

    # With use_absolute_paths=False, it uses relative paths
    prompt_relative = format_skills_for_prompt(manifests, use_absolute_paths=False)
    assert 'location="../skills/my-skill"' in prompt_relative


# ============================================================================
# Skill Reader Tests
# ============================================================================


@pytest.mark.asyncio
async def test_skill_reader_reads_skill_md(tmp_path: Path) -> None:
    """Test that SkillReader can read SKILL.md files."""
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "test-skill", description="Test skill", body="Skill instructions here")

    manifests = SkillRegistry.load_directory(skills_root)
    reader = SkillReader(manifests)

    assert reader.enabled
    assert reader.tool is not None
    assert reader.tool.name == "read_skill"

    # Read the SKILL.md file
    result = await reader.execute({"skill_name": "test-skill"})
    assert not result.isError
    assert len(result.content) == 1
    assert "Skill instructions here" in result.content[0].text


@pytest.mark.asyncio
async def test_skill_reader_reads_resources(tmp_path: Path) -> None:
    """Test that SkillReader can read resources in skill directories."""
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "test-skill"
    skill_dir.mkdir(parents=True)

    # Create SKILL.md
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: Test skill\n---\nBody",
        encoding="utf-8",
    )

    # Create a script resource
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "helper.py").write_text("print('hello')", encoding="utf-8")

    manifests = SkillRegistry.load_directory(skills_root)
    reader = SkillReader(manifests)

    # Read the script
    result = await reader.execute({"skill_name": "test-skill", "path": "scripts/helper.py"})
    assert not result.isError
    assert "print('hello')" in result.content[0].text


@pytest.mark.asyncio
async def test_skill_reader_unknown_skill(tmp_path: Path) -> None:
    """Test that SkillReader returns error for unknown skills."""
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "known-skill")

    manifests = SkillRegistry.load_directory(skills_root)
    reader = SkillReader(manifests)

    result = await reader.execute({"skill_name": "unknown-skill"})
    assert result.isError
    assert "Unknown skill" in result.content[0].text


@pytest.mark.asyncio
async def test_skill_reader_path_traversal_blocked(tmp_path: Path) -> None:
    """Test that SkillReader blocks path traversal attempts."""
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "test-skill")

    # Create a file outside the skill directory
    (tmp_path / "secret.txt").write_text("secret data", encoding="utf-8")

    manifests = SkillRegistry.load_directory(skills_root)
    reader = SkillReader(manifests)

    # Try to read file outside skill directory
    result = await reader.execute({"skill_name": "test-skill", "path": "../../secret.txt"})
    assert result.isError
    assert "outside the skill directory" in result.content[0].text


@pytest.mark.asyncio
async def test_skill_reader_file_not_found(tmp_path: Path) -> None:
    """Test that SkillReader returns error for missing files."""
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "test-skill")

    manifests = SkillRegistry.load_directory(skills_root)
    reader = SkillReader(manifests)

    result = await reader.execute({"skill_name": "test-skill", "path": "nonexistent.md"})
    assert result.isError
    assert "File not found" in result.content[0].text


@pytest.mark.asyncio
async def test_mcp_agent_includes_skill_reader_tool(tmp_path: Path) -> None:
    """Test that McpAgent includes the read_skill tool when skills are configured."""
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "alpha")

    manifests = SkillRegistry.load_directory(skills_root)
    context = Context()

    config = AgentConfig(name="test", instruction="Instruction", servers=[], skills=skills_root)
    config.skill_manifests = manifests

    agent = McpAgent(config=config, context=context)

    tools_result = await agent.list_tools()
    tool_names = {tool.name for tool in tools_result.tools}

    # Should include read_skill tool when skills are configured
    assert "read_skill" in tool_names


@pytest.mark.asyncio
async def test_mcp_agent_skill_reader_not_added_with_filesystem_runtime(tmp_path: Path) -> None:
    """Test that read_skill tool is not added when filesystem runtime is available (ACP mode)."""
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "alpha")

    manifests = SkillRegistry.load_directory(skills_root)
    context = Context()

    config = AgentConfig(name="test", instruction="Instruction", servers=[], skills=skills_root)
    config.skill_manifests = manifests

    agent = McpAgent(config=config, context=context)

    # Simulate ACP filesystem runtime being available
    class MockFilesystemRuntime:
        tools = []

    agent._filesystem_runtime = MockFilesystemRuntime()

    tools_result = await agent.list_tools()
    tool_names = {tool.name for tool in tools_result.tools}

    # Should NOT include read_skill tool when filesystem runtime is available
    assert "read_skill" not in tool_names
