from pathlib import Path
from unittest.mock import patch

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.skills.registry import SkillManifest, SkillRegistry, format_skills_for_prompt


def create_skill(
    directory: Path,
    name: str,
    description: str = "desc",
    body: str = "Body",
    license: str | None = None,
    compatibility: str | None = None,
    allowed_tools: str | None = None,
) -> None:
    """Create a skill with optional metadata fields."""
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    manifest = skill_dir / "SKILL.md"

    # Build YAML frontmatter
    lines = ["---", f"name: {name}", f"description: {description}"]
    if license:
        lines.append(f"license: {license}")
    if compatibility:
        lines.append(f"compatibility: {compatibility}")
    if allowed_tools:
        lines.append(f"allowed-tools: {allowed_tools}")
    lines.extend(["---", body, ""])

    manifest.write_text("\n".join(lines), encoding="utf-8")


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
    # Skills are not exposed as individual tools
    assert "alpha" not in tool_names
    # But read_skill tool should be available since skills are configured
    assert "read_skill" in tool_names
    # Path should be absolute
    assert manifests[0].path.is_absolute()


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
    # New standard format uses <skill> and child elements
    assert "<skill>" in agent.instruction
    assert "<name>beta</name>" in agent.instruction
    assert "<description>Beta desc</description>" in agent.instruction
    # Absolute path in <location> element
    assert "<location>" in agent.instruction
    assert str(skills_root / "beta" / "SKILL.md") in agent.instruction
    assert "<instructions>" not in agent.instruction
    # Body is not included in the skills listing
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

    # Path should be absolute per Agent Skills specification
    assert manifest.path is not None
    assert manifest.path.is_absolute()

    # format_skills_for_prompt should use the absolute path in <location>
    prompt = format_skills_for_prompt(manifests)
    assert f"<location>{manifest.path}</location>" in prompt
    assert "<skill>" in prompt
    assert "<name>external</name>" in prompt


def test_skills_relative_dir_outside_cwd(tmp_path: Path) -> None:
    """When skills dir is specified with relative path like ../skills, absolute paths are still used."""
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

    # Path is resolved to absolute per Agent Skills specification
    assert manifest.path is not None
    assert manifest.path.is_absolute()

    # format_skills_for_prompt should use the absolute path
    prompt = format_skills_for_prompt(manifests)
    assert f"<location>{manifest.path}</location>" in prompt


def test_format_skills_for_prompt_standard_format(tmp_path: Path) -> None:
    """Test that format_skills_for_prompt outputs Agent Skills standard format."""
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "test-skill", description="Test description")

    manifests = SkillRegistry.load_directory(skills_root)
    prompt = format_skills_for_prompt(manifests)

    # Check standard XML structure
    assert "<available_skills>" in prompt
    assert "</available_skills>" in prompt
    assert "<skill>" in prompt
    assert "</skill>" in prompt
    assert "<name>test-skill</name>" in prompt
    assert "<description>Test description</description>" in prompt
    assert "<location>" in prompt
    assert "</location>" in prompt

    # Check preamble mentions read_skill tool by default
    assert "read_skill" in prompt


def test_format_skills_for_prompt_custom_read_tool(tmp_path: Path) -> None:
    """Test that format_skills_for_prompt can use custom read tool name."""
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "test-skill", description="Test description")

    manifests = SkillRegistry.load_directory(skills_root)
    prompt = format_skills_for_prompt(manifests, read_tool_name="read_text_file")

    # Check preamble mentions the custom tool name
    assert "read_text_file" in prompt
    assert "read_skill" not in prompt


def test_format_skills_for_prompt_no_preamble(tmp_path: Path) -> None:
    """Test that format_skills_for_prompt can exclude preamble."""
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "test-skill", description="Test description")

    manifests = SkillRegistry.load_directory(skills_root)
    prompt = format_skills_for_prompt(manifests, include_preamble=False)

    # Should start directly with available_skills
    assert prompt.startswith("<available_skills>")
    # Should not have preamble text
    assert "Use a Skill" not in prompt


def test_skill_manifest_optional_fields(tmp_path: Path) -> None:
    """Test that optional fields from Agent Skills spec are parsed."""
    skills_root = tmp_path / "skills"
    create_skill(
        skills_root,
        "full-skill",
        description="A fully specified skill",
        license="MIT",
        compatibility="Python 3.10+, network access required",
        allowed_tools="bash python",
    )

    manifests = SkillRegistry.load_directory(skills_root)

    assert len(manifests) == 1
    manifest = manifests[0]

    assert manifest.name == "full-skill"
    assert manifest.description == "A fully specified skill"
    assert manifest.license == "MIT"
    assert manifest.compatibility == "Python 3.10+, network access required"
    assert manifest.allowed_tools == ["bash", "python"]


def test_skill_manifest_missing_optional_fields(tmp_path: Path) -> None:
    """Test that missing optional fields are None."""
    skills_root = tmp_path / "skills"
    create_skill(skills_root, "minimal-skill", description="Minimal skill")

    manifests = SkillRegistry.load_directory(skills_root)

    assert len(manifests) == 1
    manifest = manifests[0]

    assert manifest.name == "minimal-skill"
    assert manifest.description == "Minimal skill"
    assert manifest.license is None
    assert manifest.compatibility is None
    assert manifest.allowed_tools is None
    assert manifest.metadata is None
