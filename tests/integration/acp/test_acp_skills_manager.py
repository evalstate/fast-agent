"""Integration tests for ACP skills manager commands."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, cast

import pytest

from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.config import get_settings

if TYPE_CHECKING:
    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.skills.registry import SkillManifest, SkillRegistry


@dataclass
class StubAggregator:
    """Stub aggregator for testing."""

    async def get_server_instructions(self) -> dict[str, tuple[str | None, list[str]]]:
        return {}


@dataclass
class SkillAgent:
    """Skill agent that implements McpInstructionCapable protocol for testing."""

    name: str
    _instruction: str = ""
    _instruction_template: str = "{{agentSkills}}"
    _instruction_context: dict[str, str] = field(default_factory=dict)
    _skill_manifests: list[SkillManifest] = field(default_factory=list)
    _skill_registry: SkillRegistry | None = None
    _aggregator: StubAggregator = field(default_factory=StubAggregator)
    message_history: list[Any] = field(default_factory=list)
    llm: Any = None

    @property
    def instruction(self) -> str:
        return self._instruction

    def set_instruction(self, instruction: str) -> None:
        self._instruction = instruction

    @property
    def instruction_template(self) -> str:
        return self._instruction_template

    @property
    def instruction_context(self) -> dict[str, str]:
        return self._instruction_context

    def set_instruction_context(self, context: dict[str, str]) -> None:
        self._instruction_context = context

    @property
    def aggregator(self) -> StubAggregator:
        return self._aggregator

    @property
    def skill_manifests(self) -> Sequence[SkillManifest]:
        return self._skill_manifests

    @property
    def skill_registry(self) -> SkillRegistry | None:
        return self._skill_registry

    @skill_registry.setter
    def skill_registry(self, value: SkillRegistry | None) -> None:
        self._skill_registry = value

    def set_skill_manifests(self, manifests: Sequence[SkillManifest]) -> None:
        self._skill_manifests = list(manifests)

    @property
    def has_filesystem_runtime(self) -> bool:
        return False


@dataclass
class StubAgentInstance:
    agents: dict[str, Any] = field(default_factory=dict)


def _handler(instance: StubAgentInstance, agent_name: str) -> SlashCommandHandler:
    return SlashCommandHandler(
        "test-session",
        cast("AgentInstance", instance),
        agent_name,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skills_list_includes_key_details(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    skill_dir = repo_root / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)

    skill_manifest = skill_dir / "SKILL.md"
    skill_manifest.write_text(
        """---
name: test-skill
description: MAGIC_SKILL
---

Test skill body.
""",
        encoding="utf-8",
    )

    manager_dir = repo_root / "skills"
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        """default_model: passthrough
skills:
  directories: ['{manager_dir}']
""".format(manager_dir=manager_dir.as_posix()),
        encoding="utf-8",
    )

    get_settings(config_path=str(config_path))
    try:
        agent = SkillAgent(name="test-agent")
        instance = StubAgentInstance(agents={"test-agent": agent})
        handler = _handler(instance, "test-agent")

        response = await handler.execute_command("skills", "")
        assert "test-skill" in response
        assert "MAGIC_SKILL" in response
        assert skill_dir.as_posix() in response
    finally:
        get_settings(config_path=str(Path(__file__).parent / "fastagent.config.yaml"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skills_add_lists_marketplace(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    skill_dir = repo_root / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)

    skill_manifest = skill_dir / "SKILL.md"
    skill_manifest.write_text(
        """---
name: test-skill
description: MAGIC_SKILL
---

Test skill body.
""",
        encoding="utf-8",
    )

    marketplace_path = tmp_path / "marketplace.json"
    marketplace_payload = {
        "plugins": [
            {
                "name": "test-skill",
                "description": "MAGIC_SKILL",
                "repo_url": repo_root.as_posix(),
                "repo_path": "skills/test-skill",
            }
        ]
    }
    marketplace_path.write_text(
        json.dumps(marketplace_payload),
        encoding="utf-8",
    )

    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        """default_model: passthrough
skills:
  marketplace_url: '{marketplace_url}'
""".format(marketplace_url=marketplace_path.as_posix()),
        encoding="utf-8",
    )

    get_settings(config_path=str(config_path))
    try:
        agent = SkillAgent(name="test-agent")
        instance = StubAgentInstance(agents={"test-agent": agent})
        handler = _handler(instance, "test-agent")

        response = await handler.execute_command("skills", "add")
        assert "test-skill" in response
        assert "MAGIC_SKILL" in response
        assert repo_root.as_posix() in response
    finally:
        get_settings(config_path=str(Path(__file__).parent / "fastagent.config.yaml"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skills_available_lists_marketplace(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    skill_dir = repo_root / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)

    skill_manifest = skill_dir / "SKILL.md"
    skill_manifest.write_text(
        """---
name: test-skill
description: MAGIC_SKILL
---

Test skill body.
""",
        encoding="utf-8",
    )

    marketplace_path = tmp_path / "marketplace.json"
    marketplace_payload = {
        "plugins": [
            {
                "name": "test-skill",
                "description": "MAGIC_SKILL",
                "repo_url": repo_root.as_posix(),
                "repo_path": "skills/test-skill",
            }
        ]
    }
    marketplace_path.write_text(
        json.dumps(marketplace_payload),
        encoding="utf-8",
    )

    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        """default_model: passthrough
skills:
  marketplace_url: '{marketplace_url}'
""".format(marketplace_url=marketplace_path.as_posix()),
        encoding="utf-8",
    )

    get_settings(config_path=str(config_path))
    try:
        agent = SkillAgent(name="test-agent")
        instance = StubAgentInstance(agents={"test-agent": agent})
        handler = _handler(instance, "test-agent")

        response = await handler.execute_command("skills", "available")
        assert "test-skill" in response
        assert "MAGIC_SKILL" in response
        assert repo_root.as_posix() in response
    finally:
        get_settings(config_path=str(Path(__file__).parent / "fastagent.config.yaml"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skills_help_shows_usage(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("default_model: passthrough\n", encoding="utf-8")

    get_settings(config_path=str(config_path))
    try:
        agent = SkillAgent(name="test-agent")
        instance = StubAgentInstance(agents={"test-agent": agent})
        handler = _handler(instance, "test-agent")

        response = await handler.execute_command("skills", "--help")
        assert "Usage: /skills [list|available|search|add|remove|update|registry|help]" in response
    finally:
        get_settings(config_path=str(Path(__file__).parent / "fastagent.config.yaml"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skills_remove_lists_local_skills(tmp_path: Path) -> None:
    manager_dir = tmp_path / "managed-skills"
    skill_dir = manager_dir / "test-skill"
    skill_dir.mkdir(parents=True)

    skill_manifest = skill_dir / "SKILL.md"
    skill_manifest.write_text(
        """---
name: test-skill
description: MAGIC_SKILL
---

Test skill body.
""",
        encoding="utf-8",
    )

    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        """default_model: passthrough
skills:
  directories: ['{manager_dir}']
""".format(manager_dir=manager_dir.as_posix()),
        encoding="utf-8",
    )

    get_settings(config_path=str(config_path))
    try:
        agent = SkillAgent(name="test-agent")
        instance = StubAgentInstance(agents={"test-agent": agent})
        handler = _handler(instance, "test-agent")

        response = await handler.execute_command("skills", "remove")
        assert "test-skill" in response
        assert "MAGIC_SKILL" in response
        assert manager_dir.as_posix() in response
    finally:
        get_settings(config_path=str(Path(__file__).parent / "fastagent.config.yaml"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skills_add_remove_refreshes_system_prompt(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    skill_dir = repo_root / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)

    skill_manifest = skill_dir / "SKILL.md"
    skill_manifest.write_text(
        "---\nname: test-skill\ndescription: MAGIC_SKILL\n---\n\nTest skill body.\n",
        encoding="utf-8",
    )

    marketplace_path = tmp_path / "marketplace.json"
    marketplace_path.write_text(
        "{\n"
        '  "plugins": [\n'
        "    {\n"
        '      "name": "test-skill",\n'
        '      "description": "MAGIC_SKILL",\n'
        f'      "repo_url": "{repo_root.as_posix()}",\n'
        '      "repo_path": "skills/test-skill"\n'
        "    }\n"
        "  ]\n"
        "}\n",
        encoding="utf-8",
    )

    manager_dir = tmp_path / "managed-skills"
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        "skills:\n"
        f"  directories: ['{manager_dir.as_posix()}']\n"
        f"  marketplace_url: '{marketplace_path.as_posix()}'\n",
        encoding="utf-8",
    )

    get_settings(config_path=str(config_path))
    try:
        agent = SkillAgent(name="test-agent")
        instance = StubAgentInstance(agents={"test-agent": agent})
        handler = _handler(instance, "test-agent")

        response = await handler.execute_command("skills", "add test-skill")
        assert "Installed" in response

        status = await handler.execute_command("status", "system")
        assert "MAGIC_SKILL" in status

        response = await handler.execute_command("skills", "remove test-skill")
        assert "Removed" in response

        status = await handler.execute_command("status", "system")
        assert "MAGIC_SKILL" not in status
    finally:
        get_settings(config_path=str(Path(__file__).parent / "fastagent.config.yaml"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skills_registry_updates_marketplace_url(tmp_path: Path) -> None:
    marketplace_path = tmp_path / "marketplace.json"
    marketplace_path.write_text(
        "{\n"
        '  "plugins": [\n'
        "    {\n"
        '      "name": "test-skill",\n'
        '      "description": "Skill Description",\n'
        '      "repo_url": "https://github.com/example/repo",\n'
        '      "repo_path": "skills/test-skill"\n'
        "    }\n"
        "  ]\n"
        "}\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        "skills:\n"
        "  directories: ['.fast-agent/skills']\n"
        "  marketplace_urls: ['https://example.com/marketplace.json']\n",
        encoding="utf-8",
    )

    get_settings(config_path=str(config_path))
    try:
        agent = SkillAgent(name="test-agent")
        instance = StubAgentInstance(agents={"test-agent": agent})
        handler = _handler(instance, "test-agent")

        response = await handler.execute_command(
            "skills", f"registry {marketplace_path.as_posix()}"
        )
        assert "Registry set to" in response
        assert marketplace_path.as_posix() in response
        # Active registry is updated
        assert get_settings().skills.marketplace_url == marketplace_path.as_posix()
        # Configured list is preserved
        assert get_settings().skills.marketplace_urls == ["https://example.com/marketplace.json"]
    finally:
        get_settings(config_path=str(Path(__file__).parent / "fastagent.config.yaml"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skills_registry_numbered_selection(tmp_path: Path) -> None:
    """Test selecting a registry by number from the configured list."""
    marketplace1 = tmp_path / "marketplace1.json"
    marketplace1.write_text(
        '{"plugins": [{"name": "skill1", "description": "Skill 1", '
        '"repo_url": "https://github.com/example/repo", "repo_path": "skills/skill1"}]}',
        encoding="utf-8",
    )
    marketplace2 = tmp_path / "marketplace2.json"
    marketplace2.write_text(
        '{"plugins": [{"name": "skill2", "description": "Skill 2", '
        '"repo_url": "https://github.com/example/repo", "repo_path": "skills/skill2"}]}',
        encoding="utf-8",
    )

    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        "skills:\n"
        "  directories: ['.fast-agent/skills']\n"
        f"  marketplace_urls:\n"
        f"    - '{marketplace1.as_posix()}'\n"
        f"    - '{marketplace2.as_posix()}'\n",
        encoding="utf-8",
    )

    get_settings(config_path=str(config_path))
    try:
        agent = SkillAgent(name="test-agent")
        instance = StubAgentInstance(agents={"test-agent": agent})
        handler = _handler(instance, "test-agent")

        # Show registry list
        response = await handler.execute_command("skills", "registry")
        assert "Configured registries:" in response
        assert "[1]" in response
        assert "[2]" in response

        # Select by number
        response = await handler.execute_command("skills", "registry 2")
        assert "Registry set to" in response
        assert get_settings().skills.marketplace_url == marketplace2.as_posix()
        # Configured list is preserved
        marketplace_urls = get_settings().skills.marketplace_urls or []
        assert len(marketplace_urls) == 2

        # Invalid number
        response = await handler.execute_command("skills", "registry 99")
        assert "Invalid registry number" in response
    finally:
        get_settings(config_path=str(Path(__file__).parent / "fastagent.config.yaml"))
