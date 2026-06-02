from __future__ import annotations

from hashlib import sha256

import pytest

from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers.skills import (
    handle_list_marketplace_skills,
    handle_set_skills_registry,
    handle_update_skill,
)
from fast_agent.config import Settings, SkillsSettings
from fast_agent.skills.mcp_registry import McpRegistrySkill, McpSkillRegistry
from fast_agent.skills.provenance import (
    build_mcp_installed_skill_source,
    compute_skill_content_fingerprint,
    write_installed_skill_source,
)


def _digest(text: str) -> str:
    return f"sha256:{sha256(text.encode('utf-8')).hexdigest()}"


class _Aggregator:
    async def list_mcp_skill_registries(self) -> list[McpSkillRegistry]:
        return [
            McpSkillRegistry(
                server_name="hf",
                server_version="1.2.3",
                skills=[
                    McpRegistrySkill(
                        name="hub-search",
                        description="Search the Hub",
                        source_url="skill://hub-search/SKILL.md",
                        server_name="hf",
                        digest=_digest("---\nname: hub-search\ndescription: Search\n---\nv2\n"),
                        server_version="1.2.3",
                    )
                ],
            )
        ]


class _Agent:
    aggregator = _Aggregator()


def _ctx(settings: Settings) -> CommandContext:
    return CommandContext(
        agent_provider=StaticAgentProvider({"main": _Agent()}),
        current_agent_name="main",
        io=NonInteractiveCommandIOBase(),
        settings=settings,
    )


def _plain(message: object) -> str:
    value = getattr(message, "plain", None)
    return value if isinstance(value, str) else str(message)


@pytest.mark.asyncio
async def test_skills_registry_lists_mcp_servers() -> None:
    settings = Settings(
        skills=SkillsSettings(marketplace_urls=["https://github.com/example/skills"])
    )

    outcome = await handle_set_skills_registry(_ctx(settings), agent_name="main", argument=None)

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "https://github.com/example/skills" in rendered
    assert "MCP registries:" in rendered
    assert "mcp-server hf@1.2.3" in rendered


@pytest.mark.asyncio
async def test_skills_registry_can_select_mcp_server_by_name() -> None:
    settings = Settings()

    outcome = await handle_set_skills_registry(_ctx(settings), agent_name="main", argument="hf")

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert settings.skills.marketplace_url == "mcp://hf"
    assert "Registry set to: mcp-server hf@1.2.3" in rendered
    assert "Skills discovered: 1" in rendered


@pytest.mark.asyncio
async def test_skills_registry_filters_active_mcp_source_from_configured_numbers() -> None:
    settings = Settings(
        skills=SkillsSettings(
            marketplace_url="mcp://hf",
            marketplace_urls=["https://github.com/example/skills"],
        )
    )

    list_outcome = await handle_set_skills_registry(_ctx(settings), agent_name="main", argument=None)
    rendered_list = "\n".join(_plain(message.text) for message in list_outcome.messages)

    assert "https://github.com/example/skills" in rendered_list
    assert "mcp://hf" not in rendered_list
    assert "MCP registries:" in rendered_list
    assert "mcp-server hf@1.2.3" in rendered_list

    select_outcome = await handle_set_skills_registry(_ctx(settings), agent_name="main", argument="2")
    rendered_select = "\n".join(_plain(message.text) for message in select_outcome.messages)

    assert settings.skills.marketplace_url == "mcp://hf"
    assert "Registry set to: mcp-server hf@1.2.3" in rendered_select
    assert "Failed to load registry" not in rendered_select


@pytest.mark.asyncio
async def test_skills_available_uses_selected_mcp_registry() -> None:
    settings = Settings(skills=SkillsSettings(marketplace_url="mcp://hf"))

    outcome = await handle_list_marketplace_skills(
        _ctx(settings), agent_name="main", query=None
    )

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "MCP skills from mcp-server hf@1.2.3" in rendered
    assert "hub-search" in rendered


@pytest.mark.asyncio
async def test_skills_update_reports_mcp_digest_update_available(tmp_path) -> None:
    environment_dir = tmp_path / ".fast-agent"
    skill_dir = environment_dir / "skills" / "hub-search"
    skill_dir.mkdir(parents=True)
    skill_text = "---\nname: hub-search\ndescription: Search\n---\nv1\n"
    (skill_dir / "SKILL.md").write_text(skill_text, encoding="utf-8")
    fingerprint = compute_skill_content_fingerprint(skill_dir)
    write_installed_skill_source(
        skill_dir,
        build_mcp_installed_skill_source(
            server_name="hf",
            server_version="1.2.3",
            skill_uri="skill://hub-search/SKILL.md",
            fingerprint=fingerprint,
            artifact_digest=_digest(skill_text),
            artifact_type="skill-md",
        ),
    )
    settings = Settings(
        environment_dir=str(environment_dir),
        skills=SkillsSettings(),
    )

    outcome = await handle_update_skill(_ctx(settings), agent_name="main", argument=None)

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "hub-search" in rendered
    assert "update available" in rendered
