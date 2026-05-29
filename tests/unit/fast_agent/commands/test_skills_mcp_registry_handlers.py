from __future__ import annotations

import pytest

from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers.skills import (
    handle_list_marketplace_skills,
    handle_set_skills_registry,
)
from fast_agent.config import Settings, SkillsSettings
from fast_agent.skills.mcp_registry import McpRegistrySkill, McpSkillRegistry


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
async def test_skills_available_uses_selected_mcp_registry() -> None:
    settings = Settings(skills=SkillsSettings(marketplace_url="mcp://hf"))

    outcome = await handle_list_marketplace_skills(
        _ctx(settings), agent_name="main", query=None
    )

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "MCP skills from mcp-server hf@1.2.3" in rendered
    assert "hub-search" in rendered
