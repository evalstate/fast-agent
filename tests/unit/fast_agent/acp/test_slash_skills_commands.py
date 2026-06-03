from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from acp.schema import ToolCallProgress, ToolCallStart

from fast_agent.acp.slash.handlers import skills as skills_handler_module
from fast_agent.acp.slash.handlers.skills import (
    handle_skills,
    handle_skills_add,
    handle_skills_available,
    handle_skills_registry,
    handle_skills_remove,
    send_skills_update,
)
from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.commands.command_catalog import command_action_names
from fast_agent.config import get_settings
from fast_agent.core.fastagent import AgentInstance
from fast_agent.skills.models import MarketplaceSkill
from fast_agent.skills.registry import SkillManifest

if TYPE_CHECKING:
    from fast_agent.acp.acp_aware_mixin import ACPCommand, ACPModeInfo
    from fast_agent.acp.acp_context import ACPContext
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol


class _RecordingACPContext:
    def __init__(self) -> None:
        self.updates: list[object] = []

    async def send_session_update(self, update: object) -> None:
        self.updates.append(update)


class _ACPAwareAgent:
    def __init__(self, acp: _RecordingACPContext) -> None:
        self._acp = acp

    @property
    def acp(self) -> "ACPContext":
        return cast("ACPContext", self._acp)

    @property
    def is_acp_mode(self) -> bool:
        return True

    @property
    def acp_commands(self) -> dict[str, "ACPCommand"]:
        return {}

    def acp_mode_info(self) -> "ACPModeInfo | None":
        return None


class _SkillsAddHandler:
    current_agent_name = "main"

    def __init__(self, agent: _ACPAwareAgent) -> None:
        self.agent = agent

    def _get_current_agent_or_error(self, heading: str) -> tuple[_ACPAwareAgent, None]:
        del heading
        return self.agent, None


class _App:
    pass


class _SkillsOverrideHandler:
    def __init__(self, agent: object) -> None:
        self.agent = agent

    def _get_current_agent(self) -> object:
        return self.agent


def test_acp_skills_handlers_cover_catalog_actions() -> None:
    assert set(skills_handler_module._SKILLS_ACTION_HANDLERS) == set(
        command_action_names("skills")
    )


def test_skills_override_section_preserves_first_source_order(tmp_path: Path) -> None:
    first_path = tmp_path / "z-source" / "one" / "SKILL.md"
    second_path = tmp_path / "a-source" / "two" / "SKILL.md"
    first_path.parent.mkdir(parents=True)
    second_path.parent.mkdir(parents=True)
    first_path.write_text("one", encoding="utf-8")
    second_path.write_text("two", encoding="utf-8")
    config = SimpleNamespace(
        skills=["configured"],
        skill_manifests=[
            SkillManifest(name="one", description="", body="", path=first_path),
            SkillManifest(name="two", description="", body="", path=second_path),
            SkillManifest(name="one-again", description="", body="", path=first_path),
        ],
    )

    rendered = skills_handler_module.skills_override_section(
        cast("SlashCommandHandler", _SkillsOverrideHandler(SimpleNamespace(config=config)))
    )

    assert rendered is not None
    sources_line = next(line for line in rendered.splitlines() if line.startswith("Sources:"))
    assert sources_line.index("z-source/one") < sources_line.index("a-source/two")
    assert sources_line.count("z-source/one") == 1


@pytest.mark.asyncio
async def test_handle_skills_accepts_help_alias() -> None:
    message = await handle_skills(cast("SlashCommandHandler", None), "--help")

    assert "Usage: /skills [list|available|search|add|remove|update|registry|help]" in message
    assert "- /skills registry" in message


@pytest.mark.asyncio
async def test_handle_skills_accepts_available_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_urls: list[str] = []

    async def fetch_marketplace(url: str) -> list[MarketplaceSkill]:
        called_urls.append(url)
        return []

    monkeypatch.setattr(skills_handler_module, "fetch_marketplace_skills", fetch_marketplace)

    message = await handle_skills(cast("SlashCommandHandler", None), "marketplace")

    assert called_urls
    assert message == "# skills available\n\nNo skills found in the marketplace."


@pytest.mark.asyncio
async def test_handle_skills_unknown_action_uses_catalog_message() -> None:
    message = await handle_skills(cast("SlashCommandHandler", None), "availabl value")

    assert message.startswith("Unknown /skills action: availabl. Use ")
    assert "list/available/search/add/remove/update/registry/help" in message
    assert "Did you mean: `available`" in message


@pytest.mark.asyncio
async def test_handle_skills_search_escapes_backticks_in_no_match_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fetch_marketplace(_url: str) -> list[MarketplaceSkill]:
        return [
            MarketplaceSkill(
                name="alpha",
                description=None,
                repo_url="https://github.com/example/skills",
                repo_ref="main",
                repo_path="skills/alpha",
            )
        ]

    monkeypatch.setattr(skills_handler_module, "fetch_marketplace_skills", fetch_marketplace)

    message = await handle_skills_available(
        cast("SlashCommandHandler", None),
        query="missing`query",
    )

    assert "No skills matched query `` missing`query ``." in message


@pytest.mark.asyncio
async def test_handle_skills_search_consumes_registry_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_url: str | None = None

    async def fetch_marketplace(url: str) -> list[MarketplaceSkill]:
        nonlocal captured_url
        captured_url = url
        return [
            MarketplaceSkill(
                name="alpha",
                description=None,
                repo_url="https://github.com/example/skills",
                repo_ref="main",
                repo_path="skills/alpha",
            )
        ]

    monkeypatch.setattr(skills_handler_module, "fetch_marketplace_skills", fetch_marketplace)

    message = await handle_skills(
        cast("SlashCommandHandler", None),
        "search alpha --registry ./marketplace.json",
    )

    assert captured_url == "./marketplace.json"
    assert "alpha" in message


@pytest.mark.asyncio
async def test_handle_skills_registry_uses_shared_registry_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    first_registry = "https://example.com/first-marketplace.json"
    second_registry = "https://example.com/second-marketplace.json"
    resolved_registry = "https://example.com/resolved-marketplace.json"
    config_path.write_text(
        "default_model: passthrough\n"
        "skills:\n"
        f"  marketplace_url: {first_registry}\n"
        "  marketplace_urls:\n"
        f"    - {first_registry}\n"
        f"    - {second_registry}\n",
        encoding="utf-8",
    )
    settings = get_settings(config_path=str(config_path))
    fetched_urls: list[str] = []

    async def fetch_marketplace(url: str):
        fetched_urls.append(url)
        return (
            [
                MarketplaceSkill(
                    name="alpha",
                    description=None,
                    repo_url="https://github.com/example/skills",
                    repo_ref="main",
                    repo_path="skills/alpha",
                )
            ],
            resolved_registry,
        )

    monkeypatch.setattr(
        skills_handler_module.skills_handlers,
        "fetch_marketplace_skills_with_source",
        fetch_marketplace,
    )
    handler = SlashCommandHandler(
        session_id="s1",
        instance=AgentInstance(
            app=cast("AgentApp", _App()),
            agents={},
            registry_version=0,
        ),
        primary_agent_name="main",
    )

    try:
        message = await handle_skills_registry(handler, "2")
    finally:
        get_settings(config_path=str(Path(__file__).parent / "fastagent.config.yaml"))

    assert fetched_urls == [second_registry]
    assert settings.skills.marketplace_url == resolved_registry
    assert "# skills registry" in message
    assert "Registry set to:" in message
    assert "Skills discovered: 1" in message


@pytest.mark.asyncio
async def test_bare_skills_add_browses_marketplace_without_install_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acp = _RecordingACPContext()
    handler = _SkillsAddHandler(_ACPAwareAgent(acp))

    async def render_marketplace(*, registry: str | None = None) -> str:
        del registry
        return "# skills add\n\nMarketplace"

    monkeypatch.setattr(
        skills_handler_module,
        "_render_skills_add_marketplace",
        render_marketplace,
    )

    message = await handle_skills_add(cast("SlashCommandHandler", handler), "")

    assert message == "# skills add\n\nMarketplace"
    assert acp.updates == []


@pytest.mark.asyncio
async def test_skills_add_and_remove_accept_cancel_aliases_without_agent_lookup() -> None:
    assert await handle_skills_add(cast("SlashCommandHandler", None), " QUIT ") == "Cancelled."
    assert await handle_skills_remove(cast("SlashCommandHandler", None), " exit ") == "Cancelled."


@pytest.mark.asyncio
async def test_send_skills_update_emits_start_and_progress() -> None:
    acp = _RecordingACPContext()
    agent = _ACPAwareAgent(acp)

    await send_skills_update(
        cast("SlashCommandHandler", None),
        cast("AgentProtocol", agent),
        "skills-1",
        title="Install skill",
        status="completed",
        message="Installed alpha",
        start=True,
    )

    assert len(acp.updates) == 2
    start, progress = acp.updates
    assert isinstance(start, ToolCallStart)
    assert start.tool_call_id == "skills-1"
    assert start.title == "Install skill"
    assert start.kind == "fetch"
    assert start.status == "in_progress"

    assert isinstance(progress, ToolCallProgress)
    assert progress.tool_call_id == "skills-1"
    assert progress.title == "Install skill"
    assert progress.status == "completed"
    assert progress.content is not None
    assert "Installed alpha" in str(progress.content)
