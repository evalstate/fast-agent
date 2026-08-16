from __future__ import annotations

import json
from hashlib import sha256

import pytest

from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers import skills_registry as skills_registry_handlers
from fast_agent.commands.handlers.skills import (
    _format_install_result,
    handle_list_marketplace_skills,
    handle_set_skills_registry,
    handle_skills_command,
    handle_update_skill,
)
from fast_agent.config import Settings, SkillsSettings
from fast_agent.mcp.skills_extension import GetSkillResult, SkillEntry, SkillResource
from fast_agent.skills.mcp_registry import McpRegistrySkill, McpSkillRegistry
from fast_agent.skills.models import McpSkillResource
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
                        uri="skill://hub-search/SKILL.md",
                        server_name="hf",
                        server_version="1.2.3",
                        frontmatter={"name": "hub-search", "description": "Search"},
                        resources=(
                            SkillResource(
                                uri="skill://hub-search/SKILL.md",
                                digest=_digest(
                                    "---\nname: hub-search\ndescription: Search\n---\nv2\n"
                                ),
                            ),
                        ),
                    )
                ],
            )
        ]

    async def get_skill(self, uri: str, server_name: str) -> GetSkillResult:
        del server_name
        assert uri == "skill://hub-search/SKILL.md"
        return GetSkillResult(
            skill=SkillEntry(
                uri=uri,
                frontmatter={"name": "hub-search", "description": "Search"},
                resources=[
                    SkillResource(
                        uri=uri,
                        digest=_digest("---\nname: hub-search\ndescription: Search\n---\nv2\n"),
                    )
                ],
            )
        )


class _Agent:
    aggregator = _Aggregator()


class _ManyAggregator(_Aggregator):
    async def list_mcp_skill_registries(self) -> list[McpSkillRegistry]:
        return [
            McpSkillRegistry(
                server_name="hf",
                server_version="1.2.3",
                skills=[
                    McpRegistrySkill(
                        name=f"skill-{index:02}",
                        description=(f"Description for skill {index}. " * 30),
                        uri=f"skill://skill-{index:02}/SKILL.md",
                        server_name="hf",
                        server_version="1.2.3",
                    )
                    for index in range(1, 26)
                ],
            )
        ]


class _ManyAgent:
    aggregator = _ManyAggregator()


def _ctx(
    settings: Settings,
    *,
    agent_provider: StaticAgentProvider | None = None,
    acp_session_id: str | None = None,
) -> CommandContext:
    return CommandContext(
        agent_provider=agent_provider or StaticAgentProvider({"main": _Agent()}),
        current_agent_name="main",
        io=NonInteractiveCommandIOBase(),
        settings=settings,
        acp_session_id=acp_session_id,
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
async def test_skills_registry_uses_current_agent_for_mcp_servers() -> None:
    settings = Settings()
    outcome = await handle_set_skills_registry(_ctx(settings), agent_name=None, argument=None)

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "MCP registries:" in rendered
    assert "mcp-server hf@1.2.3" in rendered


@pytest.mark.asyncio
async def test_skills_registry_can_select_mcp_server_by_name() -> None:
    settings = Settings()
    ctx = _ctx(settings)

    outcome = await handle_set_skills_registry(ctx, agent_name="main", argument="hf")

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert settings.skills.marketplace_url is None
    assert ctx.active_skill_source("main") == "mcp://hf"
    assert "Registry set to: mcp-server hf@1.2.3" in rendered
    assert "Skills discovered: 1. Browse with /skills available." in rendered


@pytest.mark.asyncio
async def test_nonpersistent_registry_selection_does_not_mutate_settings() -> None:
    settings = Settings()
    overrides: dict[str, str] = {}
    ctx = CommandContext(
        agent_provider=StaticAgentProvider({"main": _Agent()}),
        current_agent_name="main",
        io=NonInteractiveCommandIOBase(),
        settings=settings,
        skill_source_overrides=overrides,
        persist_skill_source_overrides=False,
    )

    async def fetch_registry(url: str) -> tuple[list[object], str]:
        return [object()], url

    await skills_registry_handlers.handle_set_skills_registry(
        ctx,
        agent_name="main",
        argument="https://example.com/skills.json",
        fetch_skills_with_source=fetch_registry,
    )

    assert settings.skills.marketplace_url is None
    assert overrides == {"main": "https://example.com/skills.json"}


@pytest.mark.asyncio
async def test_skills_registry_filters_active_mcp_source_from_configured_numbers() -> None:
    settings = Settings(
        skills=SkillsSettings(
            marketplace_urls=["https://github.com/example/skills"],
        )
    )
    ctx = _ctx(settings)
    ctx.set_active_skill_source("main", "mcp://hf")

    list_outcome = await handle_set_skills_registry(ctx, agent_name="main", argument=None)
    rendered_list = "\n".join(_plain(message.text) for message in list_outcome.messages)

    assert "https://github.com/example/skills" in rendered_list
    assert "mcp://hf" not in rendered_list
    assert "MCP registries:" in rendered_list
    assert "mcp-server hf@1.2.3" in rendered_list

    select_outcome = await handle_set_skills_registry(ctx, agent_name="main", argument="2")
    rendered_select = "\n".join(_plain(message.text) for message in select_outcome.messages)

    assert settings.skills.marketplace_url is None
    assert ctx.active_skill_source("main") == "mcp://hf"
    assert "Registry set to: mcp-server hf@1.2.3" in rendered_select
    assert "Failed to load registry" not in rendered_select


@pytest.mark.asyncio
async def test_skills_available_uses_selected_mcp_registry() -> None:
    settings = Settings()
    ctx = _ctx(settings)
    await handle_set_skills_registry(ctx, agent_name="main", argument="hf")

    outcome = await handle_list_marketplace_skills(ctx, agent_name="main", query=None)

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "MCP skills from mcp-server hf@1.2.3" in rendered
    assert "hub-search" in rendered
    assert "integrity: SHA-256 manifest; checked on install" in rendered
    assert "They do not verify the server or publisher" in rendered


@pytest.mark.asyncio
async def test_skills_available_uses_one_shot_mcp_registry_without_changing_active_source() -> None:
    settings = Settings()
    ctx = _ctx(settings)

    outcome = await handle_list_marketplace_skills(
        ctx,
        agent_name="main",
        marketplace_url_override="hf",
        output="compact",
    )

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "Source: mcp-server hf@1.2.3" in rendered
    assert "hub-search" in rendered
    assert ctx.active_skill_source("main") is None
    assert settings.skills.marketplace_url is None


@pytest.mark.asyncio
async def test_model_skills_available_is_compact_complete_and_bounded() -> None:
    settings = Settings()
    ctx = _ctx(
        settings,
        agent_provider=StaticAgentProvider({"main": _ManyAgent()}),
    )

    outcome = await handle_skills_command(
        ctx,
        agent_name="main",
        action="available",
        argument="--registry hf",
        interactive=False,
    )

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "Showing 1-25 of 25 skills." in rendered
    assert "skill-01" in rendered
    assert "skill-25" in rendered
    assert "Install: /skills add <number|name> --registry mcp://hf" in rendered
    assert rendered.count("/skills add <number|name> --registry mcp://hf") == 1
    assert len(rendered) < 16_000
    assert ctx.active_skill_source("main") is None


@pytest.mark.asyncio
async def test_skills_available_json_is_paginated_and_machine_readable() -> None:
    ctx = _ctx(
        Settings(),
        agent_provider=StaticAgentProvider({"main": _ManyAgent()}),
    )

    outcome = await handle_skills_command(
        ctx,
        agent_name="main",
        action="available",
        argument="--registry hf --limit 10 --json",
        interactive=False,
    )

    assert outcome.direct_response is not None
    payload = json.loads(outcome.direct_response)
    assert payload["kind"] == "skill_catalog"
    assert payload["source"]["server_name"] == "hf"
    assert payload["total"] == 25
    assert payload["next_page"] == 2
    assert len(payload["skills"]) == 10
    assert payload["commands"]["install"] == "/skills add <number|name> --registry mcp://hf"
    assert "--page 2" in payload["commands"]["next_page"]


@pytest.mark.asyncio
async def test_skills_json_remains_structured_for_empty_and_error_results() -> None:
    ctx = _ctx(
        Settings(),
        agent_provider=StaticAgentProvider({"main": _ManyAgent()}),
    )

    empty = await handle_skills_command(
        ctx,
        agent_name="main",
        action="search",
        argument="missing --registry hf --json",
        interactive=False,
    )
    unavailable = await handle_skills_command(
        ctx,
        agent_name="main",
        action="available",
        argument="--registry mcp://missing --json",
        interactive=False,
    )
    invalid = await handle_skills_command(
        ctx,
        agent_name="main",
        action="available",
        argument="--page 0 --json",
        interactive=False,
    )
    unbounded = await handle_skills_command(
        ctx,
        agent_name="main",
        action="available",
        argument="--registry hf --full",
        interactive=False,
    )

    assert empty.direct_response is not None
    empty_payload = json.loads(empty.direct_response)
    assert empty_payload["skills"] == []
    assert empty_payload["next_page"] is None
    assert unavailable.direct_response is not None
    assert json.loads(unavailable.direct_response)["kind"] == "error"
    assert invalid.direct_response is not None
    assert json.loads(invalid.direct_response)["kind"] == "error"
    assert unbounded.direct_response is not None
    assert json.loads(unbounded.direct_response)["kind"] == "error"


@pytest.mark.asyncio
async def test_skills_search_accepts_one_shot_mcp_registry() -> None:
    ctx = _ctx(
        Settings(),
        agent_provider=StaticAgentProvider({"main": _ManyAgent()}),
    )

    outcome = await handle_skills_command(
        ctx,
        agent_name="main",
        action="search",
        argument="skill-12 --registry hf",
        interactive=False,
    )

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "12. skill-12" in rendered
    assert "skill-11" not in rendered


@pytest.mark.asyncio
async def test_model_skills_add_requires_selector_instead_of_listing_catalog() -> None:
    ctx = _ctx(Settings())

    outcome = await handle_skills_command(
        ctx,
        agent_name="main",
        action="add",
        argument="--registry hf",
        interactive=False,
    )

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "selector is required" in rendered
    assert "/skills available --registry hf" in rendered
    assert "MCP skills from" not in rendered


def test_mcp_install_result_qualifies_integrity_check(tmp_path) -> None:
    rendered = _format_install_result(
        "hub-search",
        tmp_path / "hub-search",
        mcp_integrity=True,
    ).plain

    assert "SHA-256 digests matched the server-supplied manifest" in rendered
    assert "does not authenticate the server or publisher" in rendered


@pytest.mark.asyncio
async def test_selected_mcp_registry_survives_fresh_command_context() -> None:
    settings = Settings()
    await handle_set_skills_registry(
        _ctx(settings, acp_session_id="skills-registry-test-session"),
        agent_name="main",
        argument="hf",
    )

    outcome = await handle_list_marketplace_skills(
        _ctx(settings, acp_session_id="skills-registry-test-session"),
        agent_name="main",
        query=None,
    )

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "MCP skills from mcp-server hf@1.2.3" in rendered
    assert "hub-search" in rendered


@pytest.mark.asyncio
async def test_skills_update_reports_mcp_digest_update_available(tmp_path) -> None:
    home = tmp_path / ".fast-agent"
    skill_dir = home / "skills" / "hub-search"
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
            resources=(
                McpSkillResource(uri="skill://hub-search/SKILL.md", digest=_digest(skill_text)),
            ),
            revision=_digest(skill_text),
        ),
    )
    settings = Settings(
        home=str(home),
        skills=SkillsSettings(),
    )

    outcome = await handle_update_skill(_ctx(settings), agent_name="main", argument=None)

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "hub-search" in rendered
    assert "update available" in rendered
