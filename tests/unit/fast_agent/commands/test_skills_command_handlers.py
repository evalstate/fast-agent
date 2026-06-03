from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from rich.text import Text

from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers import skills as skills_handlers
from fast_agent.commands.results import CommandMessage
from fast_agent.config import get_settings
from fast_agent.skills.models import MarketplaceSkill
from fast_agent.skills.registry import SkillManifest


class _CommandIO(NonInteractiveCommandIOBase):
    async def emit(self, message: CommandMessage) -> None:
        del message


@pytest.mark.asyncio
async def test_skills_registry_rejects_empty_marketplace_without_switching(
    tmp_path: Path,
) -> None:
    empty_marketplace = tmp_path / "empty-marketplace.json"
    empty_marketplace.write_text('{"plugins": []}', encoding="utf-8")

    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "default_model: passthrough\n"
        "skills:\n"
        "  marketplace_url: https://example.com/current-marketplace.json\n",
        encoding="utf-8",
    )

    settings = get_settings(config_path=str(config_path))
    try:
        ctx = CommandContext(
            agent_provider=StaticAgentProvider(),
            current_agent_name="main",
            io=_CommandIO(),
            settings=settings,
        )

        outcome = await skills_handlers.handle_set_skills_registry(
            ctx,
            argument=empty_marketplace.as_posix(),
        )

        assert settings.skills.marketplace_url == "https://example.com/current-marketplace.json"
        assert outcome.messages[0].channel == "warning"
        message_text = outcome.messages[0].text
        assert isinstance(message_text, Text)
        assert "registry unchanged" in message_text.plain
    finally:
        get_settings(config_path=str(Path(__file__).parent / "fastagent.config.yaml"))


def test_format_skill_selection_list_uses_manifest_entry_formatting(tmp_path: Path) -> None:
    manifest = SkillManifest(
        name="alpha",
        description="Alpha skill",
        body="Body",
        path=tmp_path / "alpha" / "SKILL.md",
    )

    rendered = skills_handlers._format_skill_selection_list(
        [manifest],
        skills_dir=tmp_path,
    ).plain

    assert "Skills in " in rendered
    assert "[ 1] alpha" in rendered
    assert "Alpha skill" in rendered
    assert "source:" in rendered


def test_get_agent_skill_override_sources_deduplicates_preserving_manifest_order(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "z-source"
    second_source = tmp_path / "a-source"
    first_manifest_path = first_source / "one" / "SKILL.md"
    second_manifest_path = second_source / "two" / "SKILL.md"
    first_manifest_path.parent.mkdir(parents=True)
    second_manifest_path.parent.mkdir(parents=True)
    first_manifest_path.write_text("one", encoding="utf-8")
    second_manifest_path.write_text("two", encoding="utf-8")
    manifests = [
        SkillManifest(
            name="one",
            description="One",
            body="Body",
            path=first_manifest_path,
        ),
        SkillManifest(
            name="two",
            description="Two",
            body="Body",
            path=second_manifest_path,
        ),
        SkillManifest(
            name="three",
            description="Three",
            body="Body",
            path=first_manifest_path,
        ),
    ]

    assert skills_handlers._get_agent_skill_override_sources(manifests) == [
        str(first_source / "one"),
        str(second_source / "two"),
    ]


@pytest.mark.asyncio
async def test_list_marketplace_skills_trims_search_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marketplace = [
        MarketplaceSkill(
            name="docker-helper",
            description="Manage docker containers",
            repo_url="/repo",
            repo_ref=None,
            repo_path="skills/docker-helper",
        ),
        MarketplaceSkill(
            name="pdf-reader",
            description="Read PDF files",
            repo_url="/repo",
            repo_ref=None,
            repo_path="skills/pdf-reader",
        ),
    ]

    async def fetch_marketplace(_url: str):
        return marketplace

    monkeypatch.setattr(skills_handlers, "fetch_marketplace_skills", fetch_marketplace)
    ctx = cast("CommandContext", SimpleNamespace(resolve_settings=lambda: object()))

    outcome = await skills_handlers.handle_list_marketplace_skills(
        ctx,
        agent_name="main",
        query=" docker containers ",
        marketplace_url_override="memory://skills",
    )

    rendered = outcome.messages[0].text
    assert isinstance(rendered, Text)
    assert "Marketplace skills (search: docker containers):" in rendered.plain
    assert "docker-helper" in rendered.plain
    assert "pdf-reader" not in rendered.plain


@pytest.mark.asyncio
async def test_remove_skill_uses_skills_dir_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    override_dir = tmp_path / "managed-skills"
    captured: dict[str, object] = {}

    def resolve_scope(_settings: object, *, managed_directory_override: object = None):
        captured["override"] = managed_directory_override
        return SimpleNamespace(managed_directory=override_dir)

    monkeypatch.setattr(skills_handlers, "resolve_skills_management_scope", resolve_scope)
    monkeypatch.setattr(skills_handlers.SkillRegistry, "load_directory", lambda _path: [])

    ctx = cast("CommandContext", SimpleNamespace(resolve_settings=lambda: object()))
    outcome = await skills_handlers.handle_remove_skill(
        ctx,
        agent_name="main",
        argument=f'alpha --skills-dir="{override_dir}"',
    )

    assert captured["override"] == override_dir
    assert outcome.messages[0].text == "No local skills to remove."


@pytest.mark.asyncio
async def test_update_skill_uses_skills_dir_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    override_dir = tmp_path / "managed-skills"
    captured: dict[str, object] = {}

    def resolve_scope(_settings: object, *, managed_directory_override: object = None):
        captured["override"] = managed_directory_override
        return SimpleNamespace(managed_directory=override_dir)

    def check_updates(*, destination_root: Path):
        captured["destination_root"] = destination_root
        return []

    monkeypatch.setattr(skills_handlers, "resolve_skills_management_scope", resolve_scope)
    monkeypatch.setattr(skills_handlers, "check_skill_updates", check_updates)

    ctx = cast("CommandContext", SimpleNamespace(resolve_settings=lambda: object()))
    outcome = await skills_handlers.handle_update_skill(
        ctx,
        agent_name="main",
        argument=f'--skills-dir="{override_dir}"',
    )

    assert captured == {
        "override": override_dir,
        "destination_root": override_dir,
    }
    assert outcome.messages[0].right_info == "skills"
