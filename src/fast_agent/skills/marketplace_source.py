"""Marketplace-backed skill install source."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fast_agent.skills.command_support import (
    filter_marketplace_skills,
    marketplace_repository_hint,
)
from fast_agent.skills.direct_sources import is_direct_skill_source
from fast_agent.skills.models import MarketplaceSkill, SkillUpdateInfo
from fast_agent.skills.operations import (
    apply_skill_updates,
    fetch_marketplace_skills,
    install_marketplace_skill,
    select_skill_by_name_or_index,
)
from fast_agent.skills.service import install_direct_skill
from fast_agent.skills.sources import SkillCatalogEntry, SkillInstallResult, SkillSourceRef
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class MarketplaceSkillSource:
    def __init__(self, source_url: str) -> None:
        self._source_url = source_url

    @property
    def ref(self) -> SkillSourceRef:
        return SkillSourceRef(
            kind="marketplace",
            display_name=self._source_url,
            source_url=self._source_url,
        )

    async def list_skills(self, *, query: str | None = None) -> list[SkillCatalogEntry]:
        skills = await fetch_marketplace_skills(self._source_url)
        normalized_query = strip_to_none(query)
        if normalized_query is None:
            return list(skills)
        return list(filter_marketplace_skills(skills, normalized_query))

    async def select_skill(self, selector: str) -> SkillCatalogEntry | None:
        skills = await fetch_marketplace_skills(self._source_url)
        return select_skill_by_name_or_index(skills, selector)

    async def install_skill(
        self,
        selector: str,
        *,
        destination_root: Path,
    ) -> SkillInstallResult:
        if is_direct_skill_source(selector):
            installed = await install_direct_skill(selector, destination_root=destination_root)
            return SkillInstallResult(name=installed.name, skill_dir=installed.skill_dir)

        skills = await fetch_marketplace_skills(self._source_url)
        selected = select_skill_by_name_or_index(skills, selector)
        if selected is None:
            raise LookupError(f"Skill not found: {selector}")
        install_dir = await install_marketplace_skill(selected, destination_root=destination_root)
        return SkillInstallResult(name=selected.name, skill_dir=install_dir)

    async def check_updates(self, updates: Sequence[SkillUpdateInfo]) -> list[SkillUpdateInfo]:
        return list(updates)

    async def apply_updates(
        self,
        updates: Sequence[SkillUpdateInfo],
        *,
        force: bool,
    ) -> list[SkillUpdateInfo]:
        return await asyncio.to_thread(apply_skill_updates, list(updates), force=force)

    def list_heading(self, *, query: str | None = None) -> str:
        normalized_query = strip_to_none(query)
        if normalized_query is None:
            return "Marketplace skills:"
        return f"Marketplace skills (search: {normalized_query}):"

    def empty_message(self) -> str:
        return "No skills found in the marketplace."

    def selection_options(self, entries: Sequence[SkillCatalogEntry]) -> list[str]:
        options: list[str] = []
        for entry in entries:
            options.append(entry.name)
            if isinstance(entry, MarketplaceSkill):
                options.append(entry.install_dir_name)
        return options

    def repository_hint(self, entries: Sequence[SkillCatalogEntry]) -> str | None:
        marketplace_entries = [entry for entry in entries if isinstance(entry, MarketplaceSkill)]
        return marketplace_repository_hint(marketplace_entries)
