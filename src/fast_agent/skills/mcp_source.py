"""MCP resource-manifest-backed skill source."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from fast_agent.skills.mcp_registry import (
    McpRegistrySkill,
    McpSkillInstallClient,
    McpSkillRegistry,
    get_mcp_registry_skill,
    install_mcp_registry_skill,
    select_mcp_registry_skill,
    update_mcp_registry_skill,
)
from fast_agent.skills.models import SkillUpdateInfo
from fast_agent.skills.provenance import (
    compute_skill_content_fingerprint,
    read_installed_skill_source,
)
from fast_agent.skills.sources import SkillCatalogEntry, SkillInstallResult, SkillSourceRef
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class McpSkillSource:
    def __init__(self, *, aggregator: McpSkillInstallClient, registry: McpSkillRegistry) -> None:
        self._aggregator = aggregator
        self._registry = registry

    @property
    def ref(self) -> SkillSourceRef:
        return SkillSourceRef(
            kind="mcp",
            display_name=self._registry.display_name,
            server_name=self._registry.server_name,
        )

    async def list_skills(self, *, query: str | None = None) -> list[SkillCatalogEntry]:
        query = strip_to_none(query)
        if query is None:
            return list(self._registry.skills)
        needle = query.casefold()
        return [
            skill
            for skill in self._registry.skills
            if needle in skill.name.casefold() or needle in skill.description.casefold()
        ]

    async def select_skill(self, selector: str) -> SkillCatalogEntry | None:
        if _is_uri(selector):
            return await get_mcp_registry_skill(
                self._aggregator,
                selector.strip(),
                self._registry.server_name,
                server_version=self._registry.server_version,
            )
        return select_mcp_registry_skill(self._registry.skills, selector)

    async def install_skill(self, selector: str, *, destination_root: Path) -> SkillInstallResult:
        skill = await self.select_skill(selector)
        if skill is None or not isinstance(skill, McpRegistrySkill):
            raise LookupError(f"Skill not found: {selector}")
        skill_dir = await install_mcp_registry_skill(
            self._aggregator, skill, destination_root=destination_root
        )
        return SkillInstallResult(name=skill.name, skill_dir=skill_dir)

    async def check_updates(self, updates: Sequence[SkillUpdateInfo]) -> list[SkillUpdateInfo]:
        checked: list[SkillUpdateInfo] = []
        for update in updates:
            source = update.managed_source
            if source is None:
                checked.append(update)
                continue
            try:
                skill = await self._skill_for_update(update)
            except Exception as exc:
                checked.append(_unreachable_update(update, detail=str(exc)))
                continue
            if skill is None:
                checked.append(_missing_registry_entry_update(update))
                continue
            available = skill.revision
            status = "up_to_date" if available == source.installed_revision else "update_available"
            checked.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status=status,
                    detail="already up to date"
                    if status == "up_to_date"
                    else "MCP resource set changed",
                    current_revision=source.installed_revision,
                    available_revision=available,
                    managed_source=source,
                )
            )
        return checked

    async def apply_updates(
        self, updates: Sequence[SkillUpdateInfo], *, force: bool
    ) -> list[SkillUpdateInfo]:
        results: list[SkillUpdateInfo] = []
        for update in updates:
            source = update.managed_source
            if source is None:
                results.append(
                    SkillUpdateInfo(
                        index=update.index,
                        name=update.name,
                        skill_dir=update.skill_dir,
                        status="invalid_metadata",
                        detail="missing source metadata",
                    )
                )
                continue
            try:
                skill = await self._skill_for_update(update)
            except Exception as exc:
                results.append(_unreachable_update(update, detail=str(exc)))
                continue
            if skill is None:
                results.append(_missing_registry_entry_update(update))
                continue
            if skill.revision == source.installed_revision:
                results.append(
                    SkillUpdateInfo(
                        index=update.index,
                        name=update.name,
                        skill_dir=update.skill_dir,
                        status="up_to_date",
                        detail="already up to date",
                        current_revision=source.installed_revision,
                        available_revision=skill.revision,
                        managed_source=source,
                    )
                )
                continue
            if (
                compute_skill_content_fingerprint(update.skill_dir) != source.content_fingerprint
                and not force
            ):
                results.append(
                    SkillUpdateInfo(
                        index=update.index,
                        name=update.name,
                        skill_dir=update.skill_dir,
                        status="skipped_dirty",
                        detail="local modifications detected; rerun with --force",
                        current_revision=source.installed_revision,
                        available_revision=skill.revision,
                        managed_source=source,
                    )
                )
                continue
            try:
                await update_mcp_registry_skill(self._aggregator, skill, skill_dir=update.skill_dir)
                installed = read_installed_skill_source(update.skill_dir).source
            except Exception as exc:
                results.append(_unreachable_update(update, detail=str(exc)))
                continue
            results.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status="updated",
                    detail="updated",
                    current_revision=source.installed_revision,
                    available_revision=skill.revision,
                    managed_source=installed or source,
                )
            )
        return results

    async def _skill_for_update(self, update: SkillUpdateInfo) -> McpRegistrySkill | None:
        source = update.managed_source
        if source is None or source.mcp_server_name != self._registry.server_name:
            return None
        if source.source_url:
            return await get_mcp_registry_skill(
                self._aggregator,
                source.source_url,
                self._registry.server_name,
                server_version=self._registry.server_version,
            )
        return select_mcp_registry_skill(self._registry.skills, update.name)

    def list_heading(self, *, query: str | None = None) -> str:
        query = strip_to_none(query)
        suffix = "" if query is None else f" (search: {query})"
        return f"MCP skills from {self._registry.display_name}:{suffix}"

    def empty_message(self) -> str:
        return "No skills found in the MCP registry."

    def selection_options(self, entries: Sequence[SkillCatalogEntry]) -> list[str]:
        names = [entry.name.casefold() for entry in entries]
        return [
            entry.source_url
            if (
                (names.count(entry.name.casefold()) > 1 or entry.name.isdigit())
                and entry.source_url is not None
            )
            else entry.name
            for entry in entries
        ]

    def repository_hint(self, entries: Sequence[SkillCatalogEntry]) -> str | None:
        del entries
        return None


class UnavailableMcpSkillSource:
    def __init__(self, *, server_name: str, detail: str) -> None:
        self._server_name = server_name
        self._detail = detail

    @property
    def ref(self) -> SkillSourceRef:
        return SkillSourceRef(
            kind="mcp",
            display_name=f"mcp-server {self._server_name}",
            server_name=self._server_name,
        )

    async def list_skills(self, *, query: str | None = None) -> list[SkillCatalogEntry]:
        del query
        return []

    async def select_skill(self, selector: str) -> SkillCatalogEntry | None:
        del selector
        return None

    async def install_skill(self, selector: str, *, destination_root: Path) -> SkillInstallResult:
        del selector, destination_root
        raise RuntimeError(self._detail)

    async def check_updates(self, updates: Sequence[SkillUpdateInfo]) -> list[SkillUpdateInfo]:
        return [_unreachable_update(update, detail=self._detail) for update in updates]

    async def apply_updates(
        self, updates: Sequence[SkillUpdateInfo], *, force: bool
    ) -> list[SkillUpdateInfo]:
        del force
        return [_unreachable_update(update, detail=self._detail) for update in updates]

    def list_heading(self, *, query: str | None = None) -> str:
        del query
        return f"MCP skills from mcp-server {self._server_name}:"

    def empty_message(self) -> str:
        return self._detail

    def selection_options(self, entries: Sequence[SkillCatalogEntry]) -> list[str]:
        del entries
        return []

    def repository_hint(self, entries: Sequence[SkillCatalogEntry]) -> str | None:
        del entries
        return None


def _is_uri(value: str) -> bool:
    return bool(urlsplit(value.strip()).scheme)


def _missing_registry_entry_update(update: SkillUpdateInfo) -> SkillUpdateInfo:
    source = update.managed_source
    return SkillUpdateInfo(
        index=update.index,
        name=update.name,
        skill_dir=update.skill_dir,
        status="source_path_missing",
        detail="MCP registry entry not found",
        current_revision=source.installed_revision if source else update.current_revision,
        available_revision=source.installed_revision if source else update.available_revision,
        managed_source=source,
    )


def _unreachable_update(update: SkillUpdateInfo, *, detail: str) -> SkillUpdateInfo:
    return SkillUpdateInfo(
        index=update.index,
        name=update.name,
        skill_dir=update.skill_dir,
        status="source_unreachable",
        detail=detail,
        current_revision=update.current_revision,
        available_revision=update.available_revision,
        managed_source=update.managed_source,
    )
