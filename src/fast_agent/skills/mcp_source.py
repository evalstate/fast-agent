"""MCP registry-backed skill install source."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.skills.mcp_registry import (
    McpRegistrySkill,
    McpSkillInstallClient,
    McpSkillRegistry,
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
    def __init__(
        self,
        *,
        aggregator: McpSkillInstallClient,
        registry: McpSkillRegistry,
    ) -> None:
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
        normalized_query = strip_to_none(query)
        if normalized_query is None:
            return list(self._registry.skills)
        query_lower = normalized_query.lower()
        return [
            skill
            for skill in self._registry.skills
            if query_lower in skill.name.lower() or query_lower in (skill.description or "").lower()
        ]

    async def select_skill(self, selector: str) -> SkillCatalogEntry | None:
        return select_mcp_registry_skill(self._registry.skills, selector)

    async def install_skill(
        self,
        selector: str,
        *,
        destination_root: Path,
    ) -> SkillInstallResult:
        skill = select_mcp_registry_skill(self._registry.skills, selector)
        if skill is None:
            raise LookupError(f"Skill not found: {selector}")
        install_dir = await install_mcp_registry_skill(
            self._aggregator,
            skill,
            destination_root=destination_root,
        )
        return SkillInstallResult(name=skill.name, skill_dir=install_dir)

    async def check_updates(self, updates: Sequence[SkillUpdateInfo]) -> list[SkillUpdateInfo]:
        checked: list[SkillUpdateInfo] = []
        for update in updates:
            source = update.managed_source
            if source is None:
                checked.append(update)
                continue

            skill = _find_mcp_skill_for_update(self._registry, update)
            if skill is None:
                checked.append(_missing_registry_entry_update(update))
                continue

            status = "up_to_date"
            detail = "already up to date"
            if skill.digest != source.artifact_digest:
                status = "update_available"
                detail = "MCP skill artifact changed"
            checked.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status=status,
                    detail=detail,
                    current_revision=source.artifact_digest or source.installed_revision,
                    available_revision=skill.digest,
                    managed_source=source,
                )
            )
        return checked

    async def apply_updates(
        self,
        updates: Sequence[SkillUpdateInfo],
        *,
        force: bool,
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

            skill = _find_mcp_skill_for_update(self._registry, update)
            if skill is None:
                results.append(_missing_registry_entry_update(update))
                continue

            fingerprint = compute_skill_content_fingerprint(update.skill_dir)
            if fingerprint != source.content_fingerprint and not force:
                results.append(
                    SkillUpdateInfo(
                        index=update.index,
                        name=update.name,
                        skill_dir=update.skill_dir,
                        status="skipped_dirty",
                        detail="local modifications detected; rerun with --force",
                        current_revision=source.installed_revision,
                        available_revision=skill.digest,
                        managed_source=source,
                    )
                )
                continue

            try:
                await update_mcp_registry_skill(
                    self._aggregator,
                    skill,
                    skill_dir=update.skill_dir,
                )
                installed_source = read_installed_skill_source(update.skill_dir).source
            except Exception as exc:
                results.append(
                    SkillUpdateInfo(
                        index=update.index,
                        name=update.name,
                        skill_dir=update.skill_dir,
                        status="source_unreachable",
                        detail=str(exc),
                        current_revision=source.installed_revision,
                        available_revision=skill.digest,
                        managed_source=source,
                    )
                )
                continue

            results.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status="updated",
                    detail="updated",
                    current_revision=source.installed_revision,
                    available_revision=skill.digest,
                    managed_source=installed_source or source,
                )
            )
        return results

    def list_heading(self, *, query: str | None = None) -> str:
        normalized_query = strip_to_none(query)
        if normalized_query is None:
            return f"MCP skills from {self._registry.display_name}:"
        return f"MCP skills from {self._registry.display_name} (search: {normalized_query}):"

    def empty_message(self) -> str:
        return "No skills found in the MCP registry."

    def selection_options(self, entries: Sequence[SkillCatalogEntry]) -> list[str]:
        return [entry.name for entry in entries]

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

    async def install_skill(
        self,
        selector: str,
        *,
        destination_root: Path,
    ) -> SkillInstallResult:
        del selector, destination_root
        raise RuntimeError(self._detail)

    async def check_updates(self, updates: Sequence[SkillUpdateInfo]) -> list[SkillUpdateInfo]:
        return [_unreachable_update(update, detail=self._detail) for update in updates]

    async def apply_updates(
        self,
        updates: Sequence[SkillUpdateInfo],
        *,
        force: bool,
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


def _find_mcp_skill_for_update(
    registry: McpSkillRegistry,
    update: SkillUpdateInfo,
) -> McpRegistrySkill | None:
    source = update.managed_source
    if source is None or source.mcp_server_name != registry.server_name:
        return None
    for skill in registry.skills:
        if skill.source_url == source.source_url or skill.name == update.name:
            return skill
    return None


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
