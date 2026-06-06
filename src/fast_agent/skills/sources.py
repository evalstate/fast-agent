"""Command-facing skill source contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from fast_agent.skills.models import SkillUpdateInfo

SkillSourceKind = Literal["marketplace", "mcp"]


@dataclass(frozen=True, slots=True)
class SkillSourceRef:
    kind: SkillSourceKind
    display_name: str
    source_url: str | None = None
    server_name: str | None = None


@dataclass(frozen=True, slots=True)
class SkillInstallResult:
    name: str
    skill_dir: Path


class SkillCatalogEntry(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str | None: ...

    @property
    def source_url(self) -> str | None: ...


class SkillInstallSource(Protocol):
    @property
    def ref(self) -> SkillSourceRef: ...

    async def list_skills(self, *, query: str | None = None) -> list[SkillCatalogEntry]: ...

    async def select_skill(self, selector: str) -> SkillCatalogEntry | None: ...

    async def install_skill(
        self,
        selector: str,
        *,
        destination_root: Path,
    ) -> SkillInstallResult: ...

    async def check_updates(
        self,
        updates: Sequence[SkillUpdateInfo],
    ) -> list[SkillUpdateInfo]: ...

    async def apply_updates(
        self,
        updates: Sequence[SkillUpdateInfo],
        *,
        force: bool,
    ) -> list[SkillUpdateInfo]: ...

    def list_heading(self, *, query: str | None = None) -> str: ...

    def empty_message(self) -> str: ...

    def selection_options(self, entries: Sequence[SkillCatalogEntry]) -> list[str]: ...

    def repository_hint(self, entries: Sequence[SkillCatalogEntry]) -> str | None: ...
