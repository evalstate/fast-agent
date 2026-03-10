"""Core data models and constants for skills management.

These types are intentionally lightweight and dependency-minimal so they can
serve as the stable boundary for future extraction work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

DEFAULT_SKILL_REGISTRIES = [
    "https://github.com/fast-agent-ai/skills",
    "https://github.com/huggingface/skills",
    "https://github.com/anthropics/skills",
]

DEFAULT_MARKETPLACE_URL = (
    "https://github.com/fast-agent-ai/skills/blob/main/marketplace.json"
)

SKILL_SOURCE_FILENAME = ".skill-source.json"
SKILL_SOURCE_SCHEMA_VERSION = 1
LOCAL_REVISION = "local"

SkillSourceOrigin = Literal["remote", "local"]
SkillUpdateStatus = Literal[
    "up_to_date",
    "update_available",
    "updated",
    "unmanaged",
    "invalid_metadata",
    "invalid_local_skill",
    "unknown_revision",
    "source_unreachable",
    "source_ref_missing",
    "source_path_missing",
    "skipped_dirty",
]
SkillManagementSource = Literal["override", "settings", "default"]


@dataclass(frozen=True)
class InstalledSkillSource:
    schema_version: int
    installed_via: str
    source_origin: SkillSourceOrigin
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None
    installed_commit: str | None
    installed_path_oid: str | None
    installed_revision: str
    installed_at: str
    content_fingerprint: str


@dataclass(frozen=True)
class SkillProvenance:
    status: Literal["managed", "unmanaged", "invalid_metadata"]
    summary: str
    source: InstalledSkillSource | None = None
    error: str | None = None


@dataclass(frozen=True)
class SkillUpdateInfo:
    index: int
    name: str
    skill_dir: Path
    status: SkillUpdateStatus
    detail: str | None = None
    current_revision: str | None = None
    available_revision: str | None = None
    managed_source: InstalledSkillSource | None = None


@dataclass(frozen=True)
class SkillsManagementScope:
    """Resolved skills discovery and management directories for the current context."""

    managed_directory: Path
    discovered_directories: list[Path]
    management_source: SkillManagementSource


@dataclass(frozen=True)
class MarketplaceSkill:
    name: str
    description: str | None
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None = None
    bundle_name: str | None = None
    bundle_description: str | None = None

    @property
    def repo_subdir(self) -> str:
        path = PurePosixPath(self.repo_path)
        if path.name.lower() == "skill.md":
            return str(path.parent)
        return str(path)

    @property
    def install_dir_name(self) -> str:
        path = PurePosixPath(self.repo_path)
        if path.name.lower() == "skill.md":
            return path.parent.name or self.name
        return path.name or self.name
