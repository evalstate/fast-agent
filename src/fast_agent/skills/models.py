"""Core data models and constants for skills management.

These types are intentionally lightweight and dependency-minimal so they can
serve as the stable boundary for future extraction work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from fast_agent.marketplace.source_utils import repo_subdir_for_manifest_path
from fast_agent.marketplace.update_status import CommonMarketplaceUpdateStatus
from fast_agent.utils.text import strip_casefold

DEFAULT_SKILL_REGISTRIES = [
    "https://github.com/fast-agent-ai/skills",
    "https://github.com/huggingface/skills",
    "https://github.com/anthropics/skills",
]

DEFAULT_MARKETPLACE_URL = (
    "https://github.com/fast-agent-ai/skills/blob/main/marketplace.json"
)

SKILL_SOURCE_FILENAME = ".skill-source.json"
SKILL_MANIFEST_FILENAME = "SKILL.md"
SKILL_MANIFEST_FILENAME_LOWER = strip_casefold(SKILL_MANIFEST_FILENAME)
SKILL_SOURCE_SCHEMA_VERSION = 1
LOCAL_REVISION = "local"

SkillSourceOrigin = Literal["remote", "local", "mcp"]
SkillUpdateStatus = CommonMarketplaceUpdateStatus | Literal["invalid_local_skill"]
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
    mcp_server_name: str | None = None
    mcp_server_version: str | None = None
    artifact_digest: str | None = None
    artifact_type: str | None = None


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
    install_dir_name_override: str | None = None

    @property
    def repo_subdir(self) -> str:
        return repo_subdir_for_manifest_path(self.repo_path, SKILL_MANIFEST_FILENAME)

    @property
    def install_dir_name(self) -> str:
        if self.install_dir_name_override:
            return self.install_dir_name_override
        path = PurePosixPath(self.repo_path)
        if strip_casefold(path.name) == SKILL_MANIFEST_FILENAME_LOWER:
            return path.parent.name or self.name
        return path.name or self.name
