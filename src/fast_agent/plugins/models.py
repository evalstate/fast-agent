"""Data models for first-class fast-agent command plugins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from fast_agent.command_actions.models import PluginCommandActionSpec

DEFAULT_PLUGIN_REGISTRIES = [
    "https://github.com/fast-agent-ai/card-packs",
]

DEFAULT_PLUGIN_MARKETPLACE_URL = (
    "https://github.com/fast-agent-ai/card-packs/blob/main/marketplace.json"
)

PLUGIN_MANIFEST_FILENAME = "plugin.yaml"
PLUGIN_SOURCE_FILENAME = ".plugin-source.json"
PLUGIN_SOURCE_SCHEMA_VERSION = 1
LOCAL_REVISION = "local"

PluginSourceOrigin = Literal["remote", "local"]
PluginUpdateStatus = Literal[
    "up_to_date",
    "update_available",
    "updated",
    "unmanaged",
    "invalid_metadata",
    "invalid_local_plugin",
    "unknown_revision",
    "source_unreachable",
    "source_ref_missing",
    "source_path_missing",
    "skipped_dirty",
]


@dataclass(frozen=True)
class PluginManifest:
    schema_version: int
    name: str
    version: str | None
    description: str | None
    commands: dict[str, PluginCommandActionSpec]
    path: Path


@dataclass(frozen=True)
class MarketplacePlugin:
    name: str
    description: str | None
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None = None
    bundle_name: str | None = None

    @property
    def repo_subdir(self) -> str:
        path = PurePosixPath(self.repo_path)
        if path.name.lower() == PLUGIN_MANIFEST_FILENAME:
            return str(path.parent)
        return str(path)

    @property
    def install_dir_name(self) -> str:
        path = PurePosixPath(self.repo_path)
        if path.name.lower() == PLUGIN_MANIFEST_FILENAME:
            return path.parent.name or self.name
        return path.name or self.name


@dataclass(frozen=True)
class InstalledPluginSource:
    schema_version: int
    installed_via: str
    source_origin: PluginSourceOrigin
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
class LocalPlugin:
    index: int
    name: str
    plugin_dir: Path
    manifest: PluginManifest | None
    source: InstalledPluginSource | None
    metadata_error: str | None = None
    manifest_error: str | None = None


@dataclass(frozen=True)
class PluginUpdateInfo:
    index: int
    name: str
    plugin_dir: Path
    status: PluginUpdateStatus
    detail: str | None = None
    current_revision: str | None = None
    available_revision: str | None = None
    managed_source: InstalledPluginSource | None = None
