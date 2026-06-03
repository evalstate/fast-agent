"""Data models for first-class fast-agent command plugins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from fast_agent.marketplace.source_utils import repo_subdir_for_manifest_path
from fast_agent.marketplace.update_status import CommonMarketplaceUpdateStatus
from fast_agent.utils.text import strip_casefold

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
PluginUpdateStatus = CommonMarketplaceUpdateStatus | Literal["invalid_local_plugin"]


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
        return repo_subdir_for_manifest_path(self.repo_path, PLUGIN_MANIFEST_FILENAME)

    @property
    def install_dir_name(self) -> str:
        path = PurePosixPath(self.repo_path)
        if strip_casefold(path.name) == PLUGIN_MANIFEST_FILENAME:
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
