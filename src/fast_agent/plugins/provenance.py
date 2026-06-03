"""Plugin sidecar metadata helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fast_agent.marketplace import formatting as marketplace_formatting
from fast_agent.marketplace import source_utils as marketplace_source_utils
from fast_agent.plugins.models import (
    LOCAL_REVISION,
    PLUGIN_SOURCE_FILENAME,
    PLUGIN_SOURCE_SCHEMA_VERSION,
    InstalledPluginSource,
    MarketplacePlugin,
    PluginSourceOrigin,
)

if TYPE_CHECKING:
    from pathlib import Path


def get_plugin_source_sidecar_path(plugin_dir: Path) -> Path:
    return plugin_dir / PLUGIN_SOURCE_FILENAME


def compute_plugin_content_fingerprint(plugin_dir: Path) -> str:
    root = plugin_dir.resolve()
    return marketplace_source_utils.compute_directory_content_fingerprint(
        root,
        sidecar_path=get_plugin_source_sidecar_path(root),
        ignore_path=_is_generated_runtime_artifact,
    )


def _is_generated_runtime_artifact(path: Path) -> bool:
    if path.suffix == ".pyc":
        return True
    return "__pycache__" in path.parts


def read_installed_plugin_source(
    plugin_dir: Path,
) -> marketplace_source_utils.InstalledSourceReadResult[InstalledPluginSource]:
    return marketplace_source_utils.read_installed_source_file(
        get_plugin_source_sidecar_path(plugin_dir),
        parse_payload=parse_installed_plugin_source_payload,
    )


def write_installed_plugin_source(plugin_dir: Path, source: InstalledPluginSource) -> None:
    marketplace_source_utils.write_installed_source_file(
        get_plugin_source_sidecar_path(plugin_dir),
        source,
    )


def parse_installed_plugin_source_payload(payload: dict[str, Any]) -> InstalledPluginSource:
    parsed = marketplace_source_utils.parse_installed_source_fields(
        payload,
        expected_schema_version=PLUGIN_SOURCE_SCHEMA_VERSION,
        normalize_repo_path=normalize_repo_path,
    )
    return InstalledPluginSource(
        schema_version=PLUGIN_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin=parsed.source_origin,
        repo_url=parsed.repo_url,
        repo_ref=parsed.repo_ref,
        repo_path=parsed.repo_path,
        source_url=parsed.source_url,
        installed_commit=parsed.installed_commit,
        installed_path_oid=parsed.installed_path_oid,
        installed_revision=parsed.installed_revision,
        installed_at=parsed.installed_at,
        content_fingerprint=parsed.content_fingerprint,
    )


def build_installed_plugin_source(
    *,
    plugin: MarketplacePlugin,
    source_origin: PluginSourceOrigin,
    installed_commit: str | None,
    installed_path_oid: str | None,
    fingerprint: str,
) -> InstalledPluginSource:
    installed_revision = installed_commit or LOCAL_REVISION
    return InstalledPluginSource(
        schema_version=PLUGIN_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin=source_origin,
        repo_url=plugin.repo_url,
        repo_ref=plugin.repo_ref,
        repo_path=plugin.repo_subdir,
        source_url=plugin.source_url,
        installed_commit=installed_commit,
        installed_path_oid=installed_path_oid,
        installed_revision=installed_revision,
        installed_at=marketplace_formatting.iso_utc_now(),
        content_fingerprint=fingerprint,
    )


def normalize_repo_path(path: str) -> str | None:
    return marketplace_source_utils.normalize_relative_repo_path(path)


def format_revision_short(revision: str | None) -> str:
    return marketplace_formatting.format_revision_short(revision)


def format_installed_at_display(installed_at: str | None) -> str:
    return marketplace_formatting.format_installed_at_display(installed_at)
