"""Install marketplace ``plugin_bundles`` (curated plugin sets) in one step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent.marketplace.provenance_io import resolve_managed_install_dir
from fast_agent.plugins.configuration import enable_plugin_in_config
from fast_agent.plugins.manifest import load_plugin_manifest
from fast_agent.plugins.operations import install_marketplace_plugin_sync

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from fast_agent.plugins.models import MarketplacePlugin

RECOMMENDED_BUNDLE = "recommended"


@dataclass(frozen=True, slots=True)
class BundlePluginResult:
    name: str
    plugin_dir: Path
    installed: bool
    """False when the plugin was already present and was only enabled."""


def bundle_members(plugins: Sequence[MarketplacePlugin], bundle: str) -> list[MarketplacePlugin]:
    return [plugin for plugin in plugins if bundle in plugin.bundles]


def install_plugin_bundle(
    plugins: Sequence[MarketplacePlugin],
    bundle: str,
    *,
    destination_root: Path,
    config_path: Path,
) -> list[BundlePluginResult]:
    """Install missing bundle members and enable every member in ``config_path``."""
    members = bundle_members(plugins, bundle)
    if not members:
        raise ValueError(f"Plugin bundle not found: {bundle}")
    results: list[BundlePluginResult] = []
    for plugin in members:
        plugin_dir = resolve_managed_install_dir(destination_root, plugin.install_dir_name)
        installed = not plugin_dir.exists()
        if installed:
            plugin_dir = install_marketplace_plugin_sync(plugin, destination_root=destination_root)
        name = load_plugin_manifest(plugin_dir).name
        enable_plugin_in_config(config_path, name)
        results.append(BundlePluginResult(name=name, plugin_dir=plugin_dir, installed=installed))
    return results
