"""Small service-facing API for card pack management.

This module provides a stable, integration-friendly surface that works with
plain registry sources and managed environment roots without coupling callers
to CLI or slash-command presentation details.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.cards import manager
from fast_agent.config import get_settings
from fast_agent.home import PREFERRED_CONFIG_FILENAME
from fast_agent.marketplace.update_status import is_update_applied
from fast_agent.plugins import operations as plugin_ops
from fast_agent.plugins.configuration import enable_plugin_in_config, get_marketplace_url
from fast_agent.plugins.manifest import load_plugin_manifest
from fast_agent.plugins.models import PLUGIN_MANIFEST_FILENAME

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.cards.manager import (
        CardPackInstallResult,
        CardPackPublishResult,
        CardPackRemovalResult,
        CardPackUpdateInfo,
        LocalCardPack,
        MarketplaceCardPack,
    )
    from fast_agent.config import Settings
    from fast_agent.paths import EnvironmentPaths
    from fast_agent.plugins.models import MarketplacePlugin


class CardPackLookupError(LookupError):
    """Raised when a requested marketplace or local card pack cannot be resolved."""


@dataclass(frozen=True)
class MarketplaceScanResult:
    source: str
    packs: list[MarketplaceCardPack]


@dataclass(frozen=True)
class CardPackReadmeRecord:
    pack_name: str
    pack_dir: Path
    readme: str | None


@dataclass(frozen=True)
class CardPackInstallRecord:
    pack: MarketplaceCardPack
    install_result: CardPackInstallResult
    readme: str | None


@dataclass(frozen=True)
class EnsuredCardPack:
    name: str
    pack_dir: Path
    installed: bool
    install_record: CardPackInstallRecord | None = None


@dataclass(frozen=True)
class CardPackUpdatePlan:
    available: list[CardPackUpdateInfo]
    selected: list[CardPackUpdateInfo]


@dataclass(frozen=True)
class CardPackUpdateResult:
    applied: list[CardPackUpdateInfo]
    readmes: list[CardPackReadmeRecord]


__all__ = [
    "CardPackInstallRecord",
    "CardPackLookupError",
    "CardPackReadmeRecord",
    "CardPackUpdatePlan",
    "CardPackUpdateResult",
    "EnsuredCardPack",
    "MarketplaceScanResult",
    "apply_update_plan",
    "check_updates",
    "ensure_pack_available",
    "ensure_pack_available_sync",
    "install_pack",
    "install_pack_sync",
    "install_selected_pack",
    "is_card_pack_publish_failure",
    "is_card_pack_publish_success",
    "list_installed_packs",
    "plan_updates",
    "publish_pack",
    "read_installed_pack_readme",
    "remove_pack",
    "resolve_registry",
    "scan_marketplace",
    "scan_marketplace_sync",
    "select_installed_pack",
    "select_marketplace_pack",
]


async def scan_marketplace(source: str) -> MarketplaceScanResult:
    packs, resolved_source = await manager.fetch_marketplace_card_packs_with_source(source)
    return MarketplaceScanResult(source=resolved_source, packs=packs)


def scan_marketplace_sync(source: str) -> MarketplaceScanResult:
    return asyncio.run(scan_marketplace(source))


def resolve_registry(source: str | None = None, *, settings: Settings | None = None) -> str:
    return source or manager.get_marketplace_url(settings)


def list_installed_packs(*, environment_paths: EnvironmentPaths) -> list[LocalCardPack]:
    return manager.list_local_card_packs(environment_paths=environment_paths)


def select_marketplace_pack(
    packs: Sequence[MarketplaceCardPack],
    selector: str,
) -> MarketplaceCardPack:
    selected = manager.select_card_pack_by_name_or_index(list(packs), selector)
    if selected is None:
        raise CardPackLookupError(f"Card pack not found: {selector}")
    return selected


def select_installed_pack(
    *,
    environment_paths: EnvironmentPaths,
    selector: str,
) -> LocalCardPack:
    packs = list_installed_packs(environment_paths=environment_paths)
    selected = manager.select_installed_card_pack_by_name_or_index(packs, selector)
    if selected is None:
        raise CardPackLookupError(f"Card pack not found: {selector}")
    return selected


async def install_selected_pack(
    pack: MarketplaceCardPack,
    *,
    environment_paths: EnvironmentPaths,
    force: bool,
    marketplace_source: str | None = None,
) -> CardPackInstallRecord:
    install_pack = replace(pack, source_url=marketplace_source) if marketplace_source else pack
    install_result = await manager.install_marketplace_card_pack(
        install_pack,
        environment_paths=environment_paths,
        force=force,
    )
    try:
        await _ensure_required_pack_plugins(
            install_result.pack_dir,
            environment_paths=environment_paths,
            plugin_registry=marketplace_source or pack.source_url,
        )
    except Exception:
        manager.remove_local_card_pack(
            install_result.pack_dir.name,
            environment_paths=environment_paths,
        )
        raise
    return CardPackInstallRecord(
        pack=install_pack,
        install_result=install_result,
        readme=manager.load_card_pack_readme(install_result.pack_dir),
    )


async def _ensure_required_pack_plugins(
    pack_dir: Path,
    *,
    environment_paths: EnvironmentPaths,
    plugin_registry: str | None = None,
) -> None:
    manifest = manager.load_card_pack_manifest(pack_dir)
    if not manifest.plugins_required:
        return

    installed_names, missing_plugins = _required_plugin_install_state(
        manifest.plugins_required,
        environment_paths=environment_paths,
    )

    marketplace_plugins = []
    if missing_plugins:
        settings = get_settings()
        marketplace_url = plugin_registry or get_marketplace_url(settings)
        marketplace_plugins, _ = await plugin_ops.fetch_marketplace_plugins_with_source(
            marketplace_url
        )

    _enable_required_pack_plugins(
        manifest,
        installed_names=installed_names,
        marketplace_plugins=marketplace_plugins,
        environment_paths=environment_paths,
    )


def _ensure_required_pack_plugins_sync(
    pack_dir: Path,
    *,
    environment_paths: EnvironmentPaths,
    plugin_registry: str | None = None,
) -> None:
    manifest = manager.load_card_pack_manifest(pack_dir)
    if not manifest.plugins_required:
        return

    installed_names, missing_plugins = _required_plugin_install_state(
        manifest.plugins_required,
        environment_paths=environment_paths,
    )

    marketplace_plugins = []
    if missing_plugins:
        settings = get_settings()
        marketplace_url = plugin_registry or get_marketplace_url(settings)
        marketplace_plugins, _ = plugin_ops.fetch_marketplace_plugins_with_source_sync(
            marketplace_url
        )

    _enable_required_pack_plugins(
        manifest,
        installed_names=installed_names,
        marketplace_plugins=marketplace_plugins,
        environment_paths=environment_paths,
    )


def _enable_required_pack_plugins(
    manifest: manager.CardPackManifest,
    *,
    installed_names: set[str],
    marketplace_plugins: Sequence["MarketplacePlugin"],
    environment_paths: EnvironmentPaths,
) -> None:
    settings = get_settings()
    config_path = _resolve_config_path(settings, environment_paths)

    for plugin_name in manifest.plugins_required:
        enabled_name = _resolve_required_plugin_enabled_name(
            plugin_name,
            installed_names=installed_names,
            marketplace_plugins=marketplace_plugins,
            environment_paths=environment_paths,
        )
        enable_plugin_in_config(config_path, enabled_name)


def _resolve_required_plugin_enabled_name(
    plugin_name: str,
    *,
    installed_names: set[str],
    marketplace_plugins: Sequence["MarketplacePlugin"],
    environment_paths: EnvironmentPaths,
) -> str:
    if plugin_name in installed_names:
        return plugin_name

    selected = plugin_ops.select_plugin_by_name_or_index(marketplace_plugins, plugin_name)
    if selected is None:
        raise CardPackLookupError(f"Required plugin not found in plugin registry: {plugin_name}")

    plugin_dir = environment_paths.plugins / selected.install_dir_name
    if not (plugin_dir / PLUGIN_MANIFEST_FILENAME).is_file():
        plugin_dir = plugin_ops.install_marketplace_plugin_sync(
            selected,
            destination_root=environment_paths.plugins,
        )
    return load_plugin_manifest(plugin_dir).name


def _required_plugin_install_state(
    required_plugins: Sequence[str],
    *,
    environment_paths: EnvironmentPaths,
) -> tuple[set[str], list[str]]:
    installed = plugin_ops.list_local_plugins(destination_root=environment_paths.plugins)
    installed_names = {entry.name for entry in installed}
    missing_plugins = [name for name in required_plugins if name not in installed_names]
    return installed_names, missing_plugins


def _resolve_config_path(settings: Settings, environment_paths: EnvironmentPaths) -> Path:
    if settings._config_file:
        return Path(settings._config_file)
    return environment_paths.root / PREFERRED_CONFIG_FILENAME


async def install_pack(
    source: str,
    selector: str,
    *,
    environment_paths: EnvironmentPaths,
    force: bool,
) -> CardPackInstallRecord:
    marketplace = await scan_marketplace(source)
    selected = select_marketplace_pack(marketplace.packs, selector)
    return await install_selected_pack(
        selected,
        environment_paths=environment_paths,
        force=force,
        marketplace_source=marketplace.source,
    )


def install_pack_sync(
    source: str,
    selector: str,
    *,
    environment_paths: EnvironmentPaths,
    force: bool,
) -> CardPackInstallRecord:
    return asyncio.run(
        install_pack(
            source,
            selector,
            environment_paths=environment_paths,
            force=force,
        )
    )


async def ensure_pack_available(
    *,
    selector: str,
    environment_paths: EnvironmentPaths,
    registry: str | None = None,
    force: bool = False,
) -> EnsuredCardPack:
    try:
        installed_pack = select_installed_pack(
            environment_paths=environment_paths,
            selector=selector,
        )
    except CardPackLookupError:
        installed_pack = None

    if installed_pack is not None:
        return EnsuredCardPack(
            name=installed_pack.name,
            pack_dir=installed_pack.pack_dir,
            installed=False,
        )

    install_record = await install_pack(
        resolve_registry(registry),
        selector,
        environment_paths=environment_paths,
        force=force,
    )
    return EnsuredCardPack(
        name=install_record.pack.name,
        pack_dir=install_record.install_result.pack_dir,
        installed=True,
        install_record=install_record,
    )


def ensure_pack_available_sync(
    *,
    selector: str,
    environment_paths: EnvironmentPaths,
    registry: str | None = None,
    force: bool = False,
) -> EnsuredCardPack:
    return asyncio.run(
        ensure_pack_available(
            selector=selector,
            environment_paths=environment_paths,
            registry=registry,
            force=force,
        )
    )


def remove_pack(
    *,
    environment_paths: EnvironmentPaths,
    selector: str,
) -> CardPackRemovalResult:
    selected = select_installed_pack(environment_paths=environment_paths, selector=selector)
    return manager.remove_local_card_pack(
        selected.pack_dir.name,
        environment_paths=environment_paths,
    )


def read_installed_pack_readme(
    *,
    environment_paths: EnvironmentPaths,
    selector: str,
) -> CardPackReadmeRecord:
    selected = select_installed_pack(environment_paths=environment_paths, selector=selector)
    return _build_readme_record(selected.name, selected.pack_dir)


def check_updates(*, environment_paths: EnvironmentPaths) -> list[CardPackUpdateInfo]:
    return manager.check_card_pack_updates(environment_paths=environment_paths)


def plan_updates(
    *,
    environment_paths: EnvironmentPaths,
    selector: str,
) -> CardPackUpdatePlan:
    available = check_updates(environment_paths=environment_paths)
    selected = manager.select_card_pack_updates(available, selector)
    if not selected:
        raise CardPackLookupError(f"Card pack not found: {selector}")
    return CardPackUpdatePlan(available=available, selected=selected)


def apply_update_plan(
    selected: Sequence[CardPackUpdateInfo],
    *,
    environment_paths: EnvironmentPaths,
    force: bool,
) -> CardPackUpdateResult:
    applied = manager.apply_card_pack_updates(
        list(selected),
        environment_paths=environment_paths,
        force=force,
    )
    updated = [update for update in applied if is_update_applied(update.status)]
    for update in updated:
        plugin_registry = update.managed_source.source_url if update.managed_source else None
        _ensure_required_pack_plugins_sync(
            update.pack_dir,
            environment_paths=environment_paths,
            plugin_registry=plugin_registry,
        )
    readmes = [_build_readme_record(update.name, update.pack_dir) for update in updated]
    return CardPackUpdateResult(applied=applied, readmes=readmes)


def publish_pack(
    *,
    environment_paths: EnvironmentPaths,
    selector: str,
    push: bool,
    commit_message: str | None,
    temp_dir: Path | None,
    keep_temp: bool,
) -> CardPackPublishResult:
    selected = select_installed_pack(environment_paths=environment_paths, selector=selector)
    return manager.publish_local_card_pack(
        selected.pack_dir,
        environment_paths=environment_paths,
        push=push,
        commit_message=commit_message,
        temp_dir=temp_dir,
        keep_temp=keep_temp,
    )


def is_card_pack_publish_success(status: manager.CardPackPublishStatus) -> bool:
    return manager.is_card_pack_publish_success(status)


def is_card_pack_publish_failure(status: manager.CardPackPublishStatus) -> bool:
    return manager.is_card_pack_publish_failure(status)


def _build_readme_record(pack_name: str, pack_dir: Path) -> CardPackReadmeRecord:
    return CardPackReadmeRecord(
        pack_name=pack_name,
        pack_dir=pack_dir,
        readme=manager.load_card_pack_readme(pack_dir),
    )
