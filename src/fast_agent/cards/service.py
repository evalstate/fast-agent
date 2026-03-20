"""Small service-facing API for card pack management.

This module provides a stable, integration-friendly surface that works with
plain registry sources and managed environment roots without coupling callers
to CLI or slash-command presentation details.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent.cards import manager

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

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
    "EnsuredCardPack",
    "CardPackUpdatePlan",
    "CardPackUpdateResult",
    "MarketplaceScanResult",
    "apply_update_plan",
    "ensure_pack_available",
    "ensure_pack_available_sync",
    "check_updates",
    "install_pack",
    "install_pack_sync",
    "install_selected_pack",
    "list_installed_packs",
    "plan_updates",
    "publish_pack",
    "read_installed_pack_readme",
    "resolve_registry",
    "remove_pack",
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
) -> CardPackInstallRecord:
    install_result = await manager.install_marketplace_card_pack(
        pack,
        environment_paths=environment_paths,
        force=force,
    )
    return CardPackInstallRecord(
        pack=pack,
        install_result=install_result,
        readme=manager.load_card_pack_readme(install_result.pack_dir),
    )


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
    readmes = [
        _build_readme_record(update.name, update.pack_dir)
        for update in applied
        if update.status == "updated"
    ]
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


def _build_readme_record(pack_name: str, pack_dir: Path) -> CardPackReadmeRecord:
    return CardPackReadmeRecord(
        pack_name=pack_name,
        pack_dir=pack_dir,
        readme=manager.load_card_pack_readme(pack_dir),
    )
