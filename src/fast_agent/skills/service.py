"""Small service-facing API for skills management.

This module provides a stable, integration-friendly surface that works with
plain registry sources (local paths or URLs) and managed destination roots.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent.skills import operations
from fast_agent.skills.provenance import get_skill_provenance
from fast_agent.skills.registry import SkillManifest, SkillRegistry

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.skills.models import MarketplaceSkill, SkillProvenance, SkillUpdateInfo


class SkillLookupError(LookupError):
    """Raised when a requested marketplace or local skill cannot be resolved."""


@dataclass(frozen=True)
class MarketplaceScanResult:
    source: str
    skills: list[MarketplaceSkill]


@dataclass(frozen=True)
class InstalledSkillRecord:
    name: str
    skill_dir: Path
    manifest: SkillManifest
    provenance: SkillProvenance


@dataclass(frozen=True)
class RemovedSkillRecord:
    name: str
    skill_dir: Path


__all__ = [
    "InstalledSkillRecord",
    "MarketplaceScanResult",
    "RemovedSkillRecord",
    "SkillLookupError",
    "apply_updates",
    "check_updates",
    "install_skill",
    "install_skill_sync",
    "list_installed_skills",
    "remove_skill",
    "scan_marketplace",
    "scan_marketplace_sync",
]


async def scan_marketplace(source: str) -> MarketplaceScanResult:
    skills, resolved_source = await operations.fetch_marketplace_skills_with_source(source)
    return MarketplaceScanResult(source=resolved_source, skills=skills)


def scan_marketplace_sync(source: str) -> MarketplaceScanResult:
    return asyncio.run(scan_marketplace(source))


def list_installed_skills(destination_root: Path) -> list[InstalledSkillRecord]:
    manifests = SkillRegistry.load_directory(destination_root)
    return [_record_from_manifest(manifest) for manifest in manifests]


async def install_skill(
    source: str,
    selector: str,
    *,
    destination_root: Path,
) -> InstalledSkillRecord:
    scan_result = await scan_marketplace(source)
    selected = operations.select_skill_by_name_or_index(scan_result.skills, selector)
    if selected is None:
        raise SkillLookupError(f"Skill not found in marketplace: {selector}")

    install_dir = await operations.install_marketplace_skill(
        selected,
        destination_root=destination_root,
    )
    try:
        return _record_from_install_dir(destination_root, install_dir)
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir)
        raise


def install_skill_sync(
    source: str,
    selector: str,
    *,
    destination_root: Path,
) -> InstalledSkillRecord:
    return asyncio.run(
        install_skill(
            source,
            selector,
            destination_root=destination_root,
        )
    )


def remove_skill(
    destination_root: Path,
    selector: str,
) -> RemovedSkillRecord:
    manifests = SkillRegistry.load_directory(destination_root)
    selected = operations.select_manifest_by_name_or_index(manifests, selector)
    if selected is None:
        raise SkillLookupError(f"Installed skill not found: {selector}")

    skill_dir = selected.path.parent if selected.path.is_file() else selected.path
    operations.remove_local_skill(skill_dir, destination_root=destination_root)
    return RemovedSkillRecord(name=selected.name, skill_dir=skill_dir)


def check_updates(destination_root: Path) -> list[SkillUpdateInfo]:
    return operations.check_skill_updates(destination_root=destination_root)


def apply_updates(
    destination_root: Path,
    selector: str,
    *,
    force: bool,
) -> list[SkillUpdateInfo]:
    updates = operations.check_skill_updates(destination_root=destination_root)
    selected = operations.select_skill_updates(updates, selector)
    if not selected:
        raise SkillLookupError(f"Installed skill not found: {selector}")
    return operations.apply_skill_updates(selected, force=force)


def _record_from_manifest(manifest: SkillManifest) -> InstalledSkillRecord:
    skill_dir = manifest.path.parent if manifest.path.is_file() else manifest.path
    return InstalledSkillRecord(
        name=manifest.name,
        skill_dir=skill_dir,
        manifest=manifest,
        provenance=get_skill_provenance(skill_dir),
    )


def _record_from_install_dir(
    destination_root: Path,
    install_dir: Path,
) -> InstalledSkillRecord:
    for record in list_installed_skills(destination_root):
        if record.skill_dir == install_dir:
            return record
    raise RuntimeError(f"Installed skill could not be reloaded: {install_dir}")
