"""Operational skills management helpers.

This module owns the source/installation/update lifecycle for skills while
keeping the interface path- and URI-oriented:

- scan a registry URL/path
- install into a destination root
- inspect/update an existing managed root
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from fast_agent.marketplace import fetch as marketplace_fetch
from fast_agent.marketplace import git_sources as marketplace_git_sources
from fast_agent.marketplace import source_models as marketplace_source_models
from fast_agent.marketplace import source_urls as marketplace_source_urls
from fast_agent.marketplace import update_status as marketplace_update_status
from fast_agent.marketplace.selection import (
    select_one_by_name_or_index,
    select_updates_by_name_or_index,
)
from fast_agent.marketplace.update_status import is_update_applicable
from fast_agent.skills.marketplace_parsing import (
    parse_marketplace_payload,
)
from fast_agent.skills.models import (
    LOCAL_REVISION,
    SKILL_MANIFEST_FILENAME,
    SKILL_MANIFEST_FILENAME_LOWER,
    InstalledSkillSource,
    MarketplaceSkill,
    SkillSourceOrigin,
    SkillUpdateInfo,
    SkillUpdateStatus,
)
from fast_agent.skills.provenance import (
    build_installed_skill_source,
    compute_skill_content_fingerprint,
    read_installed_skill_source,
    write_installed_skill_source,
)
from fast_agent.skills.registry import SkillManifest, SkillRegistry
from fast_agent.utils.async_utils import run_in_thread
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

HeadResolution = marketplace_source_models.SourceRevision[SkillUpdateStatus]


@dataclass(frozen=True, slots=True)
class _CopiedSkillSource:
    origin: SkillSourceOrigin
    commit: str | None
    path_oid: str | None
    resolved_repo_path: str


PathResolution = marketplace_source_models.SourcePathOid[SkillUpdateStatus]
HeadCache = dict[tuple[str, str | None], HeadResolution]
PathCache = dict[tuple[str, str | None, str, str], PathResolution]
__all__ = [
    "apply_skill_updates",
    "candidate_marketplace_urls",
    "check_skill_updates",
    "fetch_marketplace_skills",
    "fetch_marketplace_skills_with_source",
    "install_marketplace_skill",
    "install_marketplace_skill_sync",
    "normalize_marketplace_url",
    "parse_ls_remote_commit",
    "reload_skill_manifests",
    "remove_local_skill",
    "resolve_source_revision",
    "select_manifest_by_name_or_index",
    "select_skill_by_name_or_index",
    "select_skill_updates",
]


class InvalidSkillManifestError(ValueError):
    """Raised when staged skill content is not a valid installable skill."""


def normalize_marketplace_url(url: str) -> str:
    return marketplace_source_urls.normalize_marketplace_url(url)


def candidate_marketplace_urls(url: str) -> list[str]:
    return marketplace_source_urls.candidate_marketplace_urls(url)


async def fetch_marketplace_skills(url: str) -> list[MarketplaceSkill]:
    skills, _ = await fetch_marketplace_skills_with_source(url)
    return skills


async def fetch_marketplace_skills_with_source(
    url: str,
) -> tuple[list[MarketplaceSkill], str]:
    skills, resolved_source = await marketplace_fetch.fetch_marketplace_entries_with_source(
        url,
        candidate_urls=candidate_marketplace_urls,
        normalize_url=normalize_marketplace_url,
        load_local_payload=marketplace_fetch.load_local_marketplace_payload,
        parse_payload=lambda payload, source_url: parse_marketplace_payload(
            payload,
            source_url=source_url,
        ),
    )
    return await run_in_thread(_expand_implicit_skill_bundles, skills), resolved_source


async def install_marketplace_skill(
    skill: MarketplaceSkill,
    *,
    destination_root: Path,
) -> Path:
    return await run_in_thread(install_marketplace_skill_sync, skill, destination_root)


def install_marketplace_skill_sync(skill: MarketplaceSkill, destination_root: Path) -> Path:
    destination_root = destination_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    install_dir = destination_root / skill.install_dir_name
    if install_dir.exists():
        raise FileExistsError(f"Skill already exists: {install_dir}")

    try:
        copied_source = _copy_skill_from_marketplace_source(
            skill,
            destination_dir=install_dir,
            pinned_revision=None,
        )
        fingerprint = compute_skill_content_fingerprint(install_dir)
        write_installed_skill_source(
            install_dir,
            build_installed_skill_source(
                skill=skill,
                source_origin=copied_source.origin,
                installed_commit=copied_source.commit,
                installed_path_oid=copied_source.path_oid,
                repo_path=copied_source.resolved_repo_path,
                fingerprint=fingerprint,
            ),
        )
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir)
        raise
    return install_dir


def remove_local_skill(skill_dir: Path, *, destination_root: Path) -> None:
    skill_dir = skill_dir.resolve()
    destination_root = destination_root.resolve()
    if destination_root not in skill_dir.parents:
        raise ValueError("Skill path is outside of the managed skills directory.")
    if not skill_dir.exists():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")
    shutil.rmtree(skill_dir)


def select_skill_by_name_or_index(
    entries: Iterable[MarketplaceSkill],
    selector: str,
) -> MarketplaceSkill | None:
    def names(entry: MarketplaceSkill) -> list[str]:
        return [entry.name, entry.install_dir_name]

    return select_one_by_name_or_index(
        entries,
        selector,
        names=names,
    )


def select_manifest_by_name_or_index(
    manifests: Iterable[SkillManifest],
    selector: str,
) -> SkillManifest | None:
    def names(manifest: SkillManifest) -> list[str]:
        return [manifest.name]

    return select_one_by_name_or_index(manifests, selector, names=names)


def reload_skill_manifests(
    *,
    base_dir: Path | None = None,
    override_directories: list[Path] | None = None,
) -> tuple[SkillRegistry, list[SkillManifest]]:
    registry = SkillRegistry(
        base_dir=base_dir or Path.cwd(),
        directories=override_directories,
    )
    manifests = registry.load_manifests()
    return registry, manifests


def check_skill_updates(*, destination_root: Path) -> list[SkillUpdateInfo]:
    return _check_skill_updates(destination_root=destination_root)


def select_skill_updates(
    updates: Sequence[SkillUpdateInfo],
    selector: str,
) -> list[SkillUpdateInfo]:
    return select_updates_by_name_or_index(updates, selector, names=lambda update: (update.name,))


def apply_skill_updates(
    updates: Sequence[SkillUpdateInfo],
    *,
    force: bool,
) -> list[SkillUpdateInfo]:
    head_cache: HeadCache = {}
    path_cache: PathCache = {}
    results: list[SkillUpdateInfo] = []
    for update in updates:
        refreshed = _evaluate_skill_update(
            name=update.name,
            skill_dir=update.skill_dir,
            index=update.index,
            head_cache=head_cache,
            path_cache=path_cache,
        )

        if not is_update_applicable(refreshed.status):
            results.append(refreshed)
            continue

        source = refreshed.managed_source
        if source is None:
            results.append(
                SkillUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    skill_dir=refreshed.skill_dir,
                    status="invalid_metadata",
                    detail="missing source metadata",
                )
            )
            continue

        fingerprint = compute_skill_content_fingerprint(refreshed.skill_dir)
        is_dirty = fingerprint != source.content_fingerprint
        if is_dirty and not force:
            results.append(
                SkillUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    skill_dir=refreshed.skill_dir,
                    status="skipped_dirty",
                    detail="local modifications detected; rerun with --force",
                    current_revision=refreshed.current_revision,
                    available_revision=refreshed.available_revision,
                    managed_source=source,
                )
            )
            continue

        try:
            installed_source = _reinstall_skill_from_source(
                skill_name=refreshed.name,
                skill_dir=refreshed.skill_dir,
                source=source,
                revision=refreshed.available_revision,
            )
        except InvalidSkillManifestError as exc:
            results.append(
                SkillUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    skill_dir=refreshed.skill_dir,
                    status="invalid_local_skill",
                    detail=str(exc),
                    current_revision=refreshed.current_revision,
                    available_revision=refreshed.available_revision,
                    managed_source=source,
                )
            )
            continue
        except FileNotFoundError as exc:
            results.append(
                SkillUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    skill_dir=refreshed.skill_dir,
                    status="source_path_missing",
                    detail=str(exc),
                    current_revision=refreshed.current_revision,
                    available_revision=refreshed.available_revision,
                    managed_source=source,
                )
            )
            continue
        except Exception as exc:
            results.append(
                SkillUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    skill_dir=refreshed.skill_dir,
                    status="source_unreachable",
                    detail=str(exc),
                    current_revision=refreshed.current_revision,
                    available_revision=refreshed.available_revision,
                    managed_source=source,
                )
            )
            continue

        detail = "updated"
        if is_dirty and force:
            detail = "updated with --force (local changes overwritten)"

        results.append(
            SkillUpdateInfo(
                index=refreshed.index,
                name=refreshed.name,
                skill_dir=refreshed.skill_dir,
                status="updated",
                detail=detail,
                current_revision=source.installed_revision,
                available_revision=installed_source.installed_revision,
                managed_source=installed_source,
            )
        )

    return results


def _check_skill_updates(*, destination_root: Path) -> list[SkillUpdateInfo]:
    destination_root = destination_root.resolve()
    if not destination_root.exists() or not destination_root.is_dir():
        return []

    manifests, parse_errors = SkillRegistry.load_directory_with_errors(destination_root)
    manifests_by_dir = {
        (manifest.path.parent if manifest.path.is_file() else manifest.path): manifest
        for manifest in manifests
    }
    head_cache: HeadCache = {}
    path_cache: PathCache = {}
    updates: list[SkillUpdateInfo] = []

    skill_dirs = [entry for entry in sorted(destination_root.iterdir()) if entry.is_dir()]
    for index, skill_dir in enumerate(skill_dirs, start=1):
        manifest = manifests_by_dir.get(skill_dir)
        name = manifest.name if manifest else skill_dir.name
        updates.append(
            _evaluate_skill_update(
                name=name,
                skill_dir=skill_dir,
                index=index,
                head_cache=head_cache,
                path_cache=path_cache,
            )
        )

    errors_by_dir = {Path(error["path"]).parent: error["error"] for error in parse_errors}
    for update in updates:
        parse_error = errors_by_dir.get(update.skill_dir)
        if parse_error:
            updates[update.index - 1] = SkillUpdateInfo(
                index=update.index,
                name=update.name,
                skill_dir=update.skill_dir,
                status="invalid_local_skill",
                detail=parse_error,
            )

    return updates


def _evaluate_skill_update(
    *,
    name: str,
    skill_dir: Path,
    index: int,
    head_cache: HeadCache,
    path_cache: PathCache,
) -> SkillUpdateInfo:
    manifest_path = skill_dir / SKILL_MANIFEST_FILENAME
    if not manifest_path.exists() or not manifest_path.is_file():
        return SkillUpdateInfo(
            index=index,
            name=name,
            skill_dir=skill_dir,
            status="invalid_local_skill",
            detail=f"{SKILL_MANIFEST_FILENAME} not found",
        )

    source_metadata = read_installed_skill_source(skill_dir)
    if source_metadata.source is None:
        if source_metadata.error is None:
            return SkillUpdateInfo(
                index=index,
                name=name,
                skill_dir=skill_dir,
                status="unmanaged",
                detail="no sidecar metadata",
            )
        return SkillUpdateInfo(
            index=index,
            name=name,
            skill_dir=skill_dir,
            status="invalid_metadata",
            detail=source_metadata.error,
        )
    source = source_metadata.source
    return _evaluate_managed_skill_update(
        name=name,
        skill_dir=skill_dir,
        index=index,
        source=source,
        head_cache=head_cache,
        path_cache=path_cache,
    )


def _skill_update_info(
    *,
    name: str,
    skill_dir: Path,
    index: int,
    status: SkillUpdateStatus,
    detail: str | None = None,
    current_revision: str | None = None,
    available_revision: str | None = None,
    managed_source: InstalledSkillSource | None = None,
) -> SkillUpdateInfo:
    return SkillUpdateInfo(
        index=index,
        name=name,
        skill_dir=skill_dir,
        status=status,
        detail=detail,
        current_revision=current_revision,
        available_revision=available_revision,
        managed_source=managed_source,
    )


def _local_non_git_skill_update(
    *,
    name: str,
    skill_dir: Path,
    index: int,
    source: InstalledSkillSource,
) -> SkillUpdateInfo:
    source_path_error = _validate_source_path_exists(source, name)
    if source_path_error is not None:
        return _skill_update_info(
            name=name,
            skill_dir=skill_dir,
            index=index,
            status="source_path_missing",
            detail=source_path_error,
            current_revision=source.installed_revision,
            managed_source=source,
        )
    return _skill_update_info(
        name=name,
        skill_dir=skill_dir,
        index=index,
        status="unknown_revision",
        detail="source is local non-git; compare unavailable",
        current_revision=source.installed_revision,
        available_revision=source.installed_revision,
        managed_source=source,
    )


def _evaluate_managed_skill_update(
    *,
    name: str,
    skill_dir: Path,
    index: int,
    source: InstalledSkillSource,
    head_cache: HeadCache,
    path_cache: PathCache,
) -> SkillUpdateInfo:
    if source.source_origin == "mcp":
        return _skill_update_info(
            name=name,
            skill_dir=skill_dir,
            index=index,
            status="unknown_revision",
            detail="source is an MCP skill registry; compare unavailable",
            current_revision=source.installed_revision,
            available_revision=source.installed_revision,
            managed_source=source,
        )

    if source.installed_commit is None and source.installed_revision == LOCAL_REVISION:
        return _local_non_git_skill_update(
            name=name,
            skill_dir=skill_dir,
            index=index,
            source=source,
        )

    resolved_revision = resolve_source_revision(source, head_cache)
    if resolved_revision.status is not None:
        return _skill_update_info(
            name=name,
            skill_dir=skill_dir,
            index=index,
            status=resolved_revision.status,
            detail=resolved_revision.detail,
            current_revision=source.installed_revision,
            managed_source=source,
        )

    available_revision = resolved_revision.revision
    if available_revision is None:
        return _skill_update_info(
            name=name,
            skill_dir=skill_dir,
            index=index,
            status="source_unreachable",
            detail="unable to resolve source revision",
            current_revision=source.installed_revision,
            managed_source=source,
        )
    available_path = _resolve_source_path_oid(
        source,
        available_revision,
        path_cache,
    )
    if available_path.status is not None:
        return _skill_update_info(
            name=name,
            skill_dir=skill_dir,
            index=index,
            status=available_path.status,
            detail=available_path.detail,
            current_revision=source.installed_revision,
            managed_source=source,
        )

    current_path_oid = source.installed_path_oid
    if current_path_oid is None and source.installed_commit is not None:
        current_path_oid = _resolve_source_path_oid(
            source,
            source.installed_commit,
            path_cache,
        ).path_oid

    current_revision = source.installed_commit or source.installed_revision
    decision = marketplace_update_status.decide_source_update_status(
        available_path_oid=available_path.path_oid,
        current_path_oid=current_path_oid,
        available_revision=available_revision,
        current_revision=current_revision,
        content_changed_detail="skill content changed",
    )

    return _skill_update_info(
        name=name,
        skill_dir=skill_dir,
        index=index,
        status=decision.status,
        detail=decision.detail,
        current_revision=current_revision,
        available_revision=available_revision,
        managed_source=source,
    )


def _validate_source_path_exists(
    source: InstalledSkillSource,
    skill_name: str | None,
) -> str | None:
    local_repo = marketplace_git_sources.resolve_local_repo(source.repo_url)
    if local_repo is None:
        return None

    try:
        source_dir = _resolve_repo_subdir(local_repo, source.repo_path)
    except ValueError as exc:
        return str(exc)

    try:
        source_dir = _resolve_skill_source_dir(source_dir, skill_name)
    except FileNotFoundError as exc:
        return str(exc)
    if not source_dir.exists():
        return f"Skill path not found in repository: {source.repo_path}"
    if not _has_skill_manifest(source_dir):
        return f"{SKILL_MANIFEST_FILENAME} not found in repository path: {source.repo_path}"
    return None


def resolve_source_revision(
    source: InstalledSkillSource,
    head_cache: HeadCache,
    *,
    resolve_local_repo_fn: Callable[[str], Path | None] | None = None,
    run_subprocess_fn: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> HeadResolution:
    return marketplace_git_sources.resolve_source_revision(
        repo_url=source.repo_url,
        repo_ref=source.repo_ref,
        head_cache=head_cache,
        local_revision=LOCAL_REVISION,
        source_ref_missing_status="source_ref_missing",
        source_unreachable_status="source_unreachable",
        resolve_local_repo_fn=resolve_local_repo_fn or marketplace_git_sources.resolve_local_repo,
        resolve_git_commit_fn=marketplace_git_sources.resolve_git_commit,
        run_subprocess_fn=run_subprocess_fn or subprocess.run,
    )


def parse_ls_remote_commit(output: str) -> str | None:
    """Extract a commit hash from `git ls-remote` output.

    For annotated tags, prefer the peeled commit (`refs/tags/<tag>^{}`) when present.
    """
    return marketplace_git_sources.parse_ls_remote_commit(output)


def _resolve_source_path_oid(
    source: InstalledSkillSource,
    commit: str,
    path_cache: PathCache,
) -> PathResolution:
    return marketplace_git_sources.resolve_source_path_oid(
        repo_url=source.repo_url,
        repo_ref=source.repo_ref,
        repo_path=source.repo_path,
        commit=commit,
        path_cache=path_cache,
        source_ref_missing_status="source_ref_missing",
        source_unreachable_status="source_unreachable",
        source_path_missing_status="source_path_missing",
        resolve_local_repo_fn=marketplace_git_sources.resolve_local_repo,
        resolve_git_path_oid_fn=marketplace_git_sources.resolve_git_path_oid,
    )


def _reinstall_skill_from_source(
    *,
    skill_name: str,
    skill_dir: Path,
    source: InstalledSkillSource,
    revision: str | None,
) -> InstalledSkillSource:
    skill_dir = skill_dir.resolve()
    parent_dir = skill_dir.parent
    source_skill = MarketplaceSkill(
        name=skill_name,
        description=None,
        repo_url=source.repo_url,
        repo_ref=source.repo_ref,
        repo_path=source.repo_path,
        source_url=source.source_url,
    )

    with tempfile.TemporaryDirectory(
        dir=parent_dir,
        prefix=f".{skill_dir.name}.update-",
    ) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        staged_dir = temp_dir / skill_dir.name
        copied_source = _copy_skill_from_marketplace_source(
            source_skill,
            destination_dir=staged_dir,
            pinned_revision=revision,
        )
        _validate_staged_skill(staged_dir)
        fingerprint = compute_skill_content_fingerprint(staged_dir)
        staged_source = build_installed_skill_source(
            skill=source_skill,
            source_origin=copied_source.origin,
            installed_commit=copied_source.commit,
            installed_path_oid=copied_source.path_oid,
            repo_path=copied_source.resolved_repo_path,
            fingerprint=fingerprint,
        )
        write_installed_skill_source(staged_dir, staged_source)
        marketplace_git_sources.atomic_replace_directory(
            existing_dir=skill_dir,
            staged_dir=staged_dir,
        )
        return staged_source


def _validate_staged_skill(skill_dir: Path) -> None:
    manifest_path = skill_dir / SKILL_MANIFEST_FILENAME
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InvalidSkillManifestError(str(exc)) from exc

    manifest, error = SkillRegistry.parse_manifest_text(manifest_text, path=manifest_path)
    if manifest is None:
        raise InvalidSkillManifestError(error or "Failed to parse skill manifest")


def _copy_skill_from_marketplace_source(
    skill: MarketplaceSkill,
    *,
    destination_dir: Path,
    pinned_revision: str | None,
) -> _CopiedSkillSource:
    checkout_ref = marketplace_git_sources.pinned_checkout_ref(
        pinned_revision,
        local_revision=LOCAL_REVISION,
    )
    local_repo = marketplace_git_sources.resolve_local_repo(skill.repo_url)
    if local_repo is not None:
        requested_revision = checkout_ref or skill.repo_ref
        if requested_revision:
            commit = marketplace_git_sources.resolve_git_commit(
                local_repo,
                requested_revision,
            )
            if commit is None:
                raise FileNotFoundError(f"Skill source ref not found: {requested_revision}")
            resolved_repo_path = _copy_skill_source_from_git_commit(
                repo_root=local_repo,
                commit=commit,
                repo_subdir=skill.repo_subdir,
                skill_name=skill.name,
                destination_dir=destination_dir,
            )
        else:
            repo_subdir_dir = _resolve_repo_subdir(local_repo, skill.repo_subdir)
            source_dir = _resolve_skill_source_dir(repo_subdir_dir, skill.name)
            if not source_dir.exists():
                raise FileNotFoundError(f"Skill path not found in repository: {skill.repo_subdir}")
            _copy_skill_source(source_dir, destination_dir)
            resolved_repo_path = _repo_relative_skill_path(local_repo, source_dir)
            if marketplace_git_sources.is_git_source_dirty(local_repo, source_dir):
                return _CopiedSkillSource(
                    origin="local",
                    commit=None,
                    path_oid=None,
                    resolved_repo_path=resolved_repo_path,
                )
            commit = marketplace_git_sources.resolve_git_commit(local_repo, "HEAD")
        path_oid = marketplace_git_sources.resolve_git_path_oid_if_commit(
            local_repo,
            commit,
            resolved_repo_path,
            resolve_git_path_oid_fn=marketplace_git_sources.resolve_git_path_oid,
        )
        return _CopiedSkillSource(
            origin="local",
            commit=commit,
            path_oid=path_oid,
            resolved_repo_path=resolved_repo_path,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        marketplace_git_sources.clone_sparse_checkout(
            repo_url=skill.repo_url,
            repo_ref=skill.repo_ref,
            repo_subdir=skill.repo_subdir,
            destination_dir=tmp_path,
            checkout_ref=checkout_ref,
        )

        repo_subdir_dir = _resolve_repo_subdir(tmp_path, skill.repo_subdir)
        source_dir = _resolve_skill_source_dir(repo_subdir_dir, skill.name)
        if not source_dir.exists():
            raise FileNotFoundError(f"Skill path not found in repository: {skill.repo_subdir}")

        _copy_skill_source(source_dir, destination_dir)
        resolved_repo_path = _repo_relative_skill_path(tmp_path, source_dir)
        commit = marketplace_git_sources.resolve_git_commit(tmp_path, "HEAD")
        path_oid = marketplace_git_sources.resolve_git_path_oid_if_commit(
            tmp_path,
            commit,
            resolved_repo_path,
            resolve_git_path_oid_fn=marketplace_git_sources.resolve_git_path_oid,
        )
        return _CopiedSkillSource(
            origin="remote",
            commit=commit,
            path_oid=path_oid,
            resolved_repo_path=resolved_repo_path,
        )


def _expand_implicit_skill_bundles(skills: Sequence[MarketplaceSkill]) -> list[MarketplaceSkill]:
    expanded: list[MarketplaceSkill] = []
    for skill in skills:
        expanded.extend(_expand_implicit_skill_bundle(skill))
    return expanded


def _expand_implicit_skill_bundle(skill: MarketplaceSkill) -> list[MarketplaceSkill]:
    if not _may_be_implicit_skill_bundle(skill):
        return [skill]

    local_repo = marketplace_git_sources.resolve_local_repo(skill.repo_url)
    try:
        if local_repo is not None:
            if skill.repo_ref:
                commit = marketplace_git_sources.resolve_git_commit(
                    local_repo,
                    skill.repo_ref,
                )
                if commit is None:
                    return [skill]
                return _discover_nested_marketplace_skills_from_git_commit(
                    skill=skill,
                    repo_root=local_repo,
                    commit=commit,
                )
            source_dir = _resolve_repo_subdir(local_repo, skill.repo_subdir)
            return _discover_nested_marketplace_skills(skill, source_dir)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            marketplace_git_sources.clone_sparse_checkout(
                repo_url=skill.repo_url,
                repo_ref=skill.repo_ref,
                repo_subdir=skill.repo_subdir,
                destination_dir=tmp_path,
            )
            source_dir = _resolve_repo_subdir(tmp_path, skill.repo_subdir)
            return _discover_nested_marketplace_skills(skill, source_dir)
    except Exception:
        return [skill]


def _discover_nested_marketplace_skills_from_git_commit(
    *,
    skill: MarketplaceSkill,
    repo_root: Path,
    commit: str,
) -> list[MarketplaceSkill]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        extracted_dir = Path(tmp_dir) / "source"
        marketplace_git_sources.copy_git_path_from_commit(
            repo_root=repo_root,
            commit=commit,
            repo_subdir=skill.repo_subdir,
            destination_dir=extracted_dir,
            missing_message=(
                f"Skill source path not found at revision {commit}: {skill.repo_subdir}"
            ),
        )
        return _discover_nested_marketplace_skills(skill, extracted_dir)


def _may_be_implicit_skill_bundle(skill: MarketplaceSkill) -> bool:
    path = PurePosixPath(skill.repo_subdir)
    if strip_casefold(path.name) == SKILL_MANIFEST_FILENAME_LOWER:
        return False
    return "skills" not in path.parts


def _discover_nested_marketplace_skills(
    skill: MarketplaceSkill,
    source_dir: Path,
) -> list[MarketplaceSkill]:
    if _has_skill_manifest(source_dir):
        return [skill]

    skills_dir = source_dir / "skills"
    manifests = SkillRegistry.load_directory(skills_dir)
    if not manifests:
        return [skill]

    nested: list[MarketplaceSkill] = []
    for manifest in manifests:
        relative_skill_dir = manifest.path.parent.relative_to(source_dir)
        repo_path = PurePosixPath(skill.repo_subdir) / PurePosixPath(relative_skill_dir.as_posix())
        nested.append(
            MarketplaceSkill(
                name=manifest.name,
                description=manifest.description,
                repo_url=skill.repo_url,
                repo_ref=skill.repo_ref,
                repo_path=str(repo_path),
                source_url=skill.source_url,
                bundle_name=skill.bundle_name or skill.name,
                bundle_description=skill.bundle_description or skill.description,
            )
        )
    return nested


def _resolve_repo_subdir(repo_root: Path, repo_subdir: str) -> Path:
    return marketplace_git_sources.resolve_repo_subdir(
        repo_root,
        repo_subdir,
        label="Skill",
    )


def _copy_skill_source(source_dir: Path, install_dir: Path) -> None:
    if not _has_skill_manifest(source_dir):
        raise FileNotFoundError(
            f"{SKILL_MANIFEST_FILENAME} not found in the selected repository path."
        )
    if source_dir.is_file():
        install_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir, install_dir / SKILL_MANIFEST_FILENAME)
    else:
        shutil.copytree(source_dir, install_dir)


def _copy_skill_source_from_git_commit(
    *,
    repo_root: Path,
    commit: str,
    repo_subdir: str,
    skill_name: str | None,
    destination_dir: Path,
) -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        extracted_dir = Path(tmp_dir) / "source"
        marketplace_git_sources.copy_git_path_from_commit(
            repo_root=repo_root,
            commit=commit,
            repo_subdir=repo_subdir,
            destination_dir=extracted_dir,
            missing_message=f"Skill source path not found at revision {commit}: {repo_subdir}",
        )
        source_dir = _resolve_skill_source_dir(extracted_dir, skill_name)
        _copy_skill_source(source_dir, destination_dir)
        nested_path = _repo_relative_skill_path(extracted_dir, source_dir)
        if nested_path == ".":
            return repo_subdir
        return str(PurePosixPath(repo_subdir) / PurePosixPath(nested_path))


def _resolve_skill_source_dir(source_dir: Path, skill_name: str | None) -> Path:
    if _has_skill_manifest(source_dir):
        return source_dir

    skills_dir = source_dir / "skills"
    if skill_name:
        named_dir = skills_dir / skill_name
        if (named_dir / SKILL_MANIFEST_FILENAME).is_file():
            return named_dir

    if skills_dir.is_dir():
        candidates = [
            entry
            for entry in skills_dir.iterdir()
            if entry.is_dir() and (entry / SKILL_MANIFEST_FILENAME).is_file()
        ]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            raise FileNotFoundError(
                "Multiple skills found; specify plugins[].skills to select one."
            )

    return source_dir


def _repo_relative_skill_path(repo_root: Path, source_dir: Path) -> str:
    try:
        relative = source_dir.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return "."
    relative_text = relative.as_posix()
    return relative_text or "."


def _has_skill_manifest(source_dir: Path) -> bool:
    if source_dir.is_file():
        return strip_casefold(source_dir.name) == SKILL_MANIFEST_FILENAME_LOWER
    return (source_dir / SKILL_MANIFEST_FILENAME).is_file()
