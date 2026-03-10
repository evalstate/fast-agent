"""Operational skills management helpers.

This module owns the source/installation/update lifecycle for skills while
keeping the interface path- and URI-oriented:

- scan a registry URL/path
- install into a destination root
- inspect/update an existing managed root
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fast_agent.marketplace import source_utils as marketplace_source_utils
from fast_agent.skills.marketplace_parsing import (
    parse_marketplace_payload,
)
from fast_agent.skills.models import (
    LOCAL_REVISION,
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

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

HeadResolution = tuple[str | None, SkillUpdateStatus | None, str | None]
PathResolution = tuple[str | None, SkillUpdateStatus | None, str | None]
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


def normalize_marketplace_url(url: str) -> str:
    return marketplace_source_utils.normalize_marketplace_url(url)


def candidate_marketplace_urls(url: str) -> list[str]:
    return marketplace_source_utils.candidate_marketplace_urls(url)


async def fetch_marketplace_skills(url: str) -> list[MarketplaceSkill]:
    skills, _ = await fetch_marketplace_skills_with_source(url)
    return skills


async def fetch_marketplace_skills_with_source(
    url: str,
) -> tuple[list[MarketplaceSkill], str]:
    return await marketplace_source_utils.fetch_marketplace_entries_with_source(
        url,
        candidate_urls=candidate_marketplace_urls,
        normalize_url=normalize_marketplace_url,
        load_local_payload=_load_local_marketplace_payload,
        parse_payload=lambda payload, source_url: parse_marketplace_payload(
            payload,
            source_url=source_url,
        ),
    )


async def install_marketplace_skill(
    skill: MarketplaceSkill,
    *,
    destination_root: Path,
) -> Path:
    return await asyncio.to_thread(install_marketplace_skill_sync, skill, destination_root)


def install_marketplace_skill_sync(skill: MarketplaceSkill, destination_root: Path) -> Path:
    destination_root = destination_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    install_dir = destination_root / skill.install_dir_name
    if install_dir.exists():
        raise FileExistsError(f"Skill already exists: {install_dir}")

    installed_commit, installed_path_oid, source_origin = _copy_skill_from_marketplace_source(
        skill,
        destination_dir=install_dir,
        pinned_revision=None,
    )
    fingerprint = compute_skill_content_fingerprint(install_dir)
    write_installed_skill_source(
        install_dir,
        build_installed_skill_source(
            skill=skill,
            source_origin=source_origin,
            installed_commit=installed_commit,
            installed_path_oid=installed_path_oid,
            fingerprint=fingerprint,
        ),
    )
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
    selector_clean = selector.strip()
    if not selector_clean:
        return None
    entries_list = list(entries)
    if selector_clean.isdigit():
        index = int(selector_clean)
        if 1 <= index <= len(entries_list):
            return entries_list[index - 1]
        return None
    selector_lower = selector_clean.lower()
    for entry in entries_list:
        if entry.name.lower() == selector_lower:
            return entry
    return None


def select_manifest_by_name_or_index(
    manifests: Iterable[SkillManifest],
    selector: str,
) -> SkillManifest | None:
    selector_clean = selector.strip()
    if not selector_clean:
        return None
    manifests_list = list(manifests)
    if selector_clean.isdigit():
        index = int(selector_clean)
        if 1 <= index <= len(manifests_list):
            return manifests_list[index - 1]
        return None
    selector_lower = selector_clean.lower()
    for manifest in manifests_list:
        if manifest.name.lower() == selector_lower:
            return manifest
    return None


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
    selector_clean = selector.strip()
    if not selector_clean:
        return []
    if selector_clean.lower() == "all":
        return list(updates)
    if selector_clean.isdigit():
        index = int(selector_clean)
        if 1 <= index <= len(updates):
            return [updates[index - 1]]
        return []

    selector_lower = selector_clean.lower()
    for update in updates:
        if update.name.lower() == selector_lower:
            return [update]
    return []


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

        if refreshed.status in {
            "up_to_date",
            "unmanaged",
            "invalid_metadata",
            "invalid_local_skill",
            "source_unreachable",
            "source_ref_missing",
            "source_path_missing",
        }:
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
                skill_dir=refreshed.skill_dir,
                source=source,
                revision=refreshed.available_revision,
            )
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
        except Exception as exc:  # noqa: BLE001
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
    manifest_path = skill_dir / "SKILL.md"
    if not manifest_path.exists() or not manifest_path.is_file():
        return SkillUpdateInfo(
            index=index,
            name=name,
            skill_dir=skill_dir,
            status="invalid_local_skill",
            detail="SKILL.md not found",
        )

    source, error = read_installed_skill_source(skill_dir)
    if source is None:
        if error is None:
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
            detail=error,
        )

    source_path_error = _validate_source_path_exists(source)
    if source_path_error is not None:
        return SkillUpdateInfo(
            index=index,
            name=name,
            skill_dir=skill_dir,
            status="source_path_missing",
            detail=source_path_error,
            current_revision=source.installed_revision,
            managed_source=source,
        )

    if source.installed_commit is None and source.installed_revision == LOCAL_REVISION:
        return SkillUpdateInfo(
            index=index,
            name=name,
            skill_dir=skill_dir,
            status="unknown_revision",
            detail="source is local non-git; compare unavailable",
            current_revision=source.installed_revision,
            available_revision=source.installed_revision,
            managed_source=source,
        )

    available_revision, resolve_status, resolve_error = resolve_source_revision(source, head_cache)
    if resolve_status is not None:
        return SkillUpdateInfo(
            index=index,
            name=name,
            skill_dir=skill_dir,
            status=resolve_status,
            detail=resolve_error,
            current_revision=source.installed_revision,
            managed_source=source,
        )

    assert available_revision is not None
    available_path_oid, path_status, path_error = _resolve_source_path_oid(
        source,
        available_revision,
        path_cache,
    )
    if path_status is not None:
        return SkillUpdateInfo(
            index=index,
            name=name,
            skill_dir=skill_dir,
            status=path_status,
            detail=path_error,
            current_revision=source.installed_revision,
            managed_source=source,
        )

    current_path_oid = source.installed_path_oid
    if current_path_oid is None and source.installed_commit is not None:
        current_path_oid, _, _ = _resolve_source_path_oid(
            source,
            source.installed_commit,
            path_cache,
        )

    current_revision = source.installed_commit or source.installed_revision
    status: SkillUpdateStatus = "up_to_date"
    detail = "already up to date"
    if available_path_oid and current_path_oid:
        if available_path_oid != current_path_oid:
            status = "update_available"
            detail = "skill content changed"
    elif available_revision != current_revision:
        status = "update_available"
        detail = "new revision available"

    return SkillUpdateInfo(
        index=index,
        name=name,
        skill_dir=skill_dir,
        status=status,
        detail=detail,
        current_revision=current_revision,
        available_revision=available_revision,
        managed_source=source,
    )


def _validate_source_path_exists(source: InstalledSkillSource) -> str | None:
    local_repo = _resolve_local_repo(source.repo_url)
    if local_repo is None:
        return None

    try:
        source_dir = _resolve_repo_subdir(local_repo, source.repo_path)
    except ValueError as exc:
        return str(exc)

    try:
        source_dir = _resolve_skill_source_dir(source_dir, None)
    except FileNotFoundError as exc:
        return str(exc)
    if not source_dir.exists():
        return f"Skill path not found in repository: {source.repo_path}"
    return None


def resolve_source_revision(
    source: InstalledSkillSource,
    head_cache: HeadCache,
    *,
    resolve_local_repo_fn: Callable[[str], Path | None] | None = None,
    run_subprocess_fn: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> HeadResolution:
    resolve_local_repo = resolve_local_repo_fn or _resolve_local_repo
    run_subprocess = run_subprocess_fn or subprocess.run
    cache_key = (source.repo_url, source.repo_ref)
    cached = head_cache.get(cache_key)
    if cached is not None:
        return cached

    local_repo = resolve_local_repo(source.repo_url)
    if local_repo is not None:
        if source.repo_ref:
            revision = _resolve_git_commit(local_repo, source.repo_ref)
            if revision is None:
                resolved = (
                    None,
                    "source_ref_missing",
                    f"ref not found: {source.repo_ref}",
                )
                head_cache[cache_key] = resolved
                return resolved
        else:
            revision = _resolve_git_commit(local_repo, "HEAD")

        if revision is None:
            resolved = (LOCAL_REVISION, None, None)
            head_cache[cache_key] = resolved
            return resolved

        resolved = (revision, None, None)
        head_cache[cache_key] = resolved
        return resolved

    ls_remote_args = ["git", "ls-remote", source.repo_url]
    ls_remote_args.append(source.repo_ref or "HEAD")

    result = run_subprocess(ls_remote_args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "unable to reach source"
        resolved = (None, "source_unreachable", error)
        head_cache[cache_key] = resolved
        return resolved

    output = result.stdout.strip()
    if not output:
        if source.repo_ref:
            resolved = (
                None,
                "source_ref_missing",
                f"ref not found: {source.repo_ref}",
            )
        else:
            resolved = (None, "source_unreachable", "unable to resolve source HEAD")
        head_cache[cache_key] = resolved
        return resolved

    commit = parse_ls_remote_commit(output)
    if commit is None:
        resolved = (None, "source_unreachable", "unable to resolve source revision")
        head_cache[cache_key] = resolved
        return resolved

    resolved = (commit, None, None)
    head_cache[cache_key] = resolved
    return resolved


def parse_ls_remote_commit(output: str) -> str | None:
    """Extract a commit hash from `git ls-remote` output.

    For annotated tags, prefer the peeled commit (`refs/tags/<tag>^{}`) when present.
    """
    return marketplace_source_utils.parse_ls_remote_commit(output)


def _resolve_source_path_oid(
    source: InstalledSkillSource,
    commit: str,
    path_cache: PathCache,
) -> PathResolution:
    resolved = marketplace_source_utils.resolve_source_path_oid(
        repo_url=source.repo_url,
        repo_ref=source.repo_ref,
        repo_path=source.repo_path,
        commit=commit,
        path_cache=path_cache,
        resolve_local_repo_fn=_resolve_local_repo,
        resolve_git_path_oid_fn=_resolve_git_path_oid,
    )
    path_oid, status, detail = resolved
    return path_oid, cast("SkillUpdateStatus | None", status), detail


def _reinstall_skill_from_source(
    *,
    skill_dir: Path,
    source: InstalledSkillSource,
    revision: str | None,
) -> InstalledSkillSource:
    skill_dir = skill_dir.resolve()
    parent_dir = skill_dir.parent
    source_skill = MarketplaceSkill(
        name=skill_dir.name,
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
        installed_commit, installed_path_oid, _source_origin = _copy_skill_from_marketplace_source(
            source_skill,
            destination_dir=staged_dir,
            pinned_revision=revision,
        )
        fingerprint = compute_skill_content_fingerprint(staged_dir)
        staged_source = build_installed_skill_source(
            skill=source_skill,
            source_origin=source.source_origin,
            installed_commit=installed_commit,
            installed_path_oid=installed_path_oid,
            fingerprint=fingerprint,
        )
        write_installed_skill_source(staged_dir, staged_source)
        _atomic_replace_directory(existing_dir=skill_dir, staged_dir=staged_dir)
        return staged_source


def _copy_skill_from_marketplace_source(
    skill: MarketplaceSkill,
    *,
    destination_dir: Path,
    pinned_revision: str | None,
) -> tuple[str | None, str | None, SkillSourceOrigin]:
    local_repo = _resolve_local_repo(skill.repo_url)
    if local_repo is not None:
        source_dir = _resolve_repo_subdir(local_repo, skill.repo_subdir)
        source_dir = _resolve_skill_source_dir(source_dir, skill.name)
        if not source_dir.exists():
            raise FileNotFoundError(f"Skill path not found in repository: {skill.repo_subdir}")
        _copy_skill_source(source_dir, destination_dir)
        commit = _resolve_git_commit(local_repo, skill.repo_ref or "HEAD")
        path_oid = None
        if commit is not None:
            path_oid = _resolve_git_path_oid(local_repo, commit, skill.repo_path)
        return commit, path_oid, "local"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        clone_args = [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
        ]
        if skill.repo_ref:
            clone_args.extend(["--branch", skill.repo_ref])
        clone_args.extend([skill.repo_url, str(tmp_path)])

        _run_git(clone_args)
        _run_git(["git", "-C", str(tmp_path), "sparse-checkout", "set", skill.repo_subdir])
        if pinned_revision and pinned_revision != LOCAL_REVISION:
            _run_git(["git", "-C", str(tmp_path), "checkout", pinned_revision])
        else:
            _run_git(["git", "-C", str(tmp_path), "checkout"])

        source_dir = _resolve_repo_subdir(tmp_path, skill.repo_subdir)
        source_dir = _resolve_skill_source_dir(source_dir, skill.name)
        if not source_dir.exists():
            raise FileNotFoundError(f"Skill path not found in repository: {skill.repo_subdir}")

        _copy_skill_source(source_dir, destination_dir)
        commit = _resolve_git_commit(tmp_path, "HEAD")
        path_oid = None
        if commit is not None:
            path_oid = _resolve_git_path_oid(tmp_path, commit, skill.repo_path)
        return commit, path_oid, "remote"


def _atomic_replace_directory(*, existing_dir: Path, staged_dir: Path) -> None:
    marketplace_source_utils.atomic_replace_directory(
        existing_dir=existing_dir,
        staged_dir=staged_dir,
    )


def _resolve_git_commit(repo_root: Path, revision: str | None) -> str | None:
    return marketplace_source_utils.resolve_git_commit(repo_root, revision)


def _resolve_git_path_oid(repo_root: Path, commit: str, repo_path: str) -> str | None:
    return marketplace_source_utils.resolve_git_path_oid(repo_root, commit, repo_path)


def _run_git(args: list[str]) -> None:
    marketplace_source_utils.run_git(args)


def _load_local_marketplace_payload(url: str) -> Any | None:
    return marketplace_source_utils.load_local_marketplace_payload(url)


def _resolve_local_repo(repo_url: str) -> Path | None:
    return marketplace_source_utils.resolve_local_repo(repo_url)


def _resolve_repo_subdir(repo_root: Path, repo_subdir: str) -> Path:
    repo_root = repo_root.resolve()
    source_dir = (repo_root / Path(repo_subdir)).resolve()
    try:
        source_dir.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("Skill path escapes repository root.") from exc
    return source_dir


def _copy_skill_source(source_dir: Path, install_dir: Path) -> None:
    if (source_dir / "SKILL.md").exists():
        shutil.copytree(source_dir, install_dir)
    elif source_dir.name.lower() == "skill.md" and source_dir.is_file():
        install_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir, install_dir / "SKILL.md")
    else:
        raise FileNotFoundError("SKILL.md not found in the selected repository path.")


def _resolve_skill_source_dir(source_dir: Path, skill_name: str | None) -> Path:
    if (source_dir / "SKILL.md").exists():
        return source_dir
    if source_dir.is_file() and source_dir.name.lower() == "skill.md":
        return source_dir

    skills_dir = source_dir / "skills"
    if skill_name:
        named_dir = skills_dir / skill_name
        if (named_dir / "SKILL.md").exists():
            return named_dir

    if skills_dir.is_dir():
        candidates = [
            entry
            for entry in skills_dir.iterdir()
            if entry.is_dir() and (entry / "SKILL.md").exists()
        ]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            raise FileNotFoundError(
                "Multiple skills found; specify plugins[].skills to select one."
            )

    return source_dir
