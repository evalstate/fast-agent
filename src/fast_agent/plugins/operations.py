"""Install, remove, update, and load fast-agent command plugins."""

from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fast_agent.marketplace import fetch as marketplace_fetch
from fast_agent.marketplace import git_sources as marketplace_git_sources
from fast_agent.marketplace import provenance_io as marketplace_provenance_io
from fast_agent.marketplace import source_models as marketplace_source_models
from fast_agent.marketplace import source_urls as marketplace_source_urls
from fast_agent.marketplace import update_status as marketplace_update_status
from fast_agent.marketplace.selection import (
    select_one_by_name_or_index,
    select_updates_by_name_or_index,
)
from fast_agent.marketplace.update_status import is_update_applicable
from fast_agent.plugins.manifest import load_plugin_manifest
from fast_agent.plugins.marketplace import parse_marketplace_plugins
from fast_agent.plugins.models import (
    LOCAL_REVISION,
    PLUGIN_MANIFEST_FILENAME,
    InstalledPluginSource,
    LocalPlugin,
    MarketplacePlugin,
    PluginSourceOrigin,
    PluginUpdateInfo,
    PluginUpdateStatus,
)
from fast_agent.plugins.provenance import (
    build_installed_plugin_source,
    compute_plugin_content_fingerprint,
    read_installed_plugin_source,
    write_installed_plugin_source,
)
from fast_agent.utils.async_utils import run_coroutine

if TYPE_CHECKING:
    from collections.abc import Sequence

HeadCache = dict[
    tuple[str, str | None],
    marketplace_source_models.SourceRevision[PluginUpdateStatus],
]
PathCache = dict[
    tuple[str, str | None, str, str],
    marketplace_source_models.SourcePathOid[PluginUpdateStatus],
]


async def fetch_marketplace_plugins_with_source(
    url: str,
) -> tuple[list[MarketplacePlugin], str]:
    return await marketplace_fetch.fetch_marketplace_entries_with_source(
        url,
        candidate_urls=marketplace_source_urls.candidate_marketplace_urls,
        normalize_url=marketplace_source_urls.normalize_marketplace_url,
        load_local_payload=marketplace_fetch.load_local_marketplace_payload,
        parse_payload=lambda payload, source_url: parse_marketplace_plugins(
            payload,
            source_url=source_url,
        ),
    )


def fetch_marketplace_plugins_with_source_sync(url: str) -> tuple[list[MarketplacePlugin], str]:
    return run_coroutine(fetch_marketplace_plugins_with_source(url))


def list_local_plugins(*, destination_root: Path) -> list[LocalPlugin]:
    destination_root = destination_root.resolve()
    if not destination_root.is_dir():
        return []
    plugins: list[LocalPlugin] = []
    for index, plugin_dir in enumerate(
        [entry for entry in sorted(destination_root.iterdir()) if entry.is_dir()],
        start=1,
    ):
        manifest = None
        manifest_error = None
        try:
            manifest = load_plugin_manifest(plugin_dir)
            name = manifest.name
        except Exception as exc:
            manifest_error = str(exc)
            name = plugin_dir.name
        source_metadata = read_installed_plugin_source(plugin_dir)
        plugins.append(
            LocalPlugin(
                index=index,
                name=name,
                plugin_dir=plugin_dir,
                manifest=manifest,
                source=source_metadata.source,
                metadata_error=source_metadata.error,
                manifest_error=manifest_error,
            )
        )
    return plugins


def select_plugin_by_name_or_index(
    entries: Sequence[MarketplacePlugin],
    selector: str,
) -> MarketplacePlugin | None:
    def names(entry: MarketplacePlugin) -> list[str]:
        return [entry.name, entry.install_dir_name]

    return select_one_by_name_or_index(
        entries,
        selector,
        names=names,
    )


def select_local_plugin_by_name_or_index(
    entries: Sequence[LocalPlugin],
    selector: str,
) -> LocalPlugin | None:
    def names(entry: LocalPlugin) -> list[str]:
        return [entry.name, entry.plugin_dir.name]

    return select_one_by_name_or_index(
        entries,
        selector,
        names=names,
    )


def install_marketplace_plugin_sync(
    plugin: MarketplacePlugin,
    *,
    destination_root: Path,
    replace_existing: bool = False,
    pinned_revision: str | None = None,
) -> Path:
    destination_root = destination_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    install_dir = destination_root / plugin.install_dir_name
    if install_dir.exists() and not replace_existing:
        raise FileExistsError(f"Plugin already exists: {plugin.install_dir_name}")

    with tempfile.TemporaryDirectory(
        dir=destination_root,
        prefix=f".{plugin.name}.staging-",
    ) as tmp:
        staged_dir = Path(tmp) / plugin.install_dir_name
        copied_source = _copy_plugin_from_source(
            plugin,
            destination_dir=staged_dir,
            pinned_revision=pinned_revision,
        )
        manifest = load_plugin_manifest(staged_dir)
        plugin = MarketplacePlugin(
            name=manifest.name,
            description=plugin.description,
            repo_url=plugin.repo_url,
            repo_ref=plugin.repo_ref,
            repo_path=plugin.repo_path,
            source_url=plugin.source_url,
            bundle_name=plugin.bundle_name,
        )
        fingerprint = compute_plugin_content_fingerprint(staged_dir)
        source = build_installed_plugin_source(
            plugin=plugin,
            source_origin=copied_source.origin,
            installed_commit=copied_source.commit,
            installed_path_oid=copied_source.path_oid,
            fingerprint=fingerprint,
        )
        write_installed_plugin_source(staged_dir, source)
        if install_dir.exists():
            marketplace_git_sources.atomic_replace_directory(
                existing_dir=install_dir,
                staged_dir=staged_dir,
            )
        else:
            staged_dir.rename(install_dir)
    return install_dir


def remove_local_plugin(plugin_dir: Path, *, destination_root: Path) -> None:
    plugin_dir = plugin_dir.resolve()
    destination_root = destination_root.resolve()
    if destination_root not in plugin_dir.parents:
        raise ValueError("Plugin path is outside of the managed plugins directory.")
    if not plugin_dir.exists():
        raise FileNotFoundError(f"Plugin directory not found: {plugin_dir}")
    shutil.rmtree(plugin_dir)


def load_enabled_plugin_commands(
    *,
    destination_root: Path,
    enabled: Sequence[str],
) -> dict[str, Any]:
    commands: dict[str, Any] = {}
    for name in enabled:
        plugin_dir = _resolve_enabled_plugin_dir(destination_root=destination_root, name=name)
        if not (plugin_dir / PLUGIN_MANIFEST_FILENAME).is_file():
            continue
        try:
            manifest = load_plugin_manifest(plugin_dir)
        except Exception as exc:
            warnings.warn(
                f"Failed to load enabled fast-agent plugin '{name}': {exc}",
                UserWarning,
                stacklevel=2,
            )
            continue
        commands.update(manifest.commands)
    return commands


def _resolve_enabled_plugin_dir(*, destination_root: Path, name: str) -> Path:
    direct_dir = destination_root / name
    if (direct_dir / PLUGIN_MANIFEST_FILENAME).is_file():
        return direct_dir

    for entry in list_local_plugins(destination_root=destination_root):
        if entry.manifest is not None and entry.manifest.name == name:
            return entry.plugin_dir

    return direct_dir


def check_plugin_updates(*, destination_root: Path) -> list[PluginUpdateInfo]:
    destination_root = destination_root.resolve()
    if not destination_root.is_dir():
        return []
    head_cache: HeadCache = {}
    path_cache: PathCache = {}
    updates: list[PluginUpdateInfo] = []
    for index, entry in enumerate(
        [entry for entry in sorted(destination_root.iterdir()) if entry.is_dir()],
        start=1,
    ):
        updates.append(
            _evaluate_plugin_update(
                plugin_dir=entry,
                index=index,
                head_cache=head_cache,
                path_cache=path_cache,
            )
        )
    return updates


def select_plugin_updates(
    updates: Sequence[PluginUpdateInfo],
    selector: str,
) -> list[PluginUpdateInfo]:
    return select_updates_by_name_or_index(
        updates,
        selector,
        names=lambda update: (update.name, update.plugin_dir.name),
    )


def apply_plugin_updates(
    updates: Sequence[PluginUpdateInfo],
    *,
    force: bool,
) -> list[PluginUpdateInfo]:
    head_cache: HeadCache = {}
    path_cache: PathCache = {}
    results: list[PluginUpdateInfo] = []
    for update in updates:
        refreshed = _evaluate_plugin_update(
            plugin_dir=update.plugin_dir,
            index=update.index,
            head_cache=head_cache,
            path_cache=path_cache,
        )
        source = refreshed.managed_source
        if not is_update_applicable(refreshed.status):
            results.append(refreshed)
            continue
        if source is None:
            results.append(
                PluginUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    plugin_dir=refreshed.plugin_dir,
                    status="invalid_metadata",
                    detail="missing source metadata",
                )
            )
            continue

        fingerprint = compute_plugin_content_fingerprint(refreshed.plugin_dir)
        is_dirty = fingerprint != source.content_fingerprint
        if is_dirty and not force:
            results.append(
                PluginUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    plugin_dir=refreshed.plugin_dir,
                    status="skipped_dirty",
                    detail="local modifications detected; rerun with --force",
                    current_revision=refreshed.current_revision,
                    available_revision=refreshed.available_revision,
                    managed_source=source,
                )
            )
            continue
        plugin = MarketplacePlugin(
            name=refreshed.name,
            description=None,
            repo_url=source.repo_url,
            repo_ref=source.repo_ref,
            repo_path=source.repo_path,
            source_url=source.source_url,
        )
        try:
            install_marketplace_plugin_sync(
                plugin,
                destination_root=refreshed.plugin_dir.parent,
                replace_existing=True,
                pinned_revision=refreshed.available_revision,
            )
            after_apply = _evaluate_plugin_update(
                plugin_dir=refreshed.plugin_dir,
                index=refreshed.index,
                head_cache={},
                path_cache={},
            )
            results.append(
                PluginUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    plugin_dir=refreshed.plugin_dir,
                    status="updated",
                    detail=(
                        "updated with --force (local changes overwritten)"
                        if is_dirty
                        else "updated"
                    ),
                    current_revision=source.installed_revision,
                    available_revision=after_apply.current_revision,
                    managed_source=after_apply.managed_source,
                )
            )
        except Exception as exc:
            results.append(
                PluginUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    plugin_dir=refreshed.plugin_dir,
                    status="source_unreachable",
                    detail=str(exc),
                    current_revision=refreshed.current_revision,
                    available_revision=refreshed.available_revision,
                    managed_source=source,
                )
            )
    return results


def _evaluate_plugin_update(
    *,
    plugin_dir: Path,
    index: int,
    head_cache: HeadCache,
    path_cache: PathCache,
) -> PluginUpdateInfo:
    try:
        manifest = load_plugin_manifest(plugin_dir)
        name = manifest.name
    except Exception as exc:
        return PluginUpdateInfo(
            index=index,
            name=plugin_dir.name,
            plugin_dir=plugin_dir,
            status="invalid_local_plugin",
            detail=str(exc),
        )
    source_metadata = read_installed_plugin_source(plugin_dir)
    if source_metadata.source is None:
        return PluginUpdateInfo(
            index=index,
            name=name,
            plugin_dir=plugin_dir,
            status="invalid_metadata" if source_metadata.error else "unmanaged",
            detail=source_metadata.error or "no sidecar metadata",
        )
    source = source_metadata.source
    return _evaluate_managed_plugin_update(
        plugin_dir=plugin_dir,
        index=index,
        name=name,
        source=source,
        head_cache=head_cache,
        path_cache=path_cache,
    )


def _plugin_update_info(
    *,
    plugin_dir: Path,
    index: int,
    name: str,
    status: PluginUpdateStatus,
    detail: str | None = None,
    current_revision: str | None = None,
    available_revision: str | None = None,
    managed_source: InstalledPluginSource | None = None,
) -> PluginUpdateInfo:
    return PluginUpdateInfo(
        index=index,
        name=name,
        plugin_dir=plugin_dir,
        status=status,
        detail=detail,
        current_revision=current_revision,
        available_revision=available_revision,
        managed_source=managed_source,
    )


def _local_non_git_plugin_update(
    *,
    plugin_dir: Path,
    index: int,
    name: str,
    source: InstalledPluginSource,
) -> PluginUpdateInfo:
    source_path_error = _validate_source_path_exists(source)
    if source_path_error is not None:
        return _plugin_update_info(
            plugin_dir=plugin_dir,
            index=index,
            name=name,
            status="source_path_missing",
            detail=source_path_error,
            current_revision=source.installed_revision,
            managed_source=source,
        )
    return _plugin_update_info(
        plugin_dir=plugin_dir,
        index=index,
        name=name,
        status="unknown_revision",
        detail="source is local non-git; compare unavailable",
        current_revision=source.installed_revision,
        available_revision=source.installed_revision,
        managed_source=source,
    )


def _evaluate_managed_plugin_update(
    *,
    plugin_dir: Path,
    index: int,
    name: str,
    source: InstalledPluginSource,
    head_cache: HeadCache,
    path_cache: PathCache,
) -> PluginUpdateInfo:
    if source.installed_commit is None and source.installed_revision == LOCAL_REVISION:
        return _local_non_git_plugin_update(
            plugin_dir=plugin_dir,
            index=index,
            name=name,
            source=source,
        )
    resolved_revision = _resolve_source_revision(source, head_cache)
    if resolved_revision.status is not None:
        return _plugin_update_info(
            plugin_dir=plugin_dir,
            index=index,
            name=name,
            status=resolved_revision.status,
            detail=resolved_revision.detail,
            current_revision=source.installed_revision,
            managed_source=source,
        )
    available_revision = resolved_revision.revision
    if available_revision is None:
        return _plugin_update_info(
            plugin_dir=plugin_dir,
            index=index,
            name=name,
            status="source_unreachable",
            detail="unable to resolve source revision",
            current_revision=source.installed_revision,
            managed_source=source,
        )
    available_path = _resolve_source_path_oid(source, available_revision, path_cache)
    if available_path.status is not None:
        return _plugin_update_info(
            plugin_dir=plugin_dir,
            index=index,
            name=name,
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
        content_changed_detail="plugin content changed",
    )
    return _plugin_update_info(
        plugin_dir=plugin_dir,
        index=index,
        name=name,
        status=decision.status,
        detail=decision.detail,
        current_revision=current_revision,
        available_revision=available_revision,
        managed_source=source,
    )


def _validate_source_path_exists(source: InstalledPluginSource) -> str | None:
    local_repo = marketplace_git_sources.resolve_local_repo(source.repo_url)
    if local_repo is None:
        return None

    try:
        source_dir = marketplace_git_sources.resolve_repo_subdir(
            local_repo,
            marketplace_provenance_io.repo_subdir_for_manifest_path(
                source.repo_path,
                PLUGIN_MANIFEST_FILENAME,
            ),
            label="Plugin",
        )
    except ValueError as exc:
        return str(exc)

    if not source_dir.exists():
        return f"Plugin path not found in repository: {source.repo_path}"
    if not (source_dir / PLUGIN_MANIFEST_FILENAME).is_file():
        return f"Plugin manifest not found in repository: {source.repo_path}"
    return None


def _resolve_source_revision(
    source: InstalledPluginSource,
    head_cache: HeadCache,
) -> marketplace_source_models.SourceRevision[PluginUpdateStatus]:
    return marketplace_git_sources.resolve_source_revision(
        repo_url=source.repo_url,
        repo_ref=source.repo_ref,
        head_cache=head_cache,
        local_revision=LOCAL_REVISION,
        source_ref_missing_status="source_ref_missing",
        source_unreachable_status="source_unreachable",
    )


def _resolve_source_path_oid(
    source: InstalledPluginSource,
    commit: str,
    path_cache: PathCache,
) -> marketplace_source_models.SourcePathOid[PluginUpdateStatus]:
    return marketplace_git_sources.resolve_source_path_oid(
        repo_url=source.repo_url,
        repo_ref=source.repo_ref,
        repo_path=source.repo_path,
        commit=commit,
        path_cache=path_cache,
        source_ref_missing_status="source_ref_missing",
        source_unreachable_status="source_unreachable",
        source_path_missing_status="source_path_missing",
    )


def _copy_plugin_from_source(
    plugin: MarketplacePlugin,
    *,
    destination_dir: Path,
    pinned_revision: str | None,
) -> marketplace_source_models.SourceCopyResult[PluginSourceOrigin]:
    checkout_ref = marketplace_git_sources.pinned_checkout_ref(
        pinned_revision,
        local_revision=LOCAL_REVISION,
    )
    local_repo = marketplace_git_sources.resolve_local_repo(plugin.repo_url)
    if local_repo is not None:
        requested_revision = checkout_ref or plugin.repo_ref
        if requested_revision:
            commit = marketplace_git_sources.resolve_git_commit(local_repo, requested_revision)
            if commit is None:
                raise FileNotFoundError(f"Plugin source ref not found: {requested_revision}")
            _copy_plugin_source_from_git_commit(
                repo_root=local_repo,
                commit=commit,
                repo_subdir=plugin.repo_subdir,
                destination_dir=destination_dir,
            )
        else:
            source_dir = marketplace_git_sources.resolve_repo_subdir(
                local_repo,
                plugin.repo_subdir,
                label="Plugin",
            )
            _copy_plugin_source(source_dir, destination_dir)
            if marketplace_git_sources.is_git_source_dirty(local_repo, source_dir):
                return marketplace_source_models.SourceCopyResult(
                    origin="local",
                    commit=None,
                    path_oid=None,
                )
            commit = marketplace_git_sources.resolve_git_commit(local_repo, "HEAD")
        path_oid = marketplace_git_sources.resolve_git_path_oid_if_commit(
            local_repo,
            commit,
            plugin.repo_subdir,
        )
        return marketplace_source_models.SourceCopyResult(
            origin="local",
            commit=commit,
            path_oid=path_oid,
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        marketplace_git_sources.clone_sparse_checkout(
            repo_url=plugin.repo_url,
            repo_ref=plugin.repo_ref,
            repo_subdir=plugin.repo_subdir,
            destination_dir=tmp_path,
            checkout_ref=checkout_ref,
        )
        source_dir = marketplace_git_sources.resolve_repo_subdir(
            tmp_path,
            plugin.repo_subdir,
            label="Plugin",
        )
        _copy_plugin_source(source_dir, destination_dir)
        commit = marketplace_git_sources.resolve_git_commit(tmp_path, "HEAD")
        path_oid = marketplace_git_sources.resolve_git_path_oid_if_commit(
            tmp_path,
            commit,
            plugin.repo_subdir,
        )
        return marketplace_source_models.SourceCopyResult(
            origin="remote",
            commit=commit,
            path_oid=path_oid,
        )


def _copy_plugin_source(source_dir: Path, install_dir: Path) -> None:
    _validate_plugin_source_dir(source_dir)
    shutil.copytree(source_dir, install_dir)


def _copy_plugin_source_from_git_commit(
    *,
    repo_root: Path,
    commit: str,
    repo_subdir: str,
    destination_dir: Path,
) -> None:
    marketplace_git_sources.copy_git_path_from_commit(
        repo_root=repo_root,
        commit=commit,
        repo_subdir=repo_subdir,
        destination_dir=destination_dir,
        missing_message=f"Plugin source path not found at revision {commit}: {repo_subdir}",
    )

    _validate_plugin_source_dir(destination_dir)


def _validate_plugin_source_dir(source_dir: Path) -> None:
    if not (source_dir / PLUGIN_MANIFEST_FILENAME).is_file():
        raise FileNotFoundError(
            f"{PLUGIN_MANIFEST_FILENAME} not found in the selected repository path."
        )
