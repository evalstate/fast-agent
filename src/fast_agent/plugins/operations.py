"""Install, remove, update, and load fast-agent command plugins."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tarfile
import tempfile
import warnings
from pathlib import Path
from typing import Any, BinaryIO, Sequence, cast

from fast_agent.marketplace import source_utils as marketplace_source_utils
from fast_agent.plugins.manifest import load_plugin_manifest
from fast_agent.plugins.marketplace import parse_marketplace_plugins
from fast_agent.plugins.models import (
    LOCAL_REVISION,
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

HeadCache = dict[tuple[str, str | None], tuple[str | None, PluginUpdateStatus | None, str | None]]
PathCache = dict[
    tuple[str, str | None, str, str],
    tuple[str | None, PluginUpdateStatus | None, str | None],
]


async def fetch_marketplace_plugins_with_source(
    url: str,
) -> tuple[list[MarketplacePlugin], str]:
    return await marketplace_source_utils.fetch_marketplace_entries_with_source(
        url,
        candidate_urls=marketplace_source_utils.candidate_marketplace_urls,
        normalize_url=marketplace_source_utils.normalize_marketplace_url,
        load_local_payload=marketplace_source_utils.load_local_marketplace_payload,
        parse_payload=lambda payload, source_url: parse_marketplace_plugins(
            payload,
            source_url=source_url,
        ),
    )


def fetch_marketplace_plugins_with_source_sync(url: str) -> tuple[list[MarketplacePlugin], str]:
    return asyncio.run(fetch_marketplace_plugins_with_source(url))


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
        except Exception as exc:  # noqa: BLE001
            manifest_error = str(exc)
            name = plugin_dir.name
        source, metadata_error = read_installed_plugin_source(plugin_dir)
        plugins.append(
            LocalPlugin(
                index=index,
                name=name,
                plugin_dir=plugin_dir,
                manifest=manifest,
                source=source,
                metadata_error=metadata_error,
                manifest_error=manifest_error,
            )
        )
    return plugins


def select_plugin_by_name_or_index(
    entries: Sequence[MarketplacePlugin],
    selector: str,
) -> MarketplacePlugin | None:
    selector_clean = selector.strip()
    if not selector_clean:
        return None
    if selector_clean.isdigit():
        index = int(selector_clean)
        if 1 <= index <= len(entries):
            return entries[index - 1]
        return None
    selector_lower = selector_clean.lower()
    for entry in entries:
        if entry.name.lower() == selector_lower:
            return entry
    return None


def select_local_plugin_by_name_or_index(
    entries: Sequence[LocalPlugin],
    selector: str,
) -> LocalPlugin | None:
    selector_clean = selector.strip()
    if not selector_clean:
        return None
    if selector_clean.isdigit():
        index = int(selector_clean)
        if 1 <= index <= len(entries):
            return entries[index - 1]
        return None
    selector_lower = selector_clean.lower()
    for entry in entries:
        if entry.name.lower() == selector_lower or entry.plugin_dir.name.lower() == selector_lower:
            return entry
    return None


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

    with tempfile.TemporaryDirectory(dir=destination_root, prefix=f".{plugin.name}.staging-") as tmp:
        staged_dir = Path(tmp) / plugin.install_dir_name
        installed_commit, installed_path_oid, source_origin = _copy_plugin_from_source(
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
            source_origin=source_origin,
            installed_commit=installed_commit,
            installed_path_oid=installed_path_oid,
            fingerprint=fingerprint,
        )
        write_installed_plugin_source(staged_dir, source)
        if install_dir.exists():
            marketplace_source_utils.atomic_replace_directory(
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
        try:
            plugin_dir = _resolve_enabled_plugin_dir(destination_root=destination_root, name=name)
            manifest = load_plugin_manifest(plugin_dir)
        except Exception as exc:  # noqa: BLE001
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
    if (direct_dir / "plugin.yaml").is_file():
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


def select_plugin_updates(updates: Sequence[PluginUpdateInfo], selector: str) -> list[PluginUpdateInfo]:
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
    return [
        update
        for update in updates
        if update.name.lower() == selector_lower or update.plugin_dir.name.lower() == selector_lower
    ][:1]


def apply_plugin_updates(
    updates: Sequence[PluginUpdateInfo],
    *,
    force: bool,
) -> list[PluginUpdateInfo]:
    results: list[PluginUpdateInfo] = []
    for update in updates:
        source = update.managed_source
        if update.status != "update_available" or source is None:
            results.append(update)
            continue
        fingerprint = compute_plugin_content_fingerprint(update.plugin_dir)
        is_dirty = fingerprint != source.content_fingerprint
        if is_dirty and not force:
            results.append(
                PluginUpdateInfo(
                    index=update.index,
                    name=update.name,
                    plugin_dir=update.plugin_dir,
                    status="skipped_dirty",
                    detail="local modifications detected; rerun with --force",
                    current_revision=update.current_revision,
                    available_revision=update.available_revision,
                    managed_source=source,
                )
            )
            continue
        plugin = MarketplacePlugin(
            name=update.name,
            description=None,
            repo_url=source.repo_url,
            repo_ref=source.repo_ref,
            repo_path=source.repo_path,
            source_url=source.source_url,
        )
        try:
            install_marketplace_plugin_sync(
                plugin,
                destination_root=update.plugin_dir.parent,
                replace_existing=True,
                pinned_revision=update.available_revision,
            )
            refreshed = _evaluate_plugin_update(
                plugin_dir=update.plugin_dir,
                index=update.index,
                head_cache={},
                path_cache={},
            )
            results.append(
                PluginUpdateInfo(
                    index=update.index,
                    name=update.name,
                    plugin_dir=update.plugin_dir,
                    status="updated",
                    detail="updated with --force (local changes overwritten)" if is_dirty else "updated",
                    current_revision=source.installed_revision,
                    available_revision=refreshed.current_revision,
                    managed_source=refreshed.managed_source,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                PluginUpdateInfo(
                    index=update.index,
                    name=update.name,
                    plugin_dir=update.plugin_dir,
                    status="source_unreachable",
                    detail=str(exc),
                    current_revision=update.current_revision,
                    available_revision=update.available_revision,
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
    except Exception as exc:  # noqa: BLE001
        return PluginUpdateInfo(index=index, name=plugin_dir.name, plugin_dir=plugin_dir, status="invalid_local_plugin", detail=str(exc))
    source, error = read_installed_plugin_source(plugin_dir)
    if source is None:
        return PluginUpdateInfo(index=index, name=name, plugin_dir=plugin_dir, status="invalid_metadata" if error else "unmanaged", detail=error or "no sidecar metadata")
    if source.installed_commit is None and source.installed_revision == LOCAL_REVISION:
        return PluginUpdateInfo(index=index, name=name, plugin_dir=plugin_dir, status="unknown_revision", detail="source is local non-git; compare unavailable", current_revision=source.installed_revision, available_revision=source.installed_revision, managed_source=source)
    available_revision, status, detail = _resolve_source_revision(source, head_cache)
    if status is not None:
        return PluginUpdateInfo(index=index, name=name, plugin_dir=plugin_dir, status=status, detail=detail, current_revision=source.installed_revision, managed_source=source)
    assert available_revision is not None
    available_path_oid, path_status, path_detail = _resolve_source_path_oid(source, available_revision, path_cache)
    if path_status is not None:
        return PluginUpdateInfo(index=index, name=name, plugin_dir=plugin_dir, status=path_status, detail=path_detail, current_revision=source.installed_revision, managed_source=source)
    current_path_oid = source.installed_path_oid
    if current_path_oid is None and source.installed_commit is not None:
        current_path_oid, _, _ = _resolve_source_path_oid(
            source,
            source.installed_commit,
            path_cache,
        )
    current_revision = source.installed_commit or source.installed_revision
    update_status: PluginUpdateStatus = "up_to_date"
    update_detail = "already up to date"
    if available_path_oid and current_path_oid:
        if available_path_oid != current_path_oid:
            update_status = "update_available"
            update_detail = "plugin content changed"
    elif available_revision != current_revision:
        update_status = "update_available"
        update_detail = "new revision available"
    return PluginUpdateInfo(index=index, name=name, plugin_dir=plugin_dir, status=update_status, detail=update_detail, current_revision=current_revision, available_revision=available_revision, managed_source=source)


def _resolve_source_revision(source: Any, head_cache: HeadCache) -> tuple[str | None, PluginUpdateStatus | None, str | None]:
    resolved = marketplace_source_utils.resolve_local_repo(source.repo_url)
    cache_key = (source.repo_url, source.repo_ref)
    if cache_key in head_cache:
        return head_cache[cache_key]
    if resolved is not None:
        revision = marketplace_source_utils.resolve_git_commit(resolved, source.repo_ref or "HEAD")
        value = (revision or LOCAL_REVISION, None, None)
        head_cache[cache_key] = value
        return value
    args = ["git", "ls-remote", source.repo_url, source.repo_ref or "HEAD"]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        value = (None, "source_unreachable", result.stderr.strip() or result.stdout.strip() or "unable to reach source")
        head_cache[cache_key] = value
        return value
    commit = marketplace_source_utils.parse_ls_remote_commit(result.stdout)
    value = (commit, None, None) if commit else (None, "source_unreachable", "unable to resolve source revision")
    head_cache[cache_key] = value
    return value


def _resolve_source_path_oid(source: Any, commit: str, path_cache: PathCache) -> tuple[str | None, PluginUpdateStatus | None, str | None]:
    path_oid, status, detail = marketplace_source_utils.resolve_source_path_oid(
        repo_url=source.repo_url,
        repo_ref=source.repo_ref,
        repo_path=source.repo_path,
        commit=commit,
        path_cache=path_cache,
    )
    return path_oid, cast("PluginUpdateStatus | None", status), detail


def _copy_plugin_from_source(
    plugin: MarketplacePlugin,
    *,
    destination_dir: Path,
    pinned_revision: str | None,
) -> tuple[str | None, str | None, PluginSourceOrigin]:
    local_repo = marketplace_source_utils.resolve_local_repo(plugin.repo_url)
    if local_repo is not None:
        revision = pinned_revision if pinned_revision and pinned_revision != LOCAL_REVISION else None
        requested_revision = revision or plugin.repo_ref
        if requested_revision:
            commit = marketplace_source_utils.resolve_git_commit(local_repo, requested_revision)
            if commit is None:
                raise FileNotFoundError(f"Plugin source ref not found: {requested_revision}")
            _copy_plugin_source_from_git_commit(
                repo_root=local_repo,
                commit=commit,
                repo_subdir=plugin.repo_subdir,
                destination_dir=destination_dir,
            )
        else:
            source_dir = _resolve_repo_subdir(local_repo, plugin.repo_subdir)
            _copy_plugin_source(source_dir, destination_dir)
            if marketplace_source_utils.is_git_source_dirty(local_repo, source_dir):
                return None, None, "local"
            commit = marketplace_source_utils.resolve_git_commit(local_repo, "HEAD")
        path_oid = marketplace_source_utils.resolve_git_path_oid(local_repo, commit, plugin.repo_path) if commit else None
        return commit, path_oid, "local"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clone_args = ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse"]
        if plugin.repo_ref:
            clone_args.extend(["--branch", plugin.repo_ref])
        clone_args.extend([plugin.repo_url, str(tmp_path)])
        marketplace_source_utils.run_git(clone_args)
        marketplace_source_utils.run_git(["git", "-C", str(tmp_path), "sparse-checkout", "set", plugin.repo_subdir])
        if pinned_revision and pinned_revision != LOCAL_REVISION:
            marketplace_source_utils.run_git(["git", "-C", str(tmp_path), "checkout", pinned_revision])
        else:
            marketplace_source_utils.run_git(["git", "-C", str(tmp_path), "checkout"])
        source_dir = _resolve_repo_subdir(tmp_path, plugin.repo_subdir)
        _copy_plugin_source(source_dir, destination_dir)
        commit = marketplace_source_utils.resolve_git_commit(tmp_path, "HEAD")
        path_oid = marketplace_source_utils.resolve_git_path_oid(tmp_path, commit, plugin.repo_path) if commit else None
        return commit, path_oid, "remote"


def _resolve_repo_subdir(repo_root: Path, repo_subdir: str) -> Path:
    source_dir = (repo_root.resolve() / repo_subdir).resolve()
    source_dir.relative_to(repo_root.resolve())
    return source_dir


def _copy_plugin_source(source_dir: Path, install_dir: Path) -> None:
    if not (source_dir / "plugin.yaml").is_file():
        raise FileNotFoundError("plugin.yaml not found in the selected repository path.")
    shutil.copytree(source_dir, install_dir)


def _copy_plugin_source_from_git_commit(
    *,
    repo_root: Path,
    commit: str,
    repo_subdir: str,
    destination_dir: Path,
) -> None:
    archive_pathspec = f"{commit}:{repo_subdir}"
    result = subprocess.run(
        ["git", "-C", str(repo_root), "archive", "--format=tar", archive_pathspec],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise FileNotFoundError(
            stderr or f"Plugin source path not found at revision {commit}: {repo_subdir}"
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        archive.write(result.stdout)
        archive.seek(0)
        _extract_tar_safely(archive, destination_dir)

    if not (destination_dir / "plugin.yaml").is_file():
        raise FileNotFoundError("plugin.yaml not found in the selected repository path.")


def _extract_tar_safely(archive_file: BinaryIO, destination_dir: Path) -> None:
    destination_root = destination_dir.resolve()
    with tarfile.open(fileobj=archive_file, mode="r:") as archive:
        for member in archive.getmembers():
            target = (destination_root / member.name).resolve()
            target.relative_to(destination_root)
        archive.extractall(destination_root)
