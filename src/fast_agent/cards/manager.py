from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fast_agent.config import Settings, get_settings
from fast_agent.core.logging.logger import get_logger
from fast_agent.paths import EnvironmentPaths, resolve_environment_paths

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = get_logger(__name__)

DEFAULT_CARD_REGISTRIES = [
    "https://github.com/fast-agent-ai/card-packs",
]

DEFAULT_CARD_REGISTRY_URL = (
    "https://github.com/fast-agent-ai/card-packs/blob/main/marketplace.json"
)

CARD_PACK_SOURCE_FILENAME = ".card-pack-source.json"
CARD_PACK_SOURCE_SCHEMA_VERSION = 1
LOCAL_REVISION = "local"

CardPackSourceOrigin = Literal["remote", "local"]
CardPackKind = Literal["card", "bundle"]
CardPackUpdateStatus = Literal[
    "up_to_date",
    "update_available",
    "updated",
    "unmanaged",
    "invalid_metadata",
    "invalid_local_pack",
    "unknown_revision",
    "source_unreachable",
    "source_ref_missing",
    "source_path_missing",
    "skipped_dirty",
    "ownership_conflict",
]

CardPackPublishStatus = Literal[
    "published",
    "committed",
    "no_changes",
    "unmanaged",
    "invalid_metadata",
    "source_unreachable",
    "source_path_missing",
    "missing_managed_files",
    "publish_failed",
]


class OwnershipConflictError(ValueError):
    """Raised when an install/update would overwrite protected files."""


@dataclass(frozen=True)
class InstalledCardPackSource:
    schema_version: int
    installed_via: str
    source_origin: CardPackSourceOrigin
    name: str
    kind: CardPackKind
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None
    installed_commit: str | None
    installed_path_oid: str | None
    installed_revision: str
    installed_at: str
    content_fingerprint: str
    installed_files: tuple[str, ...]


@dataclass(frozen=True)
class CardPackUpdateInfo:
    index: int
    name: str
    pack_dir: Path
    status: CardPackUpdateStatus
    detail: str | None = None
    current_revision: str | None = None
    available_revision: str | None = None
    managed_source: InstalledCardPackSource | None = None


@dataclass(frozen=True)
class CardPackPublishResult:
    pack_name: str
    pack_dir: Path
    status: CardPackPublishStatus
    detail: str | None = None
    repo_root: Path | None = None
    repo_path: str | None = None
    commit: str | None = None
    patch_path: Path | None = None
    retained_temp_dir: Path | None = None


@dataclass(frozen=True)
class CardPackManifest:
    schema_version: int
    name: str
    kind: CardPackKind
    version: str | None
    agent_cards: tuple[str, ...]
    tool_cards: tuple[str, ...]
    files: tuple[str, ...]


@dataclass(frozen=True)
class CardPackInstallResult:
    pack_dir: Path
    installed_files: tuple[str, ...]
    source: InstalledCardPackSource


@dataclass(frozen=True)
class CardPackRemovalResult:
    pack_name: str
    removed_paths: tuple[str, ...]
    skipped_paths: tuple[str, ...]


@dataclass(frozen=True)
class MarketplaceCardPack:
    name: str
    description: str | None
    kind: CardPackKind
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None = None
    bundle_name: str | None = None


@dataclass(frozen=True)
class LocalCardPack:
    index: int
    name: str
    pack_dir: Path
    source: InstalledCardPackSource | None
    metadata_error: str | None = None


@dataclass(frozen=True)
class _PlannedCopy:
    source: Path
    destination_relative: str


class _InstallModel(BaseModel):
    agent_cards: list[str] = Field(default_factory=list)
    tool_cards: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class _CardPackManifestModel(BaseModel):
    schema_version: int = 1
    name: str
    kind: CardPackKind = "card"
    version: str | None = None
    install: _InstallModel = Field(default_factory=_InstallModel)

    model_config = ConfigDict(extra="ignore")


class MarketplaceEntryModel(BaseModel):
    name: str | None = None
    description: str | None = None
    kind: str | None = None
    repo_url: str | None = Field(default=None, alias="repo")
    repo_ref: str | None = None
    repo_path: str | None = None
    source_url: str | None = None
    bundle_name: str | None = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_entry(cls, data: Any, info: Any) -> Any:
        if not isinstance(data, dict):
            return data

        context = getattr(info, "context", None) or {}
        default_repo_url = context.get("repo_url")
        default_repo_ref = context.get("repo_ref")
        default_source_url = context.get("source_url")

        repo_url = _first_str(data, "repo", "repository", "git", "repo_url")
        repo_ref = _first_str(data, "ref", "branch", "tag", "revision", "commit")
        repo_path = _first_str(data, "path", "repo_path", "directory", "dir", "location")
        source_url = _first_str(data, "source_url", "url")

        parsed = _parse_github_url(repo_url) if repo_url else None
        if parsed and not repo_path:
            repo_url, repo_ref, repo_path = parsed
        elif parsed:
            repo_url = parsed[0]
            repo_ref = repo_ref or parsed[1]

        source_parsed = _parse_github_url(source_url) if source_url else None
        if source_parsed and (not repo_url or not repo_path):
            repo_url, repo_ref, repo_path = source_parsed

        name = _first_str(data, "name", "id", "slug", "title")
        description = _first_str(data, "description", "summary")
        kind = _first_str(data, "kind", "type")
        bundle_name = _first_str(data, "bundle_name")

        repo_url = repo_url or default_repo_url
        repo_ref = repo_ref or default_repo_ref
        source_url = source_url or default_source_url

        if not name and repo_path:
            name = PurePosixPath(repo_path).name or repo_path

        return {
            "name": name,
            "description": description,
            "kind": kind,
            "repo_url": repo_url,
            "repo_ref": repo_ref,
            "repo_path": repo_path,
            "source_url": source_url,
            "bundle_name": bundle_name,
        }


class MarketplacePayloadModel(BaseModel):
    entries: list[MarketplaceEntryModel] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any, info: Any) -> Any:
        entries = _extract_marketplace_entries(data)
        context = getattr(info, "context", None) or {}
        source_url = context.get("source_url")
        repo_url = context.get("repo_url")
        repo_ref = context.get("repo_ref")
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if source_url and "source_url" not in entry:
                entry["source_url"] = source_url
            if repo_url and "repo_url" not in entry and "repo" not in entry:
                entry["repo_url"] = repo_url
            if repo_ref and "repo_ref" not in entry and "ref" not in entry:
                entry["repo_ref"] = repo_ref
        return {"entries": entries}


def get_manager_directory(settings: Settings | None = None, *, cwd: Path | None = None) -> Path:
    resolved = settings or get_settings()
    env_paths = resolve_environment_paths(resolved, cwd=cwd)
    return env_paths.card_packs


def get_marketplace_url(settings: Settings | None = None) -> str:
    resolved_settings = settings or get_settings()
    cards_settings = getattr(resolved_settings, "cards", None)
    url = None
    if cards_settings is not None:
        url = getattr(cards_settings, "marketplace_url", None)
        if not url:
            urls = getattr(cards_settings, "marketplace_urls", None)
            if urls:
                url = urls[0]
    return _normalize_marketplace_url(url or DEFAULT_CARD_REGISTRY_URL)


def resolve_card_registries(settings: Settings | None = None) -> list[str]:
    resolved_settings = settings or get_settings()
    cards_settings = getattr(resolved_settings, "cards", None)
    configured = (
        getattr(cards_settings, "marketplace_urls", None) if cards_settings else None
    ) or list(DEFAULT_CARD_REGISTRIES)
    active = getattr(cards_settings, "marketplace_url", None) if cards_settings else None
    if active and active not in configured:
        configured.append(active)

    deduped: list[str] = []
    seen = set()
    for url in configured:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def format_marketplace_display_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc == "raw.githubusercontent.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 4:
            org, repo = parts[:2]
            return f"https://github.com/{org}/{repo}"
    if parsed.netloc in {"github.com", "www.github.com"}:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            org, repo = parts[:2]
            return f"https://github.com/{org}/{repo}"
    return url


def list_local_card_packs(*, environment_paths: EnvironmentPaths) -> list[LocalCardPack]:
    destination_root = environment_paths.card_packs.resolve()
    if not destination_root.exists() or not destination_root.is_dir():
        return []

    packs: list[LocalCardPack] = []
    index = 0
    for pack_dir in sorted(destination_root.iterdir()):
        if not pack_dir.is_dir():
            continue
        index += 1
        source, error = read_installed_card_pack_source(pack_dir)
        if source is not None:
            packs.append(
                LocalCardPack(
                    index=index,
                    name=source.name,
                    pack_dir=pack_dir,
                    source=source,
                    metadata_error=None,
                )
            )
            continue
        packs.append(
            LocalCardPack(
                index=index,
                name=pack_dir.name,
                pack_dir=pack_dir,
                source=None,
                metadata_error=error,
            )
        )
    return packs


def select_card_pack_by_name_or_index(
    entries: Iterable[MarketplaceCardPack], selector: str
) -> MarketplaceCardPack | None:
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


def select_installed_card_pack_by_name_or_index(
    entries: Iterable[LocalCardPack], selector: str
) -> LocalCardPack | None:
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
        if entry.name.lower() == selector_lower or entry.pack_dir.name.lower() == selector_lower:
            return entry
    return None


def get_card_pack_source_sidecar_path(pack_dir: Path) -> Path:
    return pack_dir / CARD_PACK_SOURCE_FILENAME


def compute_card_pack_content_fingerprint(
    env_root: Path,
    installed_files: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    root = env_root.resolve()

    for relative in sorted(installed_files):
        normalized = _normalize_repo_path(relative)
        if not normalized:
            continue
        target = (root / normalized).resolve()
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
        if target.exists() and target.is_file():
            file_digest = hashlib.sha256(target.read_bytes()).hexdigest()
            digest.update(file_digest.encode("utf-8"))
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")

    return f"sha256:{digest.hexdigest()}"


def read_installed_card_pack_source(
    pack_dir: Path,
) -> tuple[InstalledCardPackSource | None, str | None]:
    sidecar_path = get_card_pack_source_sidecar_path(pack_dir)
    if not sidecar_path.exists():
        return None, None

    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"invalid json: {exc}"

    if not isinstance(payload, dict):
        return None, "metadata root must be an object"

    try:
        source = _parse_installed_card_pack_source(payload)
    except ValueError as exc:
        return None, str(exc)

    return source, None


def write_installed_card_pack_source(pack_dir: Path, source: InstalledCardPackSource) -> None:
    sidecar_path = get_card_pack_source_sidecar_path(pack_dir)
    payload = {
        "schema_version": source.schema_version,
        "installed_via": source.installed_via,
        "source_origin": source.source_origin,
        "name": source.name,
        "kind": source.kind,
        "repo_url": source.repo_url,
        "repo_ref": source.repo_ref,
        "repo_path": source.repo_path,
        "source_url": source.source_url,
        "installed_commit": source.installed_commit,
        "installed_path_oid": source.installed_path_oid,
        "installed_revision": source.installed_revision,
        "installed_at": source.installed_at,
        "content_fingerprint": source.content_fingerprint,
        "installed_files": list(source.installed_files),
    }
    sidecar_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_card_pack_manifest(pack_root: Path) -> CardPackManifest:
    manifest_path = pack_root / "card-pack.yaml"
    if not manifest_path.exists() or not manifest_path.is_file():
        raise FileNotFoundError(f"card-pack.yaml not found in {pack_root}")

    raw_text = manifest_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    if data is None:
        data = {}

    model = _CardPackManifestModel.model_validate(data)
    if model.schema_version != 1:
        raise ValueError(f"Unsupported card pack schema_version: {model.schema_version}")

    agent_cards = tuple(_validate_manifest_install_path(entry) for entry in model.install.agent_cards)
    tool_cards = tuple(_validate_manifest_install_path(entry) for entry in model.install.tool_cards)
    files = tuple(_validate_manifest_install_path(entry) for entry in model.install.files)

    return CardPackManifest(
        schema_version=model.schema_version,
        name=model.name,
        kind=model.kind,
        version=model.version,
        agent_cards=agent_cards,
        tool_cards=tool_cards,
        files=files,
    )


async def fetch_marketplace_card_packs(url: str) -> list[MarketplaceCardPack]:
    packs, _ = await fetch_marketplace_card_packs_with_source(url)
    return packs


async def fetch_marketplace_card_packs_with_source(
    url: str,
) -> tuple[list[MarketplaceCardPack], str]:
    candidates = _candidate_marketplace_urls(url)
    last_error: Exception | None = None
    for candidate in candidates:
        normalized = _normalize_marketplace_url(candidate)
        local_payload = _load_local_marketplace_payload(normalized)
        if local_payload is not None:
            return _parse_marketplace_payload(local_payload, source_url=normalized), normalized
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(normalized)
                response.raise_for_status()
                data = response.json()
            return _parse_marketplace_payload(data, source_url=normalized), normalized
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    if last_error is not None:
        raise last_error

    return [], _normalize_marketplace_url(url)


async def install_marketplace_card_pack(
    pack: MarketplaceCardPack,
    *,
    environment_paths: EnvironmentPaths,
    force: bool = False,
) -> CardPackInstallResult:
    return await asyncio.to_thread(
        _install_marketplace_card_pack_sync,
        pack,
        environment_paths,
        force,
        False,
        None,
    )


def remove_local_card_pack(
    pack_name: str,
    *,
    environment_paths: EnvironmentPaths,
) -> CardPackRemovalResult:
    destination_root = environment_paths.card_packs.resolve()
    pack_dir = (destination_root / pack_name).resolve()
    if destination_root not in pack_dir.parents:
        raise ValueError("Card pack path is outside of managed card-packs directory.")
    if not pack_dir.exists():
        raise FileNotFoundError(f"Card pack not found: {pack_name}")

    source, error = read_installed_card_pack_source(pack_dir)
    if source is None:
        if error is not None:
            raise ValueError(f"invalid metadata: {error}")
        shutil.rmtree(pack_dir)
        return CardPackRemovalResult(pack_name=pack_name, removed_paths=(), skipped_paths=())

    owners = _collect_installed_file_owners(destination_root)
    removed_paths: list[str] = []
    skipped_paths: list[str] = []
    for relative in source.installed_files:
        owner_set = owners.get(relative, set())
        target = (environment_paths.root / relative).resolve()
        if owner_set != {source.name}:
            skipped_paths.append(relative)
            continue
        if target.exists() and target.is_file():
            target.unlink()
            removed_paths.append(relative)
            _prune_empty_parents(target.parent, stop_at=environment_paths.root.resolve())

    shutil.rmtree(pack_dir)
    return CardPackRemovalResult(
        pack_name=source.name,
        removed_paths=tuple(sorted(removed_paths)),
        skipped_paths=tuple(sorted(skipped_paths)),
    )


def check_card_pack_updates(
    *,
    environment_paths: EnvironmentPaths,
) -> list[CardPackUpdateInfo]:
    destination_root = environment_paths.card_packs.resolve()
    if not destination_root.exists() or not destination_root.is_dir():
        return []

    owners = _collect_installed_file_owners(destination_root)
    updates: list[CardPackUpdateInfo] = []
    head_cache: dict[tuple[str, str | None], tuple[str | None, CardPackUpdateStatus | None, str | None]] = {}
    path_cache: dict[
        tuple[str, str | None, str, str],
        tuple[str | None, CardPackUpdateStatus | None, str | None],
    ] = {}

    index = 0
    for pack_dir in sorted(destination_root.iterdir()):
        if not pack_dir.is_dir():
            continue
        index += 1
        update = _evaluate_card_pack_update(
            pack_dir=pack_dir,
            index=index,
            owners=owners,
            head_cache=head_cache,
            path_cache=path_cache,
        )
        updates.append(update)

    return updates


def select_card_pack_updates(
    updates: Sequence[CardPackUpdateInfo],
    selector: str,
) -> list[CardPackUpdateInfo]:
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
        if update.name.lower() == selector_lower or update.pack_dir.name.lower() == selector_lower:
            return [update]
    return []


def apply_card_pack_updates(
    updates: Sequence[CardPackUpdateInfo],
    *,
    environment_paths: EnvironmentPaths,
    force: bool,
) -> list[CardPackUpdateInfo]:
    destination_root = environment_paths.card_packs.resolve()
    owners = _collect_installed_file_owners(destination_root)
    head_cache: dict[tuple[str, str | None], tuple[str | None, CardPackUpdateStatus | None, str | None]] = {}
    path_cache: dict[
        tuple[str, str | None, str, str],
        tuple[str | None, CardPackUpdateStatus | None, str | None],
    ] = {}

    results: list[CardPackUpdateInfo] = []

    for update in updates:
        refreshed = _evaluate_card_pack_update(
            pack_dir=update.pack_dir,
            index=update.index,
            owners=owners,
            head_cache=head_cache,
            path_cache=path_cache,
        )

        if refreshed.status in {
            "up_to_date",
            "unmanaged",
            "invalid_metadata",
            "invalid_local_pack",
            "unknown_revision",
            "source_unreachable",
            "source_ref_missing",
            "source_path_missing",
            "ownership_conflict",
        }:
            results.append(refreshed)
            continue

        source = refreshed.managed_source
        if source is None:
            results.append(
                CardPackUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    pack_dir=refreshed.pack_dir,
                    status="invalid_metadata",
                    detail="missing source metadata",
                )
            )
            continue

        current_fingerprint = compute_card_pack_content_fingerprint(
            environment_paths.root,
            source.installed_files,
        )
        is_dirty = current_fingerprint != source.content_fingerprint
        if is_dirty and not force:
            results.append(
                CardPackUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    pack_dir=refreshed.pack_dir,
                    status="skipped_dirty",
                    detail="local modifications detected; rerun with --force",
                    current_revision=refreshed.current_revision,
                    available_revision=refreshed.available_revision,
                    managed_source=source,
                )
            )
            continue

        pack = MarketplaceCardPack(
            name=source.name,
            description=None,
            kind=source.kind,
            repo_url=source.repo_url,
            repo_ref=source.repo_ref,
            repo_path=source.repo_path,
            source_url=source.source_url,
            bundle_name=None,
        )

        try:
            install_result = _install_marketplace_card_pack_sync(
                pack,
                environment_paths,
                force,
                True,
                refreshed.available_revision,
            )
        except OwnershipConflictError as exc:
            results.append(
                CardPackUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    pack_dir=refreshed.pack_dir,
                    status="ownership_conflict",
                    detail=str(exc),
                    current_revision=refreshed.current_revision,
                    available_revision=refreshed.available_revision,
                    managed_source=source,
                )
            )
            continue
        except FileNotFoundError as exc:
            results.append(
                CardPackUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    pack_dir=refreshed.pack_dir,
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
                CardPackUpdateInfo(
                    index=refreshed.index,
                    name=refreshed.name,
                    pack_dir=refreshed.pack_dir,
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
            CardPackUpdateInfo(
                index=refreshed.index,
                name=refreshed.name,
                pack_dir=install_result.pack_dir,
                status="updated",
                detail=detail,
                current_revision=source.installed_revision,
                available_revision=install_result.source.installed_revision,
                managed_source=install_result.source,
            )
        )

    return results


def publish_local_card_pack(
    pack_dir: Path,
    *,
    environment_paths: EnvironmentPaths,
    push: bool = True,
    commit_message: str | None = None,
    temp_dir: Path | None = None,
    keep_temp: bool = False,
) -> CardPackPublishResult:
    source, error = read_installed_card_pack_source(pack_dir)
    if source is None:
        if error is None:
            return CardPackPublishResult(
                pack_name=pack_dir.name,
                pack_dir=pack_dir,
                status="unmanaged",
                detail="no sidecar metadata",
            )
        return CardPackPublishResult(
            pack_name=pack_dir.name,
            pack_dir=pack_dir,
            status="invalid_metadata",
            detail=error,
        )

    local_repo = _resolve_local_repo(source.repo_url)
    retained_temp_dir: Path | None = None

    with contextlib.ExitStack() as stack:
        repo_root: Path
        if local_repo is not None:
            repo_root = local_repo
        else:
            temp_parent = temp_dir.expanduser().resolve() if temp_dir is not None else None
            if temp_parent is not None:
                temp_parent.mkdir(parents=True, exist_ok=True)

            if keep_temp:
                repo_root = Path(
                    tempfile.mkdtemp(
                        dir=str(temp_parent) if temp_parent is not None else None,
                        prefix=f".{source.name}.publish-",
                    )
                )
                retained_temp_dir = repo_root
            else:
                repo_root = Path(
                    stack.enter_context(
                        tempfile.TemporaryDirectory(
                            dir=str(temp_parent) if temp_parent is not None else None,
                            prefix=f".{source.name}.publish-",
                        )
                    )
                )

            clone_error = _clone_publish_repository(source=source, destination_dir=repo_root)
            if clone_error is not None:
                return CardPackPublishResult(
                    pack_name=source.name,
                    pack_dir=pack_dir,
                    status="source_unreachable",
                    detail=clone_error,
                    repo_root=repo_root,
                    repo_path=source.repo_path,
                    retained_temp_dir=retained_temp_dir,
                )

        try:
            destination_pack_dir = _resolve_repo_subdir(repo_root, source.repo_path)
        except ValueError as exc:
            return CardPackPublishResult(
                pack_name=source.name,
                pack_dir=pack_dir,
                status="source_path_missing",
                detail=str(exc),
                repo_root=repo_root,
                repo_path=source.repo_path,
                retained_temp_dir=retained_temp_dir,
            )

        try:
            manifest = load_card_pack_manifest(pack_dir)
            plan = _build_install_copy_plan(pack_dir, manifest, env_root=environment_paths.root)
            missing_files = _sync_pack_from_environment(
                copy_plan=plan,
                env_root=environment_paths.root,
            )
        except Exception as exc:  # noqa: BLE001
            return CardPackPublishResult(
                pack_name=source.name,
                pack_dir=pack_dir,
                status="publish_failed",
                detail=str(exc),
                repo_root=repo_root,
                repo_path=source.repo_path,
                retained_temp_dir=retained_temp_dir,
            )

        if missing_files:
            preview = ", ".join(missing_files[:3])
            if len(missing_files) > 3:
                preview = f"{preview}, ..."
            return CardPackPublishResult(
                pack_name=source.name,
                pack_dir=pack_dir,
                status="missing_managed_files",
                detail=f"missing installed file(s) in environment: {preview}",
                repo_root=repo_root,
                repo_path=source.repo_path,
                retained_temp_dir=retained_temp_dir,
            )

        try:
            _sync_directory_contents(
                source_root=pack_dir,
                target_root=destination_pack_dir,
                ignore_names={CARD_PACK_SOURCE_FILENAME, ".publish"},
            )
        except Exception as exc:  # noqa: BLE001
            return CardPackPublishResult(
                pack_name=source.name,
                pack_dir=pack_dir,
                status="publish_failed",
                detail=str(exc),
                repo_root=repo_root,
                repo_path=source.repo_path,
                retained_temp_dir=retained_temp_dir,
            )

        _ensure_git_identity(repo_root)

        add_result = subprocess.run(
            ["git", "-C", str(repo_root), "add", "--all", "--", source.repo_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if add_result.returncode != 0:
            detail = add_result.stderr.strip() or add_result.stdout.strip() or "git add failed"
            return CardPackPublishResult(
                pack_name=source.name,
                pack_dir=pack_dir,
                status="publish_failed",
                detail=detail,
                repo_root=repo_root,
                repo_path=source.repo_path,
                retained_temp_dir=retained_temp_dir,
            )

        status_result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--", source.repo_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if status_result.returncode != 0:
            detail = status_result.stderr.strip() or status_result.stdout.strip() or "git status failed"
            return CardPackPublishResult(
                pack_name=source.name,
                pack_dir=pack_dir,
                status="publish_failed",
                detail=detail,
                repo_root=repo_root,
                repo_path=source.repo_path,
                retained_temp_dir=retained_temp_dir,
            )

        if not status_result.stdout.strip():
            current_commit = _resolve_git_commit(repo_root, "HEAD")
            if current_commit:
                try:
                    _refresh_published_sidecar(
                        pack_dir=pack_dir,
                        source=source,
                        environment_paths=environment_paths,
                        repo_root=repo_root,
                        commit=current_commit,
                    )
                except Exception as exc:  # noqa: BLE001
                    return CardPackPublishResult(
                        pack_name=source.name,
                        pack_dir=pack_dir,
                        status="publish_failed",
                        detail=f"published content but failed to update metadata: {exc}",
                        repo_root=repo_root,
                        repo_path=source.repo_path,
                        commit=current_commit,
                        retained_temp_dir=retained_temp_dir,
                    )
            return CardPackPublishResult(
                pack_name=source.name,
                pack_dir=pack_dir,
                status="no_changes",
                detail="source repository already matches local pack",
                repo_root=repo_root,
                repo_path=source.repo_path,
                commit=current_commit,
                retained_temp_dir=retained_temp_dir,
            )

        message = commit_message or f"Update card pack {source.name}"
        commit_result = subprocess.run(
            ["git", "-C", str(repo_root), "commit", "-m", message],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_result.returncode != 0:
            detail = commit_result.stderr.strip() or commit_result.stdout.strip() or "git commit failed"
            return CardPackPublishResult(
                pack_name=source.name,
                pack_dir=pack_dir,
                status="publish_failed",
                detail=detail,
                repo_root=repo_root,
                repo_path=source.repo_path,
                retained_temp_dir=retained_temp_dir,
            )

        commit = _resolve_git_commit(repo_root, "HEAD")
        if commit:
            try:
                _refresh_published_sidecar(
                    pack_dir=pack_dir,
                    source=source,
                    environment_paths=environment_paths,
                    repo_root=repo_root,
                    commit=commit,
                )
            except Exception as exc:  # noqa: BLE001
                return CardPackPublishResult(
                    pack_name=source.name,
                    pack_dir=pack_dir,
                    status="publish_failed",
                    detail=f"committed but failed to update metadata: {exc}",
                    repo_root=repo_root,
                    repo_path=source.repo_path,
                    commit=commit,
                    retained_temp_dir=retained_temp_dir,
                )

        if not push:
            return CardPackPublishResult(
                pack_name=source.name,
                pack_dir=pack_dir,
                status="committed",
                detail="changes committed locally; push skipped (--no-push)",
                repo_root=repo_root,
                repo_path=source.repo_path,
                commit=commit,
                retained_temp_dir=retained_temp_dir,
            )

        push_result = subprocess.run(
            ["git", "-C", str(repo_root), "push"],
            capture_output=True,
            text=True,
            check=False,
        )
        if push_result.returncode == 0:
            return CardPackPublishResult(
                pack_name=source.name,
                pack_dir=pack_dir,
                status="published",
                detail="changes pushed to remote",
                repo_root=repo_root,
                repo_path=source.repo_path,
                commit=commit,
                retained_temp_dir=retained_temp_dir,
            )

        push_error = push_result.stderr.strip() or push_result.stdout.strip() or "git push failed"
        patch_path = _write_publish_patch(
            repo_root=repo_root,
            pack_dir=pack_dir,
            commit=commit,
        )

        detail = "push failed"
        if push_error:
            detail = f"push failed: {push_error}"

        return CardPackPublishResult(
            pack_name=source.name,
            pack_dir=pack_dir,
            status="publish_failed",
            detail=detail,
            repo_root=repo_root,
            repo_path=source.repo_path,
            commit=commit,
            patch_path=patch_path,
            retained_temp_dir=retained_temp_dir,
        )


def format_revision_short(revision: str | None) -> str:
    if revision is None:
        return "?"
    trimmed = revision.strip()
    if not trimmed:
        return "?"
    normalized = trimmed.lower()
    if len(normalized) >= 8 and all(ch in "0123456789abcdef" for ch in normalized):
        return trimmed[:7]
    return trimmed


def format_installed_at_display(installed_at: str | None) -> str:
    if not installed_at:
        return "unknown"
    normalized = installed_at.strip()
    if not normalized:
        return "unknown"
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return installed_at
    return parsed.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _install_marketplace_card_pack_sync(
    pack: MarketplaceCardPack,
    environment_paths: EnvironmentPaths,
    force: bool,
    replace_existing: bool,
    pinned_revision: str | None,
) -> CardPackInstallResult:
    destination_root = environment_paths.card_packs.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    environment_paths.agent_cards.mkdir(parents=True, exist_ok=True)
    environment_paths.tool_cards.mkdir(parents=True, exist_ok=True)

    install_root = destination_root / pack.name
    if install_root.exists() and not replace_existing:
        raise FileExistsError(f"Card pack already exists: {pack.name}")

    previous_source: InstalledCardPackSource | None = None
    if install_root.exists():
        previous_source, previous_error = read_installed_card_pack_source(install_root)
        if previous_source is None and previous_error is not None:
            raise ValueError(f"invalid metadata for existing pack: {previous_error}")

    current_owned_files = set(previous_source.installed_files if previous_source else ())
    owners = _collect_installed_file_owners(destination_root)

    with tempfile.TemporaryDirectory(
        dir=destination_root,
        prefix=f".{pack.name}.staging-",
    ) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        staged_pack_dir = temp_dir / pack.name
        source_origin, installed_commit, installed_path_oid = _copy_pack_from_marketplace_source(
            pack,
            destination_dir=staged_pack_dir,
            pinned_revision=pinned_revision,
        )

        manifest = load_card_pack_manifest(staged_pack_dir)
        plan = _build_install_copy_plan(
            staged_pack_dir,
            manifest,
            env_root=environment_paths.root,
        )

        conflicts, overwritten_by_owner = _collect_install_conflicts(
            copy_plan=plan,
            env_root=environment_paths.root,
            owners=owners,
            current_pack=pack.name,
            current_owned_files=current_owned_files,
            force=force,
        )
        if conflicts:
            raise OwnershipConflictError("; ".join(conflicts))

        installed_files = tuple(sorted(item.destination_relative for item in plan))
        _apply_copy_plan(
            copy_plan=plan,
            env_root=environment_paths.root,
            current_owned_files=current_owned_files,
            new_owned_files=set(installed_files),
        )

        _revoke_overwritten_ownership(
            environment_paths=environment_paths,
            overwritten_by_owner=overwritten_by_owner,
        )

        fingerprint = compute_card_pack_content_fingerprint(
            environment_paths.root,
            installed_files,
        )
        source = _build_installed_card_pack_source(
            pack=pack,
            source_origin=source_origin,
            installed_commit=installed_commit,
            installed_path_oid=installed_path_oid,
            fingerprint=fingerprint,
            installed_files=installed_files,
        )
        write_installed_card_pack_source(staged_pack_dir, source)

        if install_root.exists():
            _atomic_replace_directory(existing_dir=install_root, staged_dir=staged_pack_dir)
        else:
            os.replace(staged_pack_dir, install_root)

    return CardPackInstallResult(
        pack_dir=install_root,
        installed_files=installed_files,
        source=source,
    )


def _copy_pack_from_marketplace_source(
    pack: MarketplaceCardPack,
    *,
    destination_dir: Path,
    pinned_revision: str | None,
) -> tuple[CardPackSourceOrigin, str | None, str | None]:
    local_repo = _resolve_local_repo(pack.repo_url)
    if local_repo is not None:
        source_dir = _resolve_repo_subdir(local_repo, pack.repo_path)
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"Card pack path not found in repository: {pack.repo_path}")
        if not (source_dir / "card-pack.yaml").exists():
            raise FileNotFoundError(
                f"card-pack.yaml not found in repository path: {pack.repo_path}"
            )
        shutil.copytree(source_dir, destination_dir)

        revision = pinned_revision if pinned_revision and pinned_revision != LOCAL_REVISION else None
        commit = _resolve_git_commit(local_repo, revision or pack.repo_ref or "HEAD")
        path_oid = None
        if commit is not None:
            path_oid = _resolve_git_path_oid(local_repo, commit, pack.repo_path)
        return "local", commit, path_oid

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
        if pack.repo_ref:
            clone_args.extend(["--branch", pack.repo_ref])
        clone_args.extend([pack.repo_url, str(tmp_path)])

        _run_git(clone_args)
        _run_git(["git", "-C", str(tmp_path), "sparse-checkout", "set", pack.repo_path])
        if pinned_revision and pinned_revision != LOCAL_REVISION:
            _run_git(["git", "-C", str(tmp_path), "checkout", pinned_revision])
        else:
            _run_git(["git", "-C", str(tmp_path), "checkout"])

        source_dir = _resolve_repo_subdir(tmp_path, pack.repo_path)
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"Card pack path not found in repository: {pack.repo_path}")
        if not (source_dir / "card-pack.yaml").exists():
            raise FileNotFoundError(
                f"card-pack.yaml not found in repository path: {pack.repo_path}"
            )

        shutil.copytree(source_dir, destination_dir)

        commit = _resolve_git_commit(tmp_path, "HEAD")
        path_oid = None
        if commit is not None:
            path_oid = _resolve_git_path_oid(tmp_path, commit, pack.repo_path)
        return "remote", commit, path_oid


def _build_install_copy_plan(
    pack_root: Path,
    manifest: CardPackManifest,
    *,
    env_root: Path,
) -> list[_PlannedCopy]:
    plan: list[_PlannedCopy] = []

    for entry in manifest.agent_cards:
        source = _resolve_pack_source_path(pack_root, entry)
        destination_relative = str(PurePosixPath("agent-cards") / PurePosixPath(entry).name)
        _ensure_env_target_path(destination_relative, env_root)
        plan.append(_PlannedCopy(source=source, destination_relative=destination_relative))

    for entry in manifest.tool_cards:
        source = _resolve_pack_source_path(pack_root, entry)
        destination_relative = str(PurePosixPath("tool-cards") / PurePosixPath(entry).name)
        _ensure_env_target_path(destination_relative, env_root)
        plan.append(_PlannedCopy(source=source, destination_relative=destination_relative))

    for entry in manifest.files:
        source = _resolve_pack_source_path(pack_root, entry)
        destination_relative = _validate_manifest_install_path(entry)
        _ensure_env_target_path(destination_relative, env_root)
        plan.append(_PlannedCopy(source=source, destination_relative=destination_relative))

    deduped: dict[str, _PlannedCopy] = {}
    for item in plan:
        deduped[item.destination_relative] = item
    return [deduped[key] for key in sorted(deduped.keys())]


def _collect_install_conflicts(
    *,
    copy_plan: Sequence[_PlannedCopy],
    env_root: Path,
    owners: dict[str, set[str]],
    current_pack: str,
    current_owned_files: set[str],
    force: bool,
) -> tuple[list[str], dict[str, set[str]]]:
    conflicts: list[str] = []
    overwritten_by_owner: dict[str, set[str]] = defaultdict(set)

    for item in copy_plan:
        relative = item.destination_relative
        owner_set = set(owners.get(relative, set()))
        owner_set.discard(current_pack)

        target = (env_root / relative).resolve()

        if owner_set and not force:
            owner_list = ", ".join(sorted(owner_set))
            conflicts.append(f"{relative} is owned by another pack: {owner_list}")
            continue

        if target.exists() and not owner_set and relative not in current_owned_files:
            conflicts.append(f"{relative} already exists and is unmanaged")
            continue

        if owner_set and force:
            for owner in owner_set:
                overwritten_by_owner[owner].add(relative)

    return conflicts, overwritten_by_owner


def _apply_copy_plan(
    *,
    copy_plan: Sequence[_PlannedCopy],
    env_root: Path,
    current_owned_files: set[str],
    new_owned_files: set[str],
) -> None:
    for item in copy_plan:
        target = (env_root / item.destination_relative).resolve()
        _atomic_copy_file(item.source, target)

    stale_files = sorted(current_owned_files - new_owned_files)
    for relative in stale_files:
        target = (env_root / relative).resolve()
        if target.exists() and target.is_file():
            target.unlink()
            _prune_empty_parents(target.parent, stop_at=env_root.resolve())


def _sync_pack_from_environment(
    *,
    copy_plan: Sequence[_PlannedCopy],
    env_root: Path,
) -> list[str]:
    missing: list[str] = []
    env_root_resolved = env_root.resolve()
    for item in copy_plan:
        env_file = (env_root_resolved / item.destination_relative).resolve()
        try:
            env_file.relative_to(env_root_resolved)
        except ValueError:
            missing.append(item.destination_relative)
            continue

        if not env_file.exists() or not env_file.is_file():
            missing.append(item.destination_relative)
            continue

        _atomic_copy_file(env_file, item.source)

    return missing


def _sync_directory_contents(
    *,
    source_root: Path,
    target_root: Path,
    ignore_names: set[str] | None = None,
) -> None:
    source_root = source_root.resolve()
    target_root = target_root.resolve()
    ignored = ignore_names or set()

    source_files: set[str] = set()
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(source_root).parts
        if path.name in ignored or any(part in ignored for part in relative_parts):
            continue
        relative = path.relative_to(source_root).as_posix()
        source_files.add(relative)
        _atomic_copy_file(path, target_root / relative)

    if not target_root.exists() or not target_root.is_dir():
        return

    stale_files: list[Path] = []
    for path in target_root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(target_root).parts
        if path.name in ignored or any(part in ignored for part in relative_parts):
            continue
        relative = path.relative_to(target_root).as_posix()
        if relative not in source_files:
            stale_files.append(path)

    for path in stale_files:
        path.unlink(missing_ok=True)
        _prune_empty_parents(path.parent, stop_at=target_root)


def _atomic_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        shutil.copy2(source, temp_path)
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _revoke_overwritten_ownership(
    *,
    environment_paths: EnvironmentPaths,
    overwritten_by_owner: dict[str, set[str]],
) -> None:
    if not overwritten_by_owner:
        return

    destination_root = environment_paths.card_packs.resolve()
    for owner, overwritten in overwritten_by_owner.items():
        owner_dir = destination_root / owner
        source, error = read_installed_card_pack_source(owner_dir)
        if source is None:
            if error:
                logger.warning(
                    "Unable to update overwritten card pack sidecar",
                    data={"owner": owner, "error": error},
                )
            continue

        retained = tuple(path for path in source.installed_files if path not in overwritten)
        updated = InstalledCardPackSource(
            schema_version=source.schema_version,
            installed_via=source.installed_via,
            source_origin=source.source_origin,
            name=source.name,
            kind=source.kind,
            repo_url=source.repo_url,
            repo_ref=source.repo_ref,
            repo_path=source.repo_path,
            source_url=source.source_url,
            installed_commit=source.installed_commit,
            installed_path_oid=source.installed_path_oid,
            installed_revision=source.installed_revision,
            installed_at=source.installed_at,
            content_fingerprint=compute_card_pack_content_fingerprint(
                environment_paths.root,
                retained,
            ),
            installed_files=retained,
        )
        write_installed_card_pack_source(owner_dir, updated)


def _evaluate_card_pack_update(
    *,
    pack_dir: Path,
    index: int,
    owners: dict[str, set[str]],
    head_cache: dict[tuple[str, str | None], tuple[str | None, CardPackUpdateStatus | None, str | None]],
    path_cache: dict[
        tuple[str, str | None, str, str],
        tuple[str | None, CardPackUpdateStatus | None, str | None],
    ],
) -> CardPackUpdateInfo:
    source, error = read_installed_card_pack_source(pack_dir)
    if source is None:
        if error is None:
            return CardPackUpdateInfo(
                index=index,
                name=pack_dir.name,
                pack_dir=pack_dir,
                status="unmanaged",
                detail="no sidecar metadata",
            )
        return CardPackUpdateInfo(
            index=index,
            name=pack_dir.name,
            pack_dir=pack_dir,
            status="invalid_metadata",
            detail=error,
        )

    if not (pack_dir / "card-pack.yaml").exists():
        return CardPackUpdateInfo(
            index=index,
            name=source.name,
            pack_dir=pack_dir,
            status="invalid_local_pack",
            detail="card-pack.yaml not found",
            managed_source=source,
        )

    conflicting_paths = [
        path
        for path in source.installed_files
        if len(owners.get(path, set()) - {source.name}) > 0
    ]
    if conflicting_paths:
        preview = ", ".join(conflicting_paths[:3])
        if len(conflicting_paths) > 3:
            preview = f"{preview}, ..."
        return CardPackUpdateInfo(
            index=index,
            name=source.name,
            pack_dir=pack_dir,
            status="ownership_conflict",
            detail=f"ownership overlaps detected: {preview}",
            current_revision=source.installed_revision,
            managed_source=source,
        )

    source_path_error = _validate_source_path_exists(source)
    if source_path_error is not None:
        return CardPackUpdateInfo(
            index=index,
            name=source.name,
            pack_dir=pack_dir,
            status="source_path_missing",
            detail=source_path_error,
            current_revision=source.installed_revision,
            managed_source=source,
        )

    if source.installed_commit is None and source.installed_revision == LOCAL_REVISION:
        return CardPackUpdateInfo(
            index=index,
            name=source.name,
            pack_dir=pack_dir,
            status="unknown_revision",
            detail="source is local non-git; compare unavailable",
            current_revision=source.installed_revision,
            available_revision=source.installed_revision,
            managed_source=source,
        )

    available_revision, resolve_status, resolve_error = _resolve_source_revision(
        source,
        head_cache,
    )
    if resolve_status is not None:
        return CardPackUpdateInfo(
            index=index,
            name=source.name,
            pack_dir=pack_dir,
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
        return CardPackUpdateInfo(
            index=index,
            name=source.name,
            pack_dir=pack_dir,
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
    status: CardPackUpdateStatus = "up_to_date"
    detail = "already up to date"
    if available_path_oid and current_path_oid:
        if available_path_oid != current_path_oid:
            status = "update_available"
            detail = "card pack content changed"
    elif available_revision != current_revision:
        status = "update_available"
        detail = "new revision available"

    return CardPackUpdateInfo(
        index=index,
        name=source.name,
        pack_dir=pack_dir,
        status=status,
        detail=detail,
        current_revision=current_revision,
        available_revision=available_revision,
        managed_source=source,
    )


def _collect_installed_file_owners(destination_root: Path) -> dict[str, set[str]]:
    owners: dict[str, set[str]] = defaultdict(set)
    if not destination_root.exists() or not destination_root.is_dir():
        return owners

    for pack_dir in sorted(destination_root.iterdir()):
        if not pack_dir.is_dir():
            continue
        source, error = read_installed_card_pack_source(pack_dir)
        if source is None:
            if error:
                logger.warning(
                    "Failed to read card pack metadata while collecting ownership",
                    data={"pack_dir": str(pack_dir), "error": error},
                )
            continue
        for relative in source.installed_files:
            owners[relative].add(source.name)

    return owners


def _parse_installed_card_pack_source(payload: dict[str, Any]) -> InstalledCardPackSource:
    schema_version = payload.get("schema_version")
    if schema_version != CARD_PACK_SOURCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version}")

    installed_via = payload.get("installed_via")
    if not isinstance(installed_via, str) or installed_via.strip() != "marketplace":
        raise ValueError("installed_via must be 'marketplace'")

    source_origin_raw = payload.get("source_origin")
    if source_origin_raw not in {"remote", "local"}:
        raise ValueError("source_origin must be 'remote' or 'local'")

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name is required")

    kind_raw = payload.get("kind")
    if kind_raw not in {"card", "bundle"}:
        raise ValueError("kind must be 'card' or 'bundle'")

    repo_url = payload.get("repo_url")
    if not isinstance(repo_url, str) or not repo_url.strip():
        raise ValueError("repo_url is required")

    repo_ref_value = payload.get("repo_ref")
    repo_ref: str | None
    if repo_ref_value is None:
        repo_ref = None
    elif isinstance(repo_ref_value, str):
        repo_ref = repo_ref_value.strip() or None
    else:
        raise ValueError("repo_ref must be a string or null")

    repo_path_raw = payload.get("repo_path")
    if not isinstance(repo_path_raw, str):
        raise ValueError("repo_path is required")
    repo_path = _normalize_repo_path(repo_path_raw)
    if not repo_path:
        raise ValueError("repo_path is invalid")

    source_url_value = payload.get("source_url")
    source_url: str | None
    if source_url_value is None:
        source_url = None
    elif isinstance(source_url_value, str):
        source_url = source_url_value.strip() or None
    else:
        raise ValueError("source_url must be a string or null")

    installed_commit_value = payload.get("installed_commit")
    installed_commit: str | None
    if installed_commit_value is None:
        installed_commit = None
    elif isinstance(installed_commit_value, str) and installed_commit_value.strip():
        installed_commit = installed_commit_value.strip()
    else:
        raise ValueError("installed_commit must be a non-empty string or null")

    installed_path_oid_value = payload.get("installed_path_oid")
    installed_path_oid: str | None
    if installed_path_oid_value is None:
        installed_path_oid = None
    elif isinstance(installed_path_oid_value, str) and installed_path_oid_value.strip():
        installed_path_oid = installed_path_oid_value.strip()
    else:
        raise ValueError("installed_path_oid must be a non-empty string or null")

    installed_revision = payload.get("installed_revision")
    if not isinstance(installed_revision, str) or not installed_revision.strip():
        raise ValueError("installed_revision is required")

    installed_at = payload.get("installed_at")
    if not isinstance(installed_at, str) or not installed_at.strip():
        raise ValueError("installed_at is required")

    content_fingerprint = payload.get("content_fingerprint")
    if not isinstance(content_fingerprint, str) or not content_fingerprint.startswith("sha256:"):
        raise ValueError("content_fingerprint must be a sha256 fingerprint")

    installed_files_raw = payload.get("installed_files")
    if not isinstance(installed_files_raw, list):
        raise ValueError("installed_files must be a list")

    installed_files: list[str] = []
    for entry in installed_files_raw:
        if not isinstance(entry, str):
            raise ValueError("installed_files entries must be strings")
        normalized = _normalize_repo_path(entry)
        if not normalized:
            raise ValueError(f"invalid installed_files entry: {entry}")
        installed_files.append(normalized)

    return InstalledCardPackSource(
        schema_version=CARD_PACK_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin=source_origin_raw,
        name=name.strip(),
        kind=kind_raw,
        repo_url=repo_url.strip(),
        repo_ref=repo_ref,
        repo_path=repo_path,
        source_url=source_url,
        installed_commit=installed_commit,
        installed_path_oid=installed_path_oid,
        installed_revision=installed_revision.strip(),
        installed_at=installed_at.strip(),
        content_fingerprint=content_fingerprint,
        installed_files=tuple(sorted(dict.fromkeys(installed_files))),
    )


def _build_installed_card_pack_source(
    *,
    pack: MarketplaceCardPack,
    source_origin: CardPackSourceOrigin,
    installed_commit: str | None,
    installed_path_oid: str | None,
    fingerprint: str,
    installed_files: Sequence[str],
) -> InstalledCardPackSource:
    installed_revision = installed_commit or LOCAL_REVISION
    return InstalledCardPackSource(
        schema_version=CARD_PACK_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin=source_origin,
        name=pack.name,
        kind=pack.kind,
        repo_url=pack.repo_url,
        repo_ref=pack.repo_ref,
        repo_path=pack.repo_path,
        source_url=pack.source_url,
        installed_commit=installed_commit,
        installed_path_oid=installed_path_oid,
        installed_revision=installed_revision,
        installed_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        content_fingerprint=fingerprint,
        installed_files=tuple(sorted(installed_files)),
    )


def _refresh_published_sidecar(
    *,
    pack_dir: Path,
    source: InstalledCardPackSource,
    environment_paths: EnvironmentPaths,
    repo_root: Path,
    commit: str,
) -> None:
    installed_path_oid = _resolve_git_path_oid(repo_root, commit, source.repo_path)
    fingerprint = compute_card_pack_content_fingerprint(
        environment_paths.root,
        source.installed_files,
    )
    updated = InstalledCardPackSource(
        schema_version=source.schema_version,
        installed_via=source.installed_via,
        source_origin=source.source_origin,
        name=source.name,
        kind=source.kind,
        repo_url=source.repo_url,
        repo_ref=source.repo_ref,
        repo_path=source.repo_path,
        source_url=source.source_url,
        installed_commit=commit,
        installed_path_oid=installed_path_oid,
        installed_revision=commit,
        installed_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        content_fingerprint=fingerprint,
        installed_files=source.installed_files,
    )
    write_installed_card_pack_source(pack_dir, updated)


def _validate_source_path_exists(source: InstalledCardPackSource) -> str | None:
    local_repo = _resolve_local_repo(source.repo_url)
    if local_repo is None:
        return None

    try:
        source_dir = _resolve_repo_subdir(local_repo, source.repo_path)
    except ValueError as exc:
        return str(exc)

    if not source_dir.exists() or not source_dir.is_dir():
        return f"Card pack path not found in repository: {source.repo_path}"

    if not (source_dir / "card-pack.yaml").exists():
        return f"card-pack.yaml not found in repository path: {source.repo_path}"

    return None


def _resolve_source_revision(
    source: InstalledCardPackSource,
    head_cache: dict[tuple[str, str | None], tuple[str | None, CardPackUpdateStatus | None, str | None]],
) -> tuple[str | None, CardPackUpdateStatus | None, str | None]:
    cache_key = (source.repo_url, source.repo_ref)
    cached = head_cache.get(cache_key)
    if cached is not None:
        return cached

    local_repo = _resolve_local_repo(source.repo_url)
    if local_repo is not None:
        if source.repo_ref:
            revision = _resolve_git_commit(local_repo, source.repo_ref)
            if revision is None:
                resolved = (None, "source_ref_missing", f"ref not found: {source.repo_ref}")
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
    result = subprocess.run(ls_remote_args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "unable to reach source"
        resolved = (None, "source_unreachable", error)
        head_cache[cache_key] = resolved
        return resolved

    output = result.stdout.strip()
    if not output:
        if source.repo_ref:
            resolved = (None, "source_ref_missing", f"ref not found: {source.repo_ref}")
        else:
            resolved = (None, "source_unreachable", "unable to resolve source HEAD")
        head_cache[cache_key] = resolved
        return resolved

    commit = _parse_ls_remote_commit(output)
    if commit is None:
        resolved = (None, "source_unreachable", "unable to resolve source revision")
        head_cache[cache_key] = resolved
        return resolved

    resolved = (commit, None, None)
    head_cache[cache_key] = resolved
    return resolved


def _resolve_source_path_oid(
    source: InstalledCardPackSource,
    commit: str,
    path_cache: dict[
        tuple[str, str | None, str, str],
        tuple[str | None, CardPackUpdateStatus | None, str | None],
    ],
) -> tuple[str | None, CardPackUpdateStatus | None, str | None]:
    cache_key = (source.repo_url, source.repo_ref, source.repo_path, commit)
    cached = path_cache.get(cache_key)
    if cached is not None:
        return cached

    local_repo = _resolve_local_repo(source.repo_url)
    if local_repo is not None:
        path_oid = _resolve_git_path_oid(local_repo, commit, source.repo_path)
        if path_oid is None:
            resolved = (
                None,
                "source_path_missing",
                f"path missing at revision {commit}: {source.repo_path}",
            )
            path_cache[cache_key] = resolved
            return resolved
        resolved = (path_oid, None, None)
        path_cache[cache_key] = resolved
        return resolved

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
        if source.repo_ref:
            clone_args.extend(["--branch", source.repo_ref])
        clone_args.extend([source.repo_url, str(tmp_path)])

        result = subprocess.run(clone_args, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            if source.repo_ref and "Remote branch" in stderr and "not found" in stderr:
                resolved = (None, "source_ref_missing", f"ref not found: {source.repo_ref}")
            else:
                resolved = (None, "source_unreachable", stderr or "unable to reach source")
            path_cache[cache_key] = resolved
            return resolved

        path_oid = _resolve_git_path_oid(tmp_path, commit, source.repo_path)
        if path_oid is None:
            resolved = (
                None,
                "source_path_missing",
                f"path missing at revision {commit}: {source.repo_path}",
            )
            path_cache[cache_key] = resolved
            return resolved

    resolved = (path_oid, None, None)
    path_cache[cache_key] = resolved
    return resolved


def _parse_ls_remote_commit(output: str) -> str | None:
    fallback: str | None = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        commit = parts[0].strip()
        if not commit:
            continue
        ref = parts[1].strip() if len(parts) > 1 else ""
        if ref.endswith("^{}"):
            return commit
        if fallback is None:
            fallback = commit
    return fallback


def _normalize_marketplace_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc in {"github.com", "www.github.com"}:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] == "blob":
            org, repo, _, ref = parts[:4]
            file_path = "/".join(parts[4:])
            return f"https://raw.githubusercontent.com/{org}/{repo}/{ref}/{file_path}"
    return url


def candidate_marketplace_urls(url: str) -> list[str]:
    return _candidate_marketplace_urls(url)


def _candidate_marketplace_urls(url: str) -> list[str]:
    normalized = url.strip()
    if not normalized:
        return []

    parsed = urlparse(normalized)
    if parsed.scheme in {"file", ""} and parsed.netloc == "":
        path = Path(parsed.path).expanduser()
        if path.exists() and path.is_dir():
            claude_plugin = path / ".claude-plugin" / "marketplace.json"
            if claude_plugin.exists():
                return [claude_plugin.as_posix()]
            fallback = path / "marketplace.json"
            if fallback.exists():
                return [fallback.as_posix()]
        return [normalized]

    if parsed.netloc in {"github.com", "www.github.com"}:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            org, repo = parts[:2]
            if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
                ref = parts[3]
                base_path = "/".join(parts[4:])
                return _github_marketplace_candidates(org, repo, ref, base_path)
            if len(parts) == 2:
                return [
                    *_github_marketplace_candidates(org, repo, "main", ""),
                    *_github_marketplace_candidates(org, repo, "master", ""),
                ]

    return [normalized]


def _github_marketplace_candidates(org: str, repo: str, ref: str, base_path: str) -> list[str]:
    suffixes = _marketplace_path_candidates(base_path)
    return [
        f"https://raw.githubusercontent.com/{org}/{repo}/{ref}/{suffix}"
        for suffix in suffixes
    ]


def _marketplace_path_candidates(base_path: str) -> list[str]:
    cleaned = base_path.strip().strip("/")
    if not cleaned:
        return [".claude-plugin/marketplace.json", "marketplace.json"]

    path = PurePosixPath(cleaned)
    if path.name.lower() == "marketplace.json":
        return [str(path)]
    if path.name == ".claude-plugin":
        return [str(path / "marketplace.json")]

    return [
        str(path / ".claude-plugin" / "marketplace.json"),
        str(path / "marketplace.json"),
    ]


def _parse_marketplace_payload(
    payload: Any,
    *,
    source_url: str | None = None,
) -> list[MarketplaceCardPack]:
    repo_url = None
    repo_ref = None
    if source_url:
        parsed = _parse_github_url(source_url)
        if parsed:
            repo_url, repo_ref, _ = parsed
        else:
            local_repo = _derive_local_repo_root(source_url)
            if local_repo:
                repo_url = local_repo

    try:
        model = MarketplacePayloadModel.model_validate(
            payload,
            context={
                "source_url": source_url,
                "repo_url": repo_url,
                "repo_ref": repo_ref,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse card marketplace payload", data={"error": str(exc)})
        return []

    entries: list[MarketplaceCardPack] = []
    for entry in model.entries:
        try:
            parsed_entry = _card_pack_from_entry_model(entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to parse card marketplace entry",
                data={"error": str(exc), "entry": _safe_json(entry.model_dump())},
            )
            continue
        if parsed_entry is not None:
            entries.append(parsed_entry)
    return entries


def _card_pack_from_entry_model(model: MarketplaceEntryModel) -> MarketplaceCardPack | None:
    if not model.repo_url or not model.repo_path:
        return None

    repo_path = _normalize_repo_path(model.repo_path)
    if not repo_path:
        return None

    kind_raw = (model.kind or "card").strip().lower()
    kind: CardPackKind = "bundle" if kind_raw == "bundle" else "card"

    return MarketplaceCardPack(
        name=model.name or PurePosixPath(repo_path).name,
        description=model.description,
        kind=kind,
        repo_url=model.repo_url,
        repo_ref=model.repo_ref,
        repo_path=repo_path,
        source_url=model.source_url,
        bundle_name=model.bundle_name,
    )


def _extract_marketplace_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]

    if isinstance(payload, dict):
        for key in ("card_packs", "cards", "entries", "items", "marketplace", "plugins"):
            value = payload.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]

        if all(isinstance(value, dict) for value in payload.values()):
            return [value for value in payload.values() if isinstance(value, dict)]

    raise ValueError("Unsupported marketplace payload format.")


def _resolve_pack_source_path(pack_root: Path, relative_path: str) -> Path:
    pack_root = pack_root.resolve()
    source = (pack_root / relative_path).resolve()
    try:
        source.relative_to(pack_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes pack root: {relative_path}") from exc

    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Referenced file not found: {relative_path}")

    return source


def _ensure_env_target_path(relative_path: str, env_root: Path) -> None:
    normalized = _normalize_repo_path(relative_path)
    if not normalized:
        raise ValueError(f"Invalid install target path: {relative_path}")
    target = (env_root / normalized).resolve()
    try:
        target.relative_to(env_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Install target escapes environment root: {relative_path}") from exc


def _validate_manifest_install_path(value: str) -> str:
    normalized = _normalize_repo_path(value)
    if not normalized:
        raise ValueError(f"Invalid install path: {value}")
    return normalized


def _normalize_repo_path(path: str) -> str | None:
    if not path:
        return None
    raw = path.strip()
    if not raw:
        return None
    raw = raw.replace("\\", "/")
    posix_path = PurePosixPath(raw)
    if posix_path.is_absolute():
        return None
    if ".." in posix_path.parts:
        return None
    normalized = str(posix_path).lstrip("/")
    if normalized in {"", "."}:
        return None
    return normalized


def _first_str(entry: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return str(value)


def _load_local_marketplace_payload(url: str) -> Any | None:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = Path(parsed.path)
        return _read_json_file(path)
    if parsed.scheme in {"http", "https"}:
        return None
    candidate = Path(url).expanduser()
    if candidate.exists():
        return _read_json_file(candidate)
    return None


def _read_json_file(path: Path) -> Any:
    content = path.read_text(encoding="utf-8")
    return json.loads(content)


def _resolve_local_repo(repo_url: str) -> Path | None:
    parsed = urlparse(repo_url)
    if parsed.scheme == "file":
        repo_path = Path(parsed.path)
    elif parsed.scheme in {"http", "https", "ssh"}:
        return None
    else:
        repo_path = Path(repo_url)

    repo_path = repo_path.expanduser()
    if not repo_path.is_absolute():
        repo_path = repo_path.resolve()
    if repo_path.exists():
        return repo_path
    return None


def _derive_local_repo_root(source_url: str) -> str | None:
    parsed = urlparse(source_url)
    if parsed.scheme in {"http", "https", "ssh"}:
        return None

    if parsed.scheme == "file":
        path = Path(parsed.path)
    else:
        path = Path(source_url)

    path = path.expanduser()
    if not path.is_absolute():
        path = path.resolve()

    if not path.exists():
        return None

    if path.is_file() and path.name == "marketplace.json":
        if path.parent.name == ".claude-plugin":
            repo_root = path.parent.parent
        else:
            repo_root = path.parent
        if repo_root.exists():
            return str(repo_root)

    if path.is_dir():
        return str(path)

    return None


def _resolve_repo_subdir(repo_root: Path, repo_subdir: str) -> Path:
    repo_root = repo_root.resolve()
    source_dir = (repo_root / Path(repo_subdir)).resolve()
    try:
        source_dir.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("Card pack path escapes repository root.") from exc
    return source_dir


def _resolve_git_commit(repo_root: Path, revision: str | None) -> str | None:
    rev = revision or "HEAD"
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", f"{rev}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()
    if not lines:
        return None
    commit = lines[0].strip()
    return commit or None


def _resolve_git_path_oid(repo_root: Path, commit: str, repo_path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", f"{commit}:{repo_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()
    if not lines:
        return None
    oid = lines[0].strip()
    return oid or None


def _run_git(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Git command failed: {' '.join(args)}\n{stderr}")


def _clone_publish_repository(*, source: InstalledCardPackSource, destination_dir: Path) -> str | None:
    clone_args = [
        "git",
        "clone",
        "--depth",
        "1",
        "--filter=blob:none",
        "--sparse",
    ]
    if source.repo_ref:
        clone_args.extend(["--branch", source.repo_ref])
    clone_args.extend([source.repo_url, str(destination_dir)])

    clone_result = subprocess.run(clone_args, capture_output=True, text=True, check=False)
    if clone_result.returncode != 0:
        return clone_result.stderr.strip() or clone_result.stdout.strip() or "git clone failed"

    sparse_result = subprocess.run(
        ["git", "-C", str(destination_dir), "sparse-checkout", "set", source.repo_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if sparse_result.returncode != 0:
        return (
            sparse_result.stderr.strip()
            or sparse_result.stdout.strip()
            or "git sparse-checkout failed"
        )

    checkout_target = source.repo_ref or "HEAD"
    checkout_result = subprocess.run(
        ["git", "-C", str(destination_dir), "checkout", checkout_target],
        capture_output=True,
        text=True,
        check=False,
    )
    if checkout_result.returncode != 0:
        return checkout_result.stderr.strip() or checkout_result.stdout.strip() or "git checkout failed"

    return None


def _ensure_git_identity(repo_root: Path) -> None:
    name_result = subprocess.run(
        ["git", "-C", str(repo_root), "config", "--get", "user.name"],
        capture_output=True,
        text=True,
        check=False,
    )
    email_result = subprocess.run(
        ["git", "-C", str(repo_root), "config", "--get", "user.email"],
        capture_output=True,
        text=True,
        check=False,
    )

    if name_result.returncode != 0 or not name_result.stdout.strip():
        subprocess.run(
            ["git", "-C", str(repo_root), "config", "user.name", "fast-agent"],
            capture_output=True,
            text=True,
            check=False,
        )

    if email_result.returncode != 0 or not email_result.stdout.strip():
        subprocess.run(
            ["git", "-C", str(repo_root), "config", "user.email", "fast-agent@localhost"],
            capture_output=True,
            text=True,
            check=False,
        )


def _write_publish_patch(*, repo_root: Path, pack_dir: Path, commit: str | None) -> Path | None:
    if not commit:
        return None

    patch_result = subprocess.run(
        ["git", "-C", str(repo_root), "format-patch", "-1", commit, "--stdout"],
        capture_output=True,
        text=True,
        check=False,
    )
    if patch_result.returncode != 0 or not patch_result.stdout:
        return None

    publish_dir = pack_dir / ".publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    commit_short = commit[:7]
    patch_path = publish_dir / f"{timestamp}-{commit_short}.patch"
    patch_path.write_text(patch_result.stdout, encoding="utf-8")
    return patch_path


def _parse_github_url(url: str | None) -> tuple[str, str | None, str] | None:
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.netloc in {"github.com", "www.github.com"}:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] in {"blob", "tree"}:
            org, repo, _, ref = parts[:4]
            file_path = "/".join(parts[4:])
            return f"https://github.com/{org}/{repo}", ref, file_path
    if parsed.netloc == "raw.githubusercontent.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 4:
            org, repo, ref = parts[:3]
            file_path = "/".join(parts[3:])
            return f"https://github.com/{org}/{repo}", ref, file_path
    return None


def _atomic_replace_directory(*, existing_dir: Path, staged_dir: Path) -> None:
    existing_dir = existing_dir.resolve()
    staged_dir = staged_dir.resolve()
    parent = existing_dir.parent
    backup_dir = parent / f".{existing_dir.name}.backup-{uuid4().hex}"

    os.replace(existing_dir, backup_dir)
    try:
        os.replace(staged_dir, existing_dir)
    except Exception:
        os.replace(backup_dir, existing_dir)
        raise
    shutil.rmtree(backup_dir)


def _prune_empty_parents(path: Path, *, stop_at: Path) -> None:
    current = path.resolve()
    root = stop_at.resolve()
    while current != root and root in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent
