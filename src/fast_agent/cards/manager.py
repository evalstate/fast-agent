from __future__ import annotations

import contextlib
import hashlib
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from fast_agent.config import Settings, get_settings
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.model_resolution import parse_model_reference_token
from fast_agent.home import CONFIG_FILENAMES, PREFERRED_CONFIG_FILENAME, find_config_in_directory
from fast_agent.marketplace import fetch as marketplace_fetch
from fast_agent.marketplace import formatting as marketplace_formatting
from fast_agent.marketplace import git_sources as marketplace_git_sources
from fast_agent.marketplace import provenance_io as marketplace_provenance_io
from fast_agent.marketplace import registry_urls as marketplace_registry_urls
from fast_agent.marketplace import source_models as marketplace_source_models
from fast_agent.marketplace import source_urls as marketplace_source_urls
from fast_agent.marketplace import update_status as marketplace_update_status
from fast_agent.marketplace.models import MarketplaceEntryFieldsModel
from fast_agent.marketplace.selection import (
    select_one_by_name_or_index,
    select_updates_by_name_or_index,
)
from fast_agent.marketplace.update_status import (
    CommonMarketplaceUpdateStatus,
    clean_update_status_detail,
    is_update_applicable,
)
from fast_agent.paths import HomePaths, resolve_home_paths
from fast_agent.utils.action_normalization import normalize_action_token
from fast_agent.utils.async_utils import run_in_thread
from fast_agent.utils.count_display import plural_label
from fast_agent.utils.text import strip_str_to_none

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pydantic import ValidationInfo

logger = get_logger(__name__)

_ROUND_TRIP_YAML = YAML()
_ROUND_TRIP_YAML.preserve_quotes = True

DEFAULT_CARD_REGISTRIES = [
    "https://github.com/fast-agent-ai/card-packs",
]

DEFAULT_CARD_REGISTRY_URL = "https://github.com/fast-agent-ai/card-packs/blob/main/marketplace.json"

CARD_PACK_SOURCE_FILENAME = ".card-pack-source.json"
CARD_PACK_SOURCE_SCHEMA_VERSION = 1
CARD_PACK_MANIFEST_FILENAME = "card-pack.yaml"
LOCAL_REVISION = "local"

CardPackSourceOrigin = Literal["remote", "local"]
CardPackKind = Literal["card", "bundle"]
_CARD_PACK_KINDS: tuple[CardPackKind, ...] = ("card", "bundle")
CardPackUpdateStatus = (
    CommonMarketplaceUpdateStatus
    | Literal[
        "invalid_local_pack",
        "ownership_conflict",
    ]
)

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
CARD_PACK_PUBLISH_SUCCESS_STATUSES: frozenset[CardPackPublishStatus] = frozenset(
    {"published", "committed", "no_changes"}
)
CARD_PACK_PUBLISH_FAILURE_STATUSES: frozenset[CardPackPublishStatus] = frozenset({"publish_failed"})
CARD_PACK_PUBLISH_STATUS_LABELS: dict[CardPackPublishStatus, str] = {
    "published": "published",
    "committed": "committed locally",
    "no_changes": "no changes",
    "unmanaged": "unmanaged",
    "invalid_metadata": "invalid metadata",
    "source_unreachable": "source unavailable",
    "source_path_missing": "source path missing",
    "missing_managed_files": "missing managed files",
    "publish_failed": "publish failed",
}
CARD_PACK_PUBLISH_WARNING_STATUSES: frozenset[CardPackPublishStatus] = frozenset(
    status
    for status in CARD_PACK_PUBLISH_STATUS_LABELS
    if status not in CARD_PACK_PUBLISH_SUCCESS_STATUSES and status != "unmanaged"
)
CARD_PACK_PUBLISH_STATUS_STYLES: dict[CardPackPublishStatus, str | None] = {
    **{status: "green" for status in CARD_PACK_PUBLISH_SUCCESS_STATUSES},
    **{status: "yellow" for status in CARD_PACK_PUBLISH_WARNING_STATUSES},
}

CardPackHeadResolution = marketplace_source_models.SourceRevision[CardPackUpdateStatus]
CardPackPathResolution = marketplace_source_models.SourcePathOid[CardPackUpdateStatus]
CardPackHeadCache = dict[tuple[str, str | None], CardPackHeadResolution]
CardPackPathCache = dict[tuple[str, str | None, str, str], CardPackPathResolution]


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


def format_card_pack_publish_status(result: CardPackPublishResult) -> tuple[str, str | None]:
    status_text = CARD_PACK_PUBLISH_STATUS_LABELS[result.status]
    detail = clean_update_status_detail(result.detail)
    if detail:
        status_text = f"{status_text}: {detail}"

    return status_text, CARD_PACK_PUBLISH_STATUS_STYLES.get(result.status)


def is_card_pack_publish_success(status: CardPackPublishStatus) -> bool:
    return status in CARD_PACK_PUBLISH_SUCCESS_STATUSES


def is_card_pack_publish_failure(status: CardPackPublishStatus) -> bool:
    return status in CARD_PACK_PUBLISH_FAILURE_STATUSES


@dataclass(frozen=True)
class _PublishWorkspace:
    repo_root: Path
    retained_temp_dir: Path | None


@dataclass(frozen=True)
class CardPackManifest:
    schema_version: int
    name: str
    kind: CardPackKind
    version: str | None
    agent_cards: tuple[str, ...]
    tool_cards: tuple[str, ...]
    files: tuple[str, ...]
    model_references_required: tuple[str, ...] = ()
    model_references_recommended: tuple[str, ...] = ()
    plugins_required: tuple[str, ...] = ()
    plugins_recommended: tuple[str, ...] = ()


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

    @property
    def repo_subdir(self) -> str:
        return marketplace_provenance_io.repo_subdir_for_manifest_path(
            self.repo_path,
            CARD_PACK_MANIFEST_FILENAME,
        )


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
    model_references_required: list[str] = Field(default_factory=list)
    model_references_recommended: list[str] = Field(default_factory=list)
    plugins: dict[str, list[str]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")

    @field_validator("model_references_required", "model_references_recommended")
    @classmethod
    def _validate_model_alias_tokens(cls, value: list[str]) -> list[str]:
        normalized_tokens: list[str] = []
        for token in value:
            try:
                namespace, key = parse_model_reference_token(token)
            except ModelConfigError as exc:
                raise ValueError(exc.details) from exc
            normalized_tokens.append(f"${namespace}.{key}")
        return normalized_tokens


class MarketplaceEntryModel(MarketplaceEntryFieldsModel):
    @model_validator(mode="before")
    @classmethod
    def _normalize_entry(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict):
            return data

        context = info.context or {}
        default_repo_url = context.get("repo_url")
        default_repo_ref = context.get("repo_ref")
        default_source_url = context.get("source_url")

        repo_url = marketplace_source_urls.first_nonempty_str(
            data,
            "repo",
            "repository",
            "git",
            "repo_url",
        )
        repo_ref = marketplace_source_urls.first_nonempty_str(
            data,
            "ref",
            "branch",
            "tag",
            "revision",
            "commit",
        )
        repo_path = marketplace_source_urls.first_nonempty_str(
            data,
            "path",
            "repo_path",
            "directory",
            "dir",
            "location",
        )
        explicit_source_url = marketplace_source_urls.explicit_entry_source_url(
            data,
            "url",
            "source_url",
            default_source_url=default_source_url,
        )

        parsed = marketplace_source_urls.parse_github_url(repo_url) if repo_url else None
        if parsed and not repo_path:
            repo_url = parsed.repo_url
            repo_ref = parsed.repo_ref
            repo_path = parsed.repo_path
        elif parsed:
            repo_url = parsed.repo_url
            repo_ref = repo_ref or parsed.repo_ref

        source_parsed = (
            marketplace_source_urls.parse_github_url(explicit_source_url)
            if explicit_source_url
            else None
        )
        if source_parsed and (not repo_url or not repo_path):
            repo_url = source_parsed.repo_url
            repo_ref = source_parsed.repo_ref
            repo_path = source_parsed.repo_path
        elif (
            explicit_source_url
            and marketplace_source_urls.is_git_source_url(explicit_source_url)
            and (not repo_url or repo_url == default_repo_url)
        ):
            repo_url = explicit_source_url
        elif explicit_source_url and repo_path and not repo_url:
            repo_url = explicit_source_url

        name = marketplace_source_urls.first_nonempty_str(data, "name", "id", "slug", "title")
        description = marketplace_source_urls.first_nonempty_str(data, "description", "summary")
        kind = marketplace_source_urls.first_nonempty_str(data, "kind", "type")
        bundle_name = marketplace_source_urls.first_nonempty_str(data, "bundle_name")

        repo_url = repo_url or default_repo_url
        repo_ref = repo_ref or default_repo_ref
        source_url = explicit_source_url or default_source_url

        if not name and repo_path:
            name = _card_pack_name_from_repo_path(repo_path)

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
    def _normalize_payload(cls, data: Any, info: ValidationInfo) -> Any:
        return marketplace_fetch.normalize_marketplace_payload(
            data,
            info,
            extract_entries=_extract_marketplace_entries,
        )


def get_manager_directory(settings: Settings | None = None, *, cwd: Path | None = None) -> Path:
    resolved = settings or get_settings()
    home_paths = resolve_home_paths(resolved, cwd=cwd)
    return home_paths.card_packs


def get_marketplace_url(settings: Settings | None = None) -> str:
    resolved_settings = settings or get_settings()
    cards_settings = resolved_settings.cards
    url = cards_settings.marketplace_url
    if not url and cards_settings.marketplace_urls:
        url = cards_settings.marketplace_urls[0]
    return marketplace_source_urls.normalize_marketplace_url(url or DEFAULT_CARD_REGISTRY_URL)


def resolve_card_registries(settings: Settings | None = None) -> list[str]:
    resolved_settings = settings or get_settings()
    return marketplace_registry_urls.resolve_registry_urls(
        resolved_settings.cards.marketplace_urls,
        default_urls=DEFAULT_CARD_REGISTRIES,
        active_url=resolved_settings.cards.marketplace_url,
    )


def format_marketplace_display_url(url: str) -> str:
    return marketplace_registry_urls.format_marketplace_display_url(url)


def list_local_card_packs(*, home_paths: HomePaths) -> list[LocalCardPack]:
    destination_root = home_paths.card_packs.resolve()
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
    def names(entry: MarketplaceCardPack) -> list[str]:
        return [entry.name]

    return select_one_by_name_or_index(entries, selector, names=names)


def select_installed_card_pack_by_name_or_index(
    entries: Iterable[LocalCardPack], selector: str
) -> LocalCardPack | None:
    def names(entry: LocalCardPack) -> list[str]:
        return [entry.name, entry.pack_dir.name]

    return select_one_by_name_or_index(
        entries,
        selector,
        names=names,
    )


def get_card_pack_source_sidecar_path(pack_dir: Path) -> Path:
    return pack_dir / CARD_PACK_SOURCE_FILENAME


def load_card_pack_readme(pack_dir: Path) -> str | None:
    for candidate in ("README.md", "README.markdown", "readme.md", "readme.markdown"):
        path = pack_dir / candidate
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return None


def compute_card_pack_content_fingerprint(
    home_root: Path,
    installed_files: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    root = home_root.resolve()

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
    read_result = marketplace_provenance_io.read_installed_source_file(
        get_card_pack_source_sidecar_path(pack_dir),
        parse_payload=_parse_installed_card_pack_source,
    )
    return read_result.source, read_result.error


def write_installed_card_pack_source(pack_dir: Path, source: InstalledCardPackSource) -> None:
    marketplace_provenance_io.write_installed_source_file(
        get_card_pack_source_sidecar_path(pack_dir),
        source,
        extra_payload={
            "name": source.name,
            "kind": source.kind,
            "installed_files": list(source.installed_files),
        },
    )


def load_card_pack_manifest(pack_root: Path) -> CardPackManifest:
    manifest_path = pack_root / CARD_PACK_MANIFEST_FILENAME
    if not manifest_path.exists() or not manifest_path.is_file():
        raise FileNotFoundError(f"{CARD_PACK_MANIFEST_FILENAME} not found in {pack_root}")

    raw_text = manifest_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    if data is None:
        data = {}

    model = _CardPackManifestModel.model_validate(data)
    if model.schema_version not in {1, 2}:
        raise ValueError(f"Unsupported card pack schema_version: {model.schema_version}")

    agent_cards = tuple(
        _validate_manifest_install_path(entry) for entry in model.install.agent_cards
    )
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
        model_references_required=tuple(model.model_references_required),
        model_references_recommended=tuple(model.model_references_recommended),
        plugins_required=tuple(_validate_plugin_refs(model.plugins.get("required", []))),
        plugins_recommended=tuple(_validate_plugin_refs(model.plugins.get("recommended", []))),
    )


def _validate_plugin_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("card pack plugins.required/recommended must be lists")
    refs: list[str] = []
    for item in value:
        cleaned = strip_str_to_none(item)
        if cleaned is None:
            raise ValueError("card pack plugin references must be non-empty strings")
        if cleaned not in refs:
            refs.append(cleaned)
    return refs


async def fetch_marketplace_card_packs_with_source(
    url: str,
) -> tuple[list[MarketplaceCardPack], str]:
    return await marketplace_fetch.fetch_marketplace_entries_with_source(
        url,
        candidate_urls=candidate_marketplace_urls,
        normalize_url=marketplace_source_urls.normalize_marketplace_url,
        load_local_payload=marketplace_fetch.load_local_marketplace_payload,
        parse_payload=lambda payload, source_url: _parse_marketplace_payload(
            payload,
            source_url=source_url,
        ),
    )


async def install_marketplace_card_pack(
    pack: MarketplaceCardPack,
    *,
    home_paths: HomePaths,
    force: bool = False,
) -> CardPackInstallResult:
    return await run_in_thread(
        _install_marketplace_card_pack_sync,
        pack,
        home_paths,
        force,
        False,
        None,
    )


def remove_local_card_pack(
    pack_name: str,
    *,
    home_paths: HomePaths,
) -> CardPackRemovalResult:
    destination_root = home_paths.card_packs.resolve()
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
        target = (home_paths.root / relative).resolve()
        if owner_set != {source.name}:
            skipped_paths.append(relative)
            continue
        if target.exists() and target.is_file():
            target.unlink()
            removed_paths.append(relative)
            _prune_empty_parents(target.parent, stop_at=home_paths.root.resolve())

    shutil.rmtree(pack_dir)
    return CardPackRemovalResult(
        pack_name=source.name,
        removed_paths=tuple(sorted(removed_paths)),
        skipped_paths=tuple(sorted(skipped_paths)),
    )


def check_card_pack_updates(
    *,
    home_paths: HomePaths,
) -> list[CardPackUpdateInfo]:
    destination_root = home_paths.card_packs.resolve()
    if not destination_root.exists() or not destination_root.is_dir():
        return []

    owners = _collect_installed_file_owners(destination_root)
    updates: list[CardPackUpdateInfo] = []
    head_cache: CardPackHeadCache = {}
    path_cache: CardPackPathCache = {}

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
    return select_updates_by_name_or_index(
        updates,
        selector,
        names=lambda update: (update.name, update.pack_dir.name),
    )


def apply_card_pack_updates(
    updates: Sequence[CardPackUpdateInfo],
    *,
    home_paths: HomePaths,
    force: bool,
) -> list[CardPackUpdateInfo]:
    destination_root = home_paths.card_packs.resolve()
    owners = _collect_installed_file_owners(destination_root)
    head_cache: CardPackHeadCache = {}
    path_cache: CardPackPathCache = {}

    results: list[CardPackUpdateInfo] = []

    for update in updates:
        refreshed = _evaluate_card_pack_update(
            pack_dir=update.pack_dir,
            index=update.index,
            owners=owners,
            head_cache=head_cache,
            path_cache=path_cache,
        )

        if not is_update_applicable(refreshed.status):
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
            home_paths.root,
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
                home_paths,
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
        except Exception as exc:
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
    home_paths: HomePaths,
    push: bool = True,
    commit_message: str | None = None,
    temp_dir: Path | None = None,
    keep_temp: bool = False,
) -> CardPackPublishResult:
    source, error = read_installed_card_pack_source(pack_dir)
    if source is None:
        result = CardPackPublishResult(
            pack_name=pack_dir.name,
            pack_dir=pack_dir,
            status="unmanaged" if error is None else "invalid_metadata",
            detail="no sidecar metadata" if error is None else error,
        )
    else:
        with contextlib.ExitStack() as stack:
            result = _publish_managed_local_card_pack(
                source,
                pack_dir=pack_dir,
                home_paths=home_paths,
                push=push,
                commit_message=commit_message,
                temp_dir=temp_dir,
                keep_temp=keep_temp,
                stack=stack,
            )
    return result


def _publish_managed_local_card_pack(
    source: InstalledCardPackSource,
    *,
    pack_dir: Path,
    home_paths: HomePaths,
    push: bool,
    commit_message: str | None,
    temp_dir: Path | None,
    keep_temp: bool,
    stack: contextlib.ExitStack,
) -> CardPackPublishResult:
    workspace = _prepare_publish_workspace(
        source,
        pack_dir=pack_dir,
        temp_dir=temp_dir,
        keep_temp=keep_temp,
        stack=stack,
    )
    if isinstance(workspace, CardPackPublishResult):
        result = workspace
    else:
        destination = _resolve_publish_destination(source, pack_dir, workspace)
        if isinstance(destination, CardPackPublishResult):
            result = destination
        else:
            sync_result = _sync_publish_content(
                source,
                pack_dir=pack_dir,
                home_paths=home_paths,
                destination_pack_dir=destination,
                workspace=workspace,
            )
            if sync_result is not None:
                result = sync_result
            else:
                result = _finish_staged_publish(
                    source,
                    pack_dir=pack_dir,
                    home_paths=home_paths,
                    workspace=workspace,
                    push=push,
                    commit_message=commit_message,
                )
    return result


def _finish_staged_publish(
    source: InstalledCardPackSource,
    *,
    pack_dir: Path,
    home_paths: HomePaths,
    workspace: _PublishWorkspace,
    push: bool,
    commit_message: str | None,
) -> CardPackPublishResult:
    status = _stage_publish_changes(source, pack_dir, workspace)
    if isinstance(status, CardPackPublishResult):
        result = status
    elif not status:
        result = _finish_publish_without_changes(
            source,
            pack_dir=pack_dir,
            home_paths=home_paths,
            workspace=workspace,
        )
    else:
        commit = _commit_publish_changes(
            source,
            pack_dir=pack_dir,
            home_paths=home_paths,
            workspace=workspace,
            commit_message=commit_message,
        )
        if isinstance(commit, CardPackPublishResult):
            result = commit
        elif not push:
            result = _publish_result(
                source,
                pack_dir=pack_dir,
                workspace=workspace,
                status="committed",
                detail="changes committed locally; push skipped (--no-push)",
                commit=commit,
            )
        else:
            result = _push_published_card_pack(
                source,
                pack_dir=pack_dir,
                workspace=workspace,
                commit=commit,
            )
    return result


def _publish_result(
    source: InstalledCardPackSource,
    *,
    pack_dir: Path,
    workspace: _PublishWorkspace,
    status: CardPackPublishStatus,
    detail: str | None,
    commit: str | None = None,
    patch_path: Path | None = None,
) -> CardPackPublishResult:
    return CardPackPublishResult(
        pack_name=source.name,
        pack_dir=pack_dir,
        status=status,
        detail=detail,
        repo_root=workspace.repo_root,
        repo_path=source.repo_path,
        commit=commit,
        patch_path=patch_path,
        retained_temp_dir=workspace.retained_temp_dir,
    )


def _prepare_publish_workspace(
    source: InstalledCardPackSource,
    *,
    pack_dir: Path,
    temp_dir: Path | None,
    keep_temp: bool,
    stack: contextlib.ExitStack,
) -> _PublishWorkspace | CardPackPublishResult:
    local_repo = marketplace_git_sources.resolve_local_repo(source.repo_url)
    if local_repo is not None:
        return _PublishWorkspace(repo_root=local_repo, retained_temp_dir=None)

    temp_parent = temp_dir.expanduser().resolve() if temp_dir is not None else None
    if temp_parent is not None:
        temp_parent.mkdir(parents=True, exist_ok=True)

    retained_temp_dir: Path | None = None
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

    workspace = _PublishWorkspace(repo_root=repo_root, retained_temp_dir=retained_temp_dir)
    clone_error = _clone_publish_repository(source=source, destination_dir=repo_root)
    if clone_error is None:
        return workspace
    return _publish_result(
        source,
        pack_dir=pack_dir,
        workspace=workspace,
        status="source_unreachable",
        detail=clone_error,
    )


def _resolve_publish_destination(
    source: InstalledCardPackSource,
    pack_dir: Path,
    workspace: _PublishWorkspace,
) -> Path | CardPackPublishResult:
    try:
        return marketplace_git_sources.resolve_repo_subdir(
            workspace.repo_root,
            source.repo_path,
            label="Card pack",
        )
    except ValueError as exc:
        return _publish_result(
            source,
            pack_dir=pack_dir,
            workspace=workspace,
            status="source_path_missing",
            detail=str(exc),
        )


def _sync_publish_content(
    source: InstalledCardPackSource,
    *,
    pack_dir: Path,
    home_paths: HomePaths,
    destination_pack_dir: Path,
    workspace: _PublishWorkspace,
) -> CardPackPublishResult | None:
    try:
        manifest = load_card_pack_manifest(pack_dir)
        plan = _build_install_copy_plan(pack_dir, manifest, home_root=home_paths.root)
        missing_files = _sync_pack_from_environment(
            copy_plan=plan,
            home_root=home_paths.root,
        )
    except Exception as exc:
        return _publish_result(
            source,
            pack_dir=pack_dir,
            workspace=workspace,
            status="publish_failed",
            detail=str(exc),
        )

    if missing_files:
        return _publish_result(
            source,
            pack_dir=pack_dir,
            workspace=workspace,
            status="missing_managed_files",
            detail=_format_missing_installed_files_detail(missing_files),
        )

    try:
        _sync_directory_contents(
            source_root=pack_dir,
            target_root=destination_pack_dir,
            ignore_names={CARD_PACK_SOURCE_FILENAME, ".publish"},
        )
    except Exception as exc:
        return _publish_result(
            source,
            pack_dir=pack_dir,
            workspace=workspace,
            status="publish_failed",
            detail=str(exc),
        )
    return None


def _stage_publish_changes(
    source: InstalledCardPackSource,
    pack_dir: Path,
    workspace: _PublishWorkspace,
) -> str | CardPackPublishResult:
    _ensure_git_identity(workspace.repo_root)

    add_result = subprocess.run(
        ["git", "-C", str(workspace.repo_root), "add", "--all", "--", source.repo_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if add_result.returncode != 0:
        return _publish_result(
            source,
            pack_dir=pack_dir,
            workspace=workspace,
            status="publish_failed",
            detail=marketplace_git_sources.subprocess_failure_detail(
                add_result,
                "git add failed",
            ),
        )

    status_result = subprocess.run(
        ["git", "-C", str(workspace.repo_root), "status", "--porcelain", "--", source.repo_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if status_result.returncode != 0:
        return _publish_result(
            source,
            pack_dir=pack_dir,
            workspace=workspace,
            status="publish_failed",
            detail=marketplace_git_sources.subprocess_failure_detail(
                status_result,
                "git status failed",
            ),
        )
    return status_result.stdout.strip()


def _finish_publish_without_changes(
    source: InstalledCardPackSource,
    *,
    pack_dir: Path,
    home_paths: HomePaths,
    workspace: _PublishWorkspace,
) -> CardPackPublishResult:
    current_commit = marketplace_git_sources.resolve_git_commit(workspace.repo_root, "HEAD")
    if current_commit:
        metadata_error = _refresh_publish_sidecar_safely(
            pack_dir=pack_dir,
            source=source,
            home_paths=home_paths,
            repo_root=workspace.repo_root,
            commit=current_commit,
            detail_prefix="published content but failed to update metadata",
        )
        if metadata_error is not None:
            return _publish_result(
                source,
                pack_dir=pack_dir,
                workspace=workspace,
                status="publish_failed",
                detail=metadata_error,
                commit=current_commit,
            )
    return _publish_result(
        source,
        pack_dir=pack_dir,
        workspace=workspace,
        status="no_changes",
        detail="source repository already matches local pack",
        commit=current_commit,
    )


def _commit_publish_changes(
    source: InstalledCardPackSource,
    *,
    pack_dir: Path,
    home_paths: HomePaths,
    workspace: _PublishWorkspace,
    commit_message: str | None,
) -> str | CardPackPublishResult | None:
    message = commit_message or f"Update card pack {source.name}"
    commit_result = subprocess.run(
        ["git", "-C", str(workspace.repo_root), "commit", "-m", message],
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_result.returncode != 0:
        return _publish_result(
            source,
            pack_dir=pack_dir,
            workspace=workspace,
            status="publish_failed",
            detail=marketplace_git_sources.subprocess_failure_detail(
                commit_result,
                "git commit failed",
            ),
        )

    commit = marketplace_git_sources.resolve_git_commit(workspace.repo_root, "HEAD")
    if not commit:
        return commit

    metadata_error = _refresh_publish_sidecar_safely(
        pack_dir=pack_dir,
        source=source,
        home_paths=home_paths,
        repo_root=workspace.repo_root,
        commit=commit,
        detail_prefix="committed but failed to update metadata",
    )
    if metadata_error is None:
        return commit
    return _publish_result(
        source,
        pack_dir=pack_dir,
        workspace=workspace,
        status="publish_failed",
        detail=metadata_error,
        commit=commit,
    )


def _push_published_card_pack(
    source: InstalledCardPackSource,
    *,
    pack_dir: Path,
    workspace: _PublishWorkspace,
    commit: str | None,
) -> CardPackPublishResult:
    push_result = subprocess.run(
        ["git", "-C", str(workspace.repo_root), "push"],
        capture_output=True,
        text=True,
        check=False,
    )
    if push_result.returncode == 0:
        return _publish_result(
            source,
            pack_dir=pack_dir,
            workspace=workspace,
            status="published",
            detail="changes pushed to remote",
            commit=commit,
        )

    push_error = marketplace_git_sources.subprocess_failure_detail(
        push_result,
        "git push failed",
    )
    patch_path = _write_publish_patch(
        repo_root=workspace.repo_root,
        pack_dir=pack_dir,
        commit=commit,
    )
    return _publish_result(
        source,
        pack_dir=pack_dir,
        workspace=workspace,
        status="publish_failed",
        detail=f"push failed: {push_error}" if push_error else "push failed",
        commit=commit,
        patch_path=patch_path,
    )


def _refresh_publish_sidecar_safely(
    *,
    pack_dir: Path,
    source: InstalledCardPackSource,
    home_paths: HomePaths,
    repo_root: Path,
    commit: str,
    detail_prefix: str,
) -> str | None:
    try:
        _refresh_published_sidecar(
            pack_dir=pack_dir,
            source=source,
            home_paths=home_paths,
            repo_root=repo_root,
            commit=commit,
        )
    except Exception as exc:
        return f"{detail_prefix}: {exc}"
    return None


def format_revision_short(revision: str | None) -> str:
    return marketplace_formatting.format_revision_short(revision)


def format_installed_at_display(installed_at: str | None) -> str:
    return marketplace_formatting.format_installed_at_display(installed_at)


def _install_marketplace_card_pack_sync(
    pack: MarketplaceCardPack,
    home_paths: HomePaths,
    force: bool,
    replace_existing: bool,
    pinned_revision: str | None,
) -> CardPackInstallResult:
    destination_root = home_paths.card_packs.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    home_paths.agent_cards.mkdir(parents=True, exist_ok=True)
    home_paths.tool_cards.mkdir(parents=True, exist_ok=True)

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
        copied_source = _copy_pack_from_marketplace_source(
            pack,
            destination_dir=staged_pack_dir,
            pinned_revision=pinned_revision,
        )

        manifest = load_card_pack_manifest(staged_pack_dir)
        plan = _build_install_copy_plan(
            staged_pack_dir,
            manifest,
            home_root=home_paths.root,
        )
        plan, mergeable_unmanaged_files = _merge_last_used_model_into_copy_plan(
            copy_plan=plan,
            temp_dir=temp_dir,
            home_root=home_paths.root,
            owners=owners,
            current_pack=pack.name,
            current_owned_files=current_owned_files,
        )

        conflicts, overwritten_by_owner = _collect_install_conflicts(
            copy_plan=plan,
            home_root=home_paths.root,
            owners=owners,
            current_pack=pack.name,
            current_owned_files=current_owned_files,
            mergeable_unmanaged_files=mergeable_unmanaged_files,
            force=force,
        )
        if conflicts:
            raise OwnershipConflictError("; ".join(conflicts))

        installed_files = tuple(sorted(item.destination_relative for item in plan))
        _apply_copy_plan(
            copy_plan=plan,
            home_root=home_paths.root,
            current_owned_files=current_owned_files,
            new_owned_files=set(installed_files),
        )

        _revoke_overwritten_ownership(
            home_paths=home_paths,
            overwritten_by_owner=overwritten_by_owner,
        )

        fingerprint = compute_card_pack_content_fingerprint(
            home_paths.root,
            installed_files,
        )
        source = _build_installed_card_pack_source(
            pack=pack,
            source_origin=copied_source.origin,
            installed_commit=copied_source.commit,
            installed_path_oid=copied_source.path_oid,
            fingerprint=fingerprint,
            installed_files=installed_files,
        )
        write_installed_card_pack_source(staged_pack_dir, source)

        if install_root.exists():
            marketplace_git_sources.atomic_replace_directory(
                existing_dir=install_root,
                staged_dir=staged_pack_dir,
            )
        else:
            staged_pack_dir.replace(install_root)

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
) -> marketplace_source_models.SourceCopyResult[CardPackSourceOrigin]:
    checkout_ref = marketplace_git_sources.pinned_checkout_ref(
        pinned_revision,
        local_revision=LOCAL_REVISION,
    )
    local_repo = marketplace_git_sources.resolve_local_repo(pack.repo_url)
    if local_repo is not None:
        requested_revision = checkout_ref or pack.repo_ref
        if requested_revision:
            commit = marketplace_git_sources.resolve_git_commit(local_repo, requested_revision)
            if commit is None:
                raise FileNotFoundError(f"Card pack source ref not found: {requested_revision}")
            _copy_pack_source_from_git_commit(
                repo_root=local_repo,
                commit=commit,
                repo_path=pack.repo_subdir,
                destination_dir=destination_dir,
            )
        else:
            source_dir = marketplace_git_sources.resolve_repo_subdir(
                local_repo,
                pack.repo_subdir,
                label="Card pack",
            )
            _validate_pack_source_dir(source_dir, pack.repo_subdir)
            shutil.copytree(source_dir, destination_dir)
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
            pack.repo_subdir,
            resolve_git_path_oid_fn=marketplace_git_sources.resolve_git_path_oid,
        )
        return marketplace_source_models.SourceCopyResult(
            origin="local",
            commit=commit,
            path_oid=path_oid,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        marketplace_git_sources.clone_sparse_checkout(
            repo_url=pack.repo_url,
            repo_ref=pack.repo_ref,
            repo_subdir=pack.repo_subdir,
            destination_dir=tmp_path,
            checkout_ref=checkout_ref,
        )

        source_dir = marketplace_git_sources.resolve_repo_subdir(
            tmp_path,
            pack.repo_subdir,
            label="Card pack",
        )
        _validate_pack_source_dir(source_dir, pack.repo_subdir)

        shutil.copytree(source_dir, destination_dir)

        commit = marketplace_git_sources.resolve_git_commit(tmp_path, "HEAD")
        path_oid = marketplace_git_sources.resolve_git_path_oid_if_commit(
            tmp_path,
            commit,
            pack.repo_subdir,
            resolve_git_path_oid_fn=marketplace_git_sources.resolve_git_path_oid,
        )
        return marketplace_source_models.SourceCopyResult(
            origin="remote",
            commit=commit,
            path_oid=path_oid,
        )


def _validate_pack_source_dir(source_dir: Path, repo_path: str) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Card pack path not found in repository: {repo_path}")
    if not (source_dir / CARD_PACK_MANIFEST_FILENAME).is_file():
        raise FileNotFoundError(
            f"{CARD_PACK_MANIFEST_FILENAME} not found in repository path: {repo_path}"
        )


def _copy_pack_source_from_git_commit(
    *,
    repo_root: Path,
    commit: str,
    repo_path: str,
    destination_dir: Path,
) -> None:
    marketplace_git_sources.copy_git_path_from_commit(
        repo_root=repo_root,
        commit=commit,
        repo_subdir=repo_path,
        destination_dir=destination_dir,
        missing_message=f"Card pack source path not found at revision {commit}: {repo_path}",
    )
    _validate_pack_source_dir(destination_dir, repo_path)


def _build_install_copy_plan(
    pack_root: Path,
    manifest: CardPackManifest,
    *,
    home_root: Path,
) -> list[_PlannedCopy]:
    plan: list[_PlannedCopy] = []

    for entry in manifest.agent_cards:
        source = _resolve_pack_source_path(pack_root, entry)
        destination_relative = str(PurePosixPath("agent-cards") / PurePosixPath(entry).name)
        _ensure_home_target_path(destination_relative, home_root)
        plan.append(_PlannedCopy(source=source, destination_relative=destination_relative))

    for entry in manifest.tool_cards:
        source = _resolve_pack_source_path(pack_root, entry)
        destination_relative = str(PurePosixPath("tool-cards") / PurePosixPath(entry).name)
        _ensure_home_target_path(destination_relative, home_root)
        plan.append(_PlannedCopy(source=source, destination_relative=destination_relative))

    for entry in manifest.files:
        source = _resolve_pack_source_path(pack_root, entry)
        destination_relative = _resolve_manifest_file_install_path(
            _validate_manifest_install_path(entry),
            home_root=home_root,
        )
        _ensure_home_target_path(destination_relative, home_root)
        plan.append(_PlannedCopy(source=source, destination_relative=destination_relative))

    deduped: dict[str, _PlannedCopy] = {}
    for item in plan:
        deduped[item.destination_relative] = item
    return [deduped[key] for key in sorted(deduped.keys())]


def _collect_install_conflicts(
    *,
    copy_plan: Sequence[_PlannedCopy],
    home_root: Path,
    owners: dict[str, set[str]],
    current_pack: str,
    current_owned_files: set[str],
    mergeable_unmanaged_files: set[str],
    force: bool,
) -> tuple[list[str], dict[str, set[str]]]:
    conflicts: list[str] = []
    overwritten_by_owner: dict[str, set[str]] = defaultdict(set)

    for item in copy_plan:
        relative = item.destination_relative
        owner_set = set(owners.get(relative, set()))
        owner_set.discard(current_pack)

        target = (home_root / relative).resolve()

        if owner_set and not force:
            owner_list = ", ".join(sorted(owner_set))
            conflicts.append(f"{relative} is owned by another pack: {owner_list}")
            continue

        if (
            target.exists()
            and not owner_set
            and relative not in current_owned_files
            and relative not in mergeable_unmanaged_files
        ):
            conflicts.append(f"{relative} already exists and is unmanaged")
            continue

        if owner_set and force:
            for owner in owner_set:
                overwritten_by_owner[owner].add(relative)

    return conflicts, overwritten_by_owner


def _merge_last_used_model_into_copy_plan(
    *,
    copy_plan: Sequence[_PlannedCopy],
    temp_dir: Path,
    home_root: Path,
    owners: dict[str, set[str]],
    current_pack: str,
    current_owned_files: set[str],
) -> tuple[list[_PlannedCopy], set[str]]:
    merged_plan: list[_PlannedCopy] = []
    mergeable_unmanaged_files: set[str] = set()

    for item in copy_plan:
        if item.destination_relative not in CONFIG_FILENAMES:
            merged_plan.append(item)
            continue

        target = (home_root / item.destination_relative).resolve()
        if not target.exists() or not target.is_file():
            merged_plan.append(item)
            continue

        owner_set = set(owners.get(item.destination_relative, set()))
        owner_set.discard(current_pack)
        if owner_set:
            merged_plan.append(item)
            continue

        is_unmanaged_target = item.destination_relative not in current_owned_files
        preserved_last_used = _extract_installable_last_used_model(
            target,
            require_last_used_only=is_unmanaged_target,
        )
        if preserved_last_used is None:
            merged_plan.append(item)
            continue

        merged_source = temp_dir / f".merged-{Path(item.destination_relative).name}"
        _write_merged_last_used_config(
            source_path=item.source,
            destination_path=merged_source,
            last_used_model=preserved_last_used,
        )
        merged_plan.append(
            _PlannedCopy(
                source=merged_source,
                destination_relative=item.destination_relative,
            )
        )
        if is_unmanaged_target:
            mergeable_unmanaged_files.add(item.destination_relative)

    return merged_plan, mergeable_unmanaged_files


def _resolve_manifest_file_install_path(relative: str, *, home_root: Path) -> str:
    if relative not in CONFIG_FILENAMES:
        return relative

    existing_config = find_config_in_directory(home_root)
    if existing_config is not None:
        return existing_config.name

    return PREFERRED_CONFIG_FILENAME


def _extract_installable_last_used_model(
    config_path: Path,
    *,
    require_last_used_only: bool,
) -> str | None:
    payload = _load_yaml_mapping(config_path)
    last_used_model = _system_last_used_model_reference(payload)
    if payload is None or last_used_model is None:
        return None

    if require_last_used_only and not _is_last_used_only_config(payload):
        return None

    return last_used_model


def _load_yaml_mapping(config_path: Path) -> dict[str, Any] | None:
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _system_last_used_model_reference(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    model_references = payload.get("model_references")
    if not isinstance(model_references, dict):
        return None

    system_references = model_references.get("system")
    if not isinstance(system_references, dict):
        return None

    raw_last_used = system_references.get("last_used")
    if not isinstance(raw_last_used, str):
        return None

    last_used_model = raw_last_used.strip()
    if not last_used_model:
        return None
    return last_used_model


def _is_last_used_only_config(payload: dict[str, Any]) -> bool:
    normalized_payload = _prune_empty_config_nodes(payload)
    return normalized_payload == {
        "model_references": {
            "system": {
                "last_used": normalized_payload["model_references"]["system"]["last_used"],
            }
        }
    }


def _prune_empty_config_nodes(value: Any) -> Any:
    if isinstance(value, dict):
        normalized_mapping: dict[str, Any] = {}
        for key, child in value.items():
            normalized_child = _prune_empty_config_nodes(child)
            if normalized_child in ({}, [], None):
                continue
            normalized_mapping[key] = normalized_child
        return normalized_mapping

    if isinstance(value, list):
        return [
            normalized_child
            for child in value
            if (normalized_child := _prune_empty_config_nodes(child)) not in ({}, [], None)
        ]

    return value


def _write_merged_last_used_config(
    *,
    source_path: Path,
    destination_path: Path,
    last_used_model: str,
) -> None:
    document = _load_round_trip_mapping(source_path)
    _set_last_used_model_reference(document, last_used_model)
    _write_round_trip_mapping(document, destination_path)


def _load_round_trip_mapping(path: Path) -> CommentedMap:
    with path.open("r", encoding="utf-8") as handle:
        payload = _ROUND_TRIP_YAML.load(handle)

    if payload is None:
        return CommentedMap()
    if isinstance(payload, CommentedMap):
        return payload
    if isinstance(payload, dict):
        return CommentedMap(payload)
    raise ValueError(f"Top-level YAML at {path} must be a mapping")


def _set_last_used_model_reference(document: CommentedMap, last_used_model: str) -> None:
    model_references = document.get("model_references")
    if not isinstance(model_references, dict):
        model_references = CommentedMap()
        document["model_references"] = model_references

    system_references = model_references.get("system")
    if not isinstance(system_references, dict):
        system_references = CommentedMap()
        model_references["system"] = system_references

    system_references["last_used"] = last_used_model


def _write_round_trip_mapping(document: CommentedMap, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        _ROUND_TRIP_YAML.dump(document, handle)


def _apply_copy_plan(
    *,
    copy_plan: Sequence[_PlannedCopy],
    home_root: Path,
    current_owned_files: set[str],
    new_owned_files: set[str],
) -> None:
    for item in copy_plan:
        target = (home_root / item.destination_relative).resolve()
        _atomic_copy_file(item.source, target)

    stale_files = sorted(current_owned_files - new_owned_files)
    for relative in stale_files:
        target = (home_root / relative).resolve()
        if target.exists() and target.is_file():
            target.unlink()
            _prune_empty_parents(target.parent, stop_at=home_root.resolve())


def _sync_pack_from_environment(
    *,
    copy_plan: Sequence[_PlannedCopy],
    home_root: Path,
) -> list[str]:
    missing: list[str] = []
    home_root_resolved = home_root.resolve()
    for item in copy_plan:
        env_file = (home_root_resolved / item.destination_relative).resolve()
        try:
            env_file.relative_to(home_root_resolved)
        except ValueError:
            missing.append(item.destination_relative)
            continue

        if not env_file.exists() or not env_file.is_file():
            missing.append(item.destination_relative)
            continue

        _atomic_copy_file(env_file, item.source)

    return missing


def _format_missing_installed_files_detail(missing_files: Sequence[str]) -> str:
    preview = ", ".join(missing_files[:3])
    if len(missing_files) > 3:
        preview = f"{preview}, ..."
    file_label = plural_label(len(missing_files), "file")
    return f"missing installed {file_label} in environment: {preview}"


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
        temp_path.replace(target)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _revoke_overwritten_ownership(
    *,
    home_paths: HomePaths,
    overwritten_by_owner: dict[str, set[str]],
) -> None:
    if not overwritten_by_owner:
        return

    destination_root = home_paths.card_packs.resolve()
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
        updated = replace(
            source,
            content_fingerprint=compute_card_pack_content_fingerprint(
                home_paths.root,
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
    head_cache: CardPackHeadCache,
    path_cache: CardPackPathCache,
) -> CardPackUpdateInfo:
    source, error = read_installed_card_pack_source(pack_dir)
    if source is None:
        result = CardPackUpdateInfo(
            index=index,
            name=pack_dir.name,
            pack_dir=pack_dir,
            status="unmanaged" if error is None else "invalid_metadata",
            detail="no sidecar metadata" if error is None else error,
        )
    else:
        result = _evaluate_managed_card_pack_update(
            source,
            index=index,
            pack_dir=pack_dir,
            owners=owners,
            head_cache=head_cache,
            path_cache=path_cache,
        )
    return result


def _evaluate_managed_card_pack_update(
    source: InstalledCardPackSource,
    *,
    index: int,
    pack_dir: Path,
    owners: dict[str, set[str]],
    head_cache: CardPackHeadCache,
    path_cache: CardPackPathCache,
) -> CardPackUpdateInfo:
    preflight_update = _managed_card_pack_update_preflight(
        source,
        index=index,
        pack_dir=pack_dir,
        owners=owners,
    )
    if preflight_update is not None:
        result = preflight_update
    else:
        revision = _resolve_available_card_pack_revision(
            source,
            index=index,
            pack_dir=pack_dir,
            head_cache=head_cache,
        )
        if isinstance(revision, CardPackUpdateInfo):
            result = revision
        else:
            result = _evaluate_card_pack_path_update(
                source,
                index=index,
                pack_dir=pack_dir,
                available_revision=revision,
                path_cache=path_cache,
            )
    return result


def _managed_card_pack_update_preflight(
    source: InstalledCardPackSource,
    *,
    index: int,
    pack_dir: Path,
    owners: dict[str, set[str]],
) -> CardPackUpdateInfo | None:
    if not (pack_dir / CARD_PACK_MANIFEST_FILENAME).is_file():
        return _managed_card_pack_update(
            source,
            index=index,
            pack_dir=pack_dir,
            status="invalid_local_pack",
            detail=f"{CARD_PACK_MANIFEST_FILENAME} not found",
        )

    invalid_update = _preflight_card_pack_update(
        source,
        index=index,
        pack_dir=pack_dir,
        owners=owners,
    )
    if invalid_update is not None:
        return invalid_update

    if source.installed_commit is None and source.installed_revision == LOCAL_REVISION:
        return _managed_card_pack_update(
            source,
            index=index,
            pack_dir=pack_dir,
            status="unknown_revision",
            detail="source is local non-git; compare unavailable",
            available_revision=source.installed_revision,
        )
    return None


def _resolve_available_card_pack_revision(
    source: InstalledCardPackSource,
    *,
    index: int,
    pack_dir: Path,
    head_cache: CardPackHeadCache,
) -> str | CardPackUpdateInfo:
    resolved_revision = _resolve_source_revision(
        source,
        head_cache,
    )
    if resolved_revision.status is not None:
        return _managed_card_pack_update(
            source,
            index=index,
            pack_dir=pack_dir,
            status=resolved_revision.status,
            detail=resolved_revision.detail,
        )

    available_revision = resolved_revision.revision
    if available_revision is None:
        return _managed_card_pack_update(
            source,
            index=index,
            pack_dir=pack_dir,
            status="source_unreachable",
            detail="unable to resolve source revision",
        )
    return available_revision


def _evaluate_card_pack_path_update(
    source: InstalledCardPackSource,
    *,
    index: int,
    pack_dir: Path,
    available_revision: str,
    path_cache: CardPackPathCache,
) -> CardPackUpdateInfo:
    available_path = _resolve_source_path_oid(
        source,
        available_revision,
        path_cache,
    )
    if available_path.status is not None:
        return _managed_card_pack_update(
            source,
            index=index,
            pack_dir=pack_dir,
            status=available_path.status,
            detail=available_path.detail,
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
        content_changed_detail="card pack content changed",
    )

    return CardPackUpdateInfo(
        index=index,
        name=source.name,
        pack_dir=pack_dir,
        status=decision.status,
        detail=decision.detail,
        current_revision=current_revision,
        available_revision=available_revision,
        managed_source=source,
    )


def _managed_card_pack_update(
    source: InstalledCardPackSource,
    *,
    index: int,
    pack_dir: Path,
    status: CardPackUpdateStatus,
    detail: str | None,
    current_revision: str | None = None,
    available_revision: str | None = None,
) -> CardPackUpdateInfo:
    return CardPackUpdateInfo(
        index=index,
        name=source.name,
        pack_dir=pack_dir,
        status=status,
        detail=detail,
        current_revision=current_revision or source.installed_revision,
        available_revision=available_revision,
        managed_source=source,
    )


def _preflight_card_pack_update(
    source: InstalledCardPackSource,
    *,
    index: int,
    pack_dir: Path,
    owners: dict[str, set[str]],
) -> CardPackUpdateInfo | None:
    conflicting_paths = _conflicting_installed_paths(source, owners)
    if conflicting_paths:
        return _managed_card_pack_update(
            source,
            index=index,
            pack_dir=pack_dir,
            status="ownership_conflict",
            detail=f"ownership overlaps detected: {_preview_paths(conflicting_paths)}",
        )

    source_path_error = _validate_source_path_exists(source)
    if source_path_error is not None:
        return _managed_card_pack_update(
            source,
            index=index,
            pack_dir=pack_dir,
            status="source_path_missing",
            detail=source_path_error,
        )

    return None


def _conflicting_installed_paths(
    source: InstalledCardPackSource,
    owners: dict[str, set[str]],
) -> list[str]:
    return [path for path in source.installed_files if owners.get(path, set()) - {source.name}]


def _preview_paths(paths: Sequence[str]) -> str:
    preview = ", ".join(paths[:3])
    if len(paths) > 3:
        return f"{preview}, ..."
    return preview


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
    parsed = marketplace_provenance_io.parse_installed_source_fields(
        payload,
        expected_schema_version=CARD_PACK_SOURCE_SCHEMA_VERSION,
        normalize_repo_path=_normalize_repo_path,
    )

    name_value = strip_str_to_none(payload.get("name"))
    if name_value is None:
        raise ValueError("name is required")

    kind_raw = payload.get("kind")
    kind = _normalize_card_pack_kind(kind_raw)
    if kind is None:
        raise ValueError("kind must be 'card' or 'bundle'")

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
        source_origin=parsed.source_origin,
        name=name_value,
        kind=kind,
        repo_url=parsed.repo_url,
        repo_ref=parsed.repo_ref,
        repo_path=parsed.repo_path,
        source_url=parsed.source_url,
        installed_commit=parsed.installed_commit,
        installed_path_oid=parsed.installed_path_oid,
        installed_revision=parsed.installed_revision,
        installed_at=parsed.installed_at,
        content_fingerprint=parsed.content_fingerprint,
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
        repo_path=pack.repo_subdir,
        source_url=pack.source_url,
        installed_commit=installed_commit,
        installed_path_oid=installed_path_oid,
        installed_revision=installed_revision,
        installed_at=marketplace_formatting.iso_utc_now(),
        content_fingerprint=fingerprint,
        installed_files=tuple(sorted(installed_files)),
    )


def _refresh_published_sidecar(
    *,
    pack_dir: Path,
    source: InstalledCardPackSource,
    home_paths: HomePaths,
    repo_root: Path,
    commit: str,
) -> None:
    installed_path_oid = marketplace_git_sources.resolve_git_path_oid(
        repo_root,
        commit,
        source.repo_path,
    )
    fingerprint = compute_card_pack_content_fingerprint(
        home_paths.root,
        source.installed_files,
    )
    updated = replace(
        source,
        installed_commit=commit,
        installed_path_oid=installed_path_oid,
        installed_revision=commit,
        installed_at=marketplace_formatting.iso_utc_now(),
        content_fingerprint=fingerprint,
    )
    write_installed_card_pack_source(pack_dir, updated)


def _validate_source_path_exists(source: InstalledCardPackSource) -> str | None:
    local_repo = marketplace_git_sources.resolve_local_repo(source.repo_url)
    if local_repo is None:
        return None

    try:
        source_dir = marketplace_git_sources.resolve_repo_subdir(
            local_repo,
            source.repo_path,
            label="Card pack",
        )
    except ValueError as exc:
        return str(exc)

    try:
        _validate_pack_source_dir(source_dir, source.repo_path)
    except FileNotFoundError as exc:
        return str(exc)

    return None


def _resolve_source_revision(
    source: InstalledCardPackSource,
    head_cache: CardPackHeadCache,
) -> CardPackHeadResolution:
    return marketplace_git_sources.resolve_source_revision(
        repo_url=source.repo_url,
        repo_ref=source.repo_ref,
        head_cache=head_cache,
        local_revision=LOCAL_REVISION,
        source_ref_missing_status="source_ref_missing",
        source_unreachable_status="source_unreachable",
        resolve_local_repo_fn=marketplace_git_sources.resolve_local_repo,
        resolve_git_commit_fn=marketplace_git_sources.resolve_git_commit,
    )


def _resolve_source_path_oid(
    source: InstalledCardPackSource,
    commit: str,
    path_cache: CardPackPathCache,
) -> CardPackPathResolution:
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


def candidate_marketplace_urls(url: str) -> list[str]:
    return marketplace_source_urls.candidate_marketplace_urls(url)


def _parse_marketplace_payload(
    payload: Any,
    *,
    source_url: str | None = None,
) -> list[MarketplaceCardPack]:
    source_context = marketplace_fetch.marketplace_source_context(source_url)

    try:
        model = MarketplacePayloadModel.model_validate(
            payload,
            context=source_context.as_validation_context(),
        )
    except ValidationError as exc:
        logger.warning("Failed to parse card marketplace payload", data={"error": str(exc)})
        return []

    entries: list[MarketplaceCardPack] = []
    for entry in model.entries:
        parsed_entry = _card_pack_from_entry_model(entry)
        if parsed_entry is not None:
            entries.append(parsed_entry)
    return entries


def _card_pack_from_entry_model(model: MarketplaceEntryModel) -> MarketplaceCardPack | None:
    if not model.repo_url or not model.repo_path:
        return None

    repo_path = _normalize_repo_path(model.repo_path)
    if not repo_path:
        return None

    kind = _normalize_card_pack_kind(model.kind) or "card"

    return MarketplaceCardPack(
        name=model.name or _card_pack_name_from_repo_path(repo_path),
        description=model.description,
        kind=kind,
        repo_url=model.repo_url,
        repo_ref=model.repo_ref,
        repo_path=repo_path,
        source_url=model.source_url,
        bundle_name=model.bundle_name,
    )


def _card_pack_name_from_repo_path(repo_path: str) -> str:
    return marketplace_provenance_io.repo_name_for_manifest_path(
        repo_path,
        CARD_PACK_MANIFEST_FILENAME,
    )


def _normalize_card_pack_kind(value: object) -> CardPackKind | None:
    if not isinstance(value, str):
        return None
    normalized = normalize_action_token(value)
    if normalized == "card":
        return "card"
    if normalized == "bundle":
        return "bundle"
    return None


def _extract_marketplace_entries(payload: Any) -> list[dict[str, Any]]:
    return marketplace_fetch.extract_dict_entries(
        payload,
        ("card_packs", "cards", "entries", "items", "marketplace", "plugins"),
        allow_mapping_values=True,
    )


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


def _ensure_home_target_path(relative_path: str, home_root: Path) -> None:
    normalized = _normalize_repo_path(relative_path)
    if not normalized:
        raise ValueError(f"Invalid install target path: {relative_path}")
    target = (home_root / normalized).resolve()
    try:
        target.relative_to(home_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Install target escapes environment root: {relative_path}") from exc


def _validate_manifest_install_path(value: str) -> str:
    normalized = _normalize_repo_path(value)
    if not normalized:
        raise ValueError(f"Invalid install path: {value}")
    return normalized


def _normalize_repo_path(path: str) -> str | None:
    return marketplace_provenance_io.normalize_relative_repo_path(path)


def _clone_publish_repository(
    *, source: InstalledCardPackSource, destination_dir: Path
) -> str | None:
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
        return marketplace_git_sources.subprocess_failure_detail(
            clone_result,
            "git clone failed",
        )

    sparse_result = subprocess.run(
        ["git", "-C", str(destination_dir), "sparse-checkout", "set", source.repo_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if sparse_result.returncode != 0:
        return marketplace_git_sources.subprocess_failure_detail(
            sparse_result,
            "git sparse-checkout failed",
        )

    checkout_target = source.repo_ref or "HEAD"
    checkout_result = subprocess.run(
        ["git", "-C", str(destination_dir), "checkout", checkout_target],
        capture_output=True,
        text=True,
        check=False,
    )
    if checkout_result.returncode != 0:
        return marketplace_git_sources.subprocess_failure_detail(
            checkout_result,
            "git checkout failed",
        )

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


def _prune_empty_parents(path: Path, *, stop_at: Path) -> None:
    current = path.resolve()
    root = stop_at.resolve()
    while current != root and root in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent
