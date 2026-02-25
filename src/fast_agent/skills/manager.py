from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fast_agent.config import Settings, get_settings
from fast_agent.core.logging.logger import get_logger
from fast_agent.marketplace import formatting as marketplace_formatting
from fast_agent.marketplace import registry_urls as marketplace_registry_urls
from fast_agent.marketplace import source_utils as marketplace_source_utils
from fast_agent.paths import default_skill_paths
from fast_agent.skills.registry import SkillManifest, SkillRegistry

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = get_logger(__name__)

DEFAULT_SKILL_REGISTRIES = [
    "https://github.com/fast-agent-ai/skills",
    "https://github.com/huggingface/skills",
    "https://github.com/anthropics/skills",
]

DEFAULT_MARKETPLACE_URL = (
    "https://github.com/fast-agent-ai/skills/blob/main/marketplace.json"
)

SKILL_SOURCE_FILENAME = ".skill-source.json"
SKILL_SOURCE_SCHEMA_VERSION = 1
LOCAL_REVISION = "local"

SkillSourceOrigin = Literal["remote", "local"]
SkillUpdateStatus = Literal[
    "up_to_date",
    "update_available",
    "updated",
    "unmanaged",
    "invalid_metadata",
    "invalid_local_skill",
    "unknown_revision",
    "source_unreachable",
    "source_ref_missing",
    "source_path_missing",
    "skipped_dirty",
]


@dataclass(frozen=True)
class InstalledSkillSource:
    schema_version: int
    installed_via: str
    source_origin: SkillSourceOrigin
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None
    installed_commit: str | None
    installed_path_oid: str | None
    installed_revision: str
    installed_at: str
    content_fingerprint: str


@dataclass(frozen=True)
class SkillProvenance:
    status: Literal["managed", "unmanaged", "invalid_metadata"]
    summary: str
    source: InstalledSkillSource | None = None
    error: str | None = None


@dataclass(frozen=True)
class SkillUpdateInfo:
    index: int
    name: str
    skill_dir: Path
    status: SkillUpdateStatus
    detail: str | None = None
    current_revision: str | None = None
    available_revision: str | None = None
    managed_source: InstalledSkillSource | None = None


@dataclass(frozen=True)
class MarketplaceSkill:
    name: str
    description: str | None
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None = None
    bundle_name: str | None = None
    bundle_description: str | None = None

    @property
    def repo_subdir(self) -> str:
        path = PurePosixPath(self.repo_path)
        if path.name.lower() == "skill.md":
            return str(path.parent)
        return str(path)

    @property
    def install_dir_name(self) -> str:
        path = PurePosixPath(self.repo_path)
        if path.name.lower() == "skill.md":
            return path.parent.name or self.name
        return path.name or self.name


class MarketplaceEntryModel(BaseModel):
    name: str | None = None
    description: str | None = None
    repo_url: str | None = Field(default=None, alias="repo")
    repo_ref: str | None = None
    repo_path: str | None = None
    source_url: str | None = None
    bundle_name: str | None = None
    bundle_description: str | None = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_entry(cls, data: Any, info: Any) -> Any:
        if not isinstance(data, dict):
            return data

        context = getattr(info, "context", None) or {}
        default_repo_url = context.get("repo_url")
        default_repo_ref = context.get("repo_ref")

        repo_url = _first_str(data, "repo", "repository", "git", "repo_url")
        repo_ref = _first_str(data, "ref", "branch", "tag", "revision", "commit")
        repo_path = _first_str(
            data,
            "path",
            "skill_path",
            "directory",
            "dir",
            "location",
            "repo_path",
        )
        source_value = _first_str(data, "url", "skill_url", "source", "skill_source")
        source_url = source_value if _is_probable_url(source_value) else None

        parsed = _parse_github_url(repo_url) if repo_url else None
        if parsed and not repo_path:
            repo_url, repo_ref, repo_path = parsed
        elif parsed:
            repo_url = parsed[0]
            repo_ref = repo_ref or parsed[1]

        if source_url and (not repo_url or not repo_path):
            parsed_skill = _parse_github_url(source_url)
            if parsed_skill:
                repo_url, repo_ref, repo_path = parsed_skill
        elif source_value and not _is_probable_url(source_value) and not repo_path:
            repo_path = _normalize_source_path(source_value, data)

        name = _first_str(data, "name", "id", "slug", "title")
        description = _first_str(data, "description", "summary")
        bundle_name = _first_str(data, "bundle_name")
        bundle_description = _first_str(data, "bundle_description")
        if not name and repo_path:
            guessed = PurePosixPath(repo_path).parent.name
            name = guessed or repo_path

        repo_url = repo_url or default_repo_url
        repo_ref = repo_ref or default_repo_ref

        return {
            "name": name,
            "description": description,
            "repo_url": repo_url,
            "repo_ref": repo_ref,
            "repo_path": repo_path,
            "source_url": source_url,
            "bundle_name": bundle_name,
            "bundle_description": bundle_description,
        }

    @classmethod
    def from_entry(cls, entry: dict[str, Any], *, source_url: str | None = None) -> "MarketplaceEntryModel":
        model = cls.model_validate(entry)
        if source_url and not model.source_url:
            return model.model_copy(update={"source_url": source_url})
        return model


class MarketplacePayloadModel(BaseModel):
    entries: list[MarketplaceEntryModel] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any, info: Any) -> Any:
        return marketplace_source_utils.normalize_marketplace_payload(
            data,
            info,
            extract_entries=_extract_marketplace_entries,
        )


def get_manager_directory(
    settings: Settings | None = None, *, cwd: Path | None = None
) -> Path:
    """Resolve the local skills directory the manager operates on."""
    base = cwd or Path.cwd()
    resolved_settings = settings or get_settings()
    skills_settings = getattr(resolved_settings, "skills", None)

    directory = None
    if skills_settings and getattr(skills_settings, "directories", None):
        if skills_settings.directories:
            directory = skills_settings.directories[0]
    if not directory:
        directory = default_skill_paths(resolved_settings, cwd=base)[0]

    path = Path(directory).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def get_marketplace_url(settings: Settings | None = None) -> str:
    resolved_settings = settings or get_settings()
    skills_settings = getattr(resolved_settings, "skills", None)
    url = None
    if skills_settings is not None:
        # Check active registry first (set by /skills registry command)
        url = getattr(skills_settings, "marketplace_url", None)
        # Fall back to first configured registry
        if not url:
            urls = getattr(skills_settings, "marketplace_urls", None)
            if urls:
                url = urls[0]
    return _normalize_marketplace_url(url or DEFAULT_MARKETPLACE_URL)


def format_marketplace_display_url(url: str) -> str:
    return marketplace_registry_urls.format_marketplace_display_url(url)


def resolve_skill_registries(settings: Settings | None = None) -> list[str]:
    resolved_settings = settings or get_settings()
    skills_settings = getattr(resolved_settings, "skills", None)
    configured = getattr(skills_settings, "marketplace_urls", None) if skills_settings else None
    active = getattr(skills_settings, "marketplace_url", None) if skills_settings else None
    return marketplace_registry_urls.resolve_registry_urls(
        configured,
        default_urls=DEFAULT_SKILL_REGISTRIES,
        active_url=active,
    )


def resolve_skill_directories(
    settings: Settings | None = None, *, cwd: Path | None = None
) -> list[Path]:
    base = cwd or Path.cwd()
    resolved_settings = settings or get_settings()
    skills_settings = getattr(resolved_settings, "skills", None)
    override_dirs: list[Path] | None = None
    if skills_settings and getattr(skills_settings, "directories", None):
        # Resolve paths the same way get_manager_directory does
        resolved: list[Path] = []
        for entry in skills_settings.directories:
            path = Path(entry).expanduser()
            if not path.is_absolute():
                path = (base / path).resolve()
            resolved.append(path)
        override_dirs = resolved
    manager_dir = get_manager_directory(resolved_settings, cwd=cwd)
    if override_dirs is None:
        return default_skill_paths(resolved_settings, cwd=base)
    if manager_dir not in override_dirs:
        override_dirs.append(manager_dir)
    return override_dirs


def list_local_skills(directory: Path) -> list[SkillManifest]:
    return SkillRegistry.load_directory(directory)


def get_skill_source_sidecar_path(skill_dir: Path) -> Path:
    return skill_dir / SKILL_SOURCE_FILENAME


def compute_skill_content_fingerprint(skill_dir: Path) -> str:
    digest = hashlib.sha256()
    root = skill_dir.resolve()
    sidecar_path = get_skill_source_sidecar_path(root)

    for path in sorted(root.rglob("*")):
        if path == sidecar_path:
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return f"sha256:{digest.hexdigest()}"


def read_installed_skill_source(skill_dir: Path) -> tuple[InstalledSkillSource | None, str | None]:
    sidecar_path = get_skill_source_sidecar_path(skill_dir)
    if not sidecar_path.exists():
        return None, None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"invalid json: {exc}"

    if not isinstance(payload, dict):
        return None, "metadata root must be an object"

    try:
        source = _parse_installed_skill_source(payload)
    except ValueError as exc:
        return None, str(exc)
    return source, None


def write_installed_skill_source(skill_dir: Path, source: InstalledSkillSource) -> None:
    sidecar_path = get_skill_source_sidecar_path(skill_dir)
    payload = {
        "schema_version": source.schema_version,
        "installed_via": source.installed_via,
        "source_origin": source.source_origin,
        "repo_url": source.repo_url,
        "repo_ref": source.repo_ref,
        "repo_path": source.repo_path,
        "source_url": source.source_url,
        "installed_commit": source.installed_commit,
        "installed_path_oid": source.installed_path_oid,
        "installed_revision": source.installed_revision,
        "installed_at": source.installed_at,
        "content_fingerprint": source.content_fingerprint,
    }
    sidecar_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def get_skill_provenance(skill_dir: Path) -> SkillProvenance:
    source, error = read_installed_skill_source(skill_dir)
    if source is None:
        if error is None:
            return SkillProvenance(
                status="unmanaged",
                summary="unmanaged (no sidecar)",
            )
        return SkillProvenance(
            status="invalid_metadata",
            summary=f"invalid metadata ({error})",
            error=error,
        )

    ref_label = f"@{source.repo_ref}" if source.repo_ref else ""
    if source.source_origin == "remote":
        summary = (
            "managed (marketplace)"
            f" • {source.repo_url}{ref_label}"
            f" • {source.repo_path}"
        )
    else:
        summary = (
            "managed (local source)"
            f" • {source.repo_url}{ref_label}"
            f" • {source.repo_path}"
        )
    return SkillProvenance(status="managed", summary=summary, source=source)


def format_skill_provenance(skill_dir: Path) -> str:
    return get_skill_provenance(skill_dir).summary


def format_revision_short(revision: str | None) -> str:
    return marketplace_formatting.format_revision_short(revision)


def format_installed_at_display(installed_at: str | None) -> str:
    return marketplace_formatting.format_installed_at_display(installed_at)


def format_skill_provenance_details(skill_dir: Path) -> tuple[str, str | None]:
    provenance = get_skill_provenance(skill_dir)
    if provenance.status == "unmanaged":
        return "unmanaged.", None
    if provenance.status != "managed" or provenance.source is None:
        return provenance.summary, None

    source = provenance.source
    ref_label = f"@{source.repo_ref}" if source.repo_ref else ""
    provenance_value = f"{source.repo_url}{ref_label} ({source.repo_path})"
    installed_value = (
        f"{format_installed_at_display(source.installed_at)} "
        f"revision: {format_revision_short(source.installed_revision)}"
    )
    return provenance_value, installed_value


def order_skill_directories_for_display(
    directories: Sequence[Path],
    *,
    settings: Settings | None = None,
    cwd: Path | None = None,
) -> list[Path]:
    manager_dir = get_manager_directory(settings, cwd=cwd)
    ordered: list[Path] = []
    manager_entries: list[Path] = []
    for directory in directories:
        if directory == manager_dir:
            manager_entries.append(directory)
        else:
            ordered.append(directory)
    ordered.extend(manager_entries)
    return ordered


def check_skill_updates(*, destination_root: Path) -> list[SkillUpdateInfo]:
    return _check_skill_updates(destination_root=destination_root)


def select_skill_updates(
    updates: Sequence[SkillUpdateInfo], selector: str
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
    head_cache: dict[tuple[str, str | None], tuple[str | None, SkillUpdateStatus | None, str | None]] = {}
    path_cache: dict[
        tuple[str, str | None, str, str],
        tuple[str | None, SkillUpdateStatus | None, str | None],
    ] = {}
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


async def fetch_marketplace_skills(url: str) -> list[MarketplaceSkill]:
    skills, _ = await fetch_marketplace_skills_with_source(url)
    return skills


async def fetch_marketplace_skills_with_source(
    url: str,
) -> tuple[list[MarketplaceSkill], str]:
    return await marketplace_source_utils.fetch_marketplace_entries_with_source(
        url,
        candidate_urls=_candidate_marketplace_urls,
        normalize_url=_normalize_marketplace_url,
        load_local_payload=_load_local_marketplace_payload,
        parse_payload=lambda payload, source_url: _parse_marketplace_payload(
            payload,
            source_url=source_url,
        ),
    )


async def install_marketplace_skill(
    skill: MarketplaceSkill,
    *,
    destination_root: Path,
) -> Path:
    return await asyncio.to_thread(_install_marketplace_skill_sync, skill, destination_root)


def remove_local_skill(skill_dir: Path, *, destination_root: Path) -> None:
    skill_dir = skill_dir.resolve()
    destination_root = destination_root.resolve()
    if destination_root not in skill_dir.parents:
        raise ValueError("Skill path is outside of the managed skills directory.")
    if not skill_dir.exists():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")
    shutil.rmtree(skill_dir)


def select_skill_by_name_or_index(
    entries: Iterable[MarketplaceSkill], selector: str
) -> MarketplaceSkill | None:
    selector = selector.strip()
    if not selector:
        return None
    if selector.isdigit():
        index = int(selector)
        entries_list = list(entries)
        if 1 <= index <= len(entries_list):
            return entries_list[index - 1]
        return None
    selector_lower = selector.lower()
    for entry in entries:
        if entry.name.lower() == selector_lower:
            return entry
    return None


def select_manifest_by_name_or_index(
    manifests: Iterable[SkillManifest], selector: str
) -> SkillManifest | None:
    selector = selector.strip()
    if not selector:
        return None
    manifests_list = list(manifests)
    if selector.isdigit():
        index = int(selector)
        if 1 <= index <= len(manifests_list):
            return manifests_list[index - 1]
        return None
    selector_lower = selector.lower()
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


def _normalize_marketplace_url(url: str) -> str:
    return marketplace_source_utils.normalize_marketplace_url(url)


def _candidate_marketplace_urls(url: str) -> list[str]:
    return marketplace_source_utils.candidate_marketplace_urls(url)


def candidate_marketplace_urls(url: str) -> list[str]:
    return _candidate_marketplace_urls(url)


def _parse_marketplace_payload(
    payload: Any, *, source_url: str | None = None
) -> list[MarketplaceSkill]:
    repo_url = None
    repo_ref = None
    if source_url:
        parsed = marketplace_source_utils.parse_github_url(source_url)
        if parsed:
            repo_url, repo_ref, _ = parsed
        else:
            # Check if source_url is a local path and derive repo root
            local_repo = marketplace_source_utils.derive_local_repo_root(source_url)
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
        logger.warning(
            "Failed to parse marketplace payload",
            data={"error": str(exc)},
        )
        return []

    skills: list[MarketplaceSkill] = []
    for entry in model.entries:
        try:
            skill = _skill_from_entry_model(entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to parse marketplace entry",
                data={"error": str(exc), "entry": _safe_json(entry.model_dump())},
            )
            continue
        if skill:
            skills.append(skill)
    return skills


def _extract_marketplace_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("plugins"), list):
            plugin_root = None
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                plugin_root = metadata.get("pluginRoot") or metadata.get("plugin_root")
            entries: list[dict[str, Any]] = []
            for entry in payload.get("plugins", []):
                if isinstance(entry, dict):
                    entries.extend(_expand_plugin_entry(entry, plugin_root))
            return entries
        for key in ("skills", "items", "entries", "marketplace", "plugins"):
            value = payload.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            return [value for value in payload.values() if isinstance(value, dict)]
    raise ValueError("Unsupported marketplace payload format.")


def _skill_from_entry_model(model: MarketplaceEntryModel) -> MarketplaceSkill | None:
    if not model.repo_url or not model.repo_path:
        return None

    repo_path = _normalize_repo_path(model.repo_path)
    if not repo_path:
        return None

    return MarketplaceSkill(
        name=model.name or repo_path,
        description=model.description,
        repo_url=model.repo_url,
        repo_ref=model.repo_ref,
        repo_path=repo_path,
        source_url=model.source_url,
        bundle_name=model.bundle_name,
        bundle_description=model.bundle_description,
    )


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


def _expand_plugin_entry(entry: dict[str, Any], plugin_root: str | None) -> list[dict[str, Any]]:
    source = entry.get("source")
    repo_url, repo_ref, repo_path = _parse_plugin_source(source, plugin_root)
    skills = entry.get("skills")
    bundle_name = entry.get("name")
    bundle_description = entry.get("description")
    base_entry = dict(entry)
    base_entry.pop("skills", None)
    if repo_url and not base_entry.get("repo_url"):
        base_entry["repo_url"] = repo_url
    if repo_ref and not base_entry.get("repo_ref"):
        base_entry["repo_ref"] = repo_ref
    if repo_path and not base_entry.get("repo_path"):
        base_entry["repo_path"] = repo_path

    if isinstance(skills, list) and skills:
        expanded: list[dict[str, Any]] = []
        for skill in skills:
            if not isinstance(skill, str) or not skill.strip():
                continue
            skill_name = PurePosixPath(skill).name or skill.strip()
            combined_path = _join_relative_paths(repo_path, skill)
            skill_entry = dict(base_entry)
            skill_entry["name"] = skill_name
            skill_entry["description"] = None
            skill_entry["bundle_name"] = bundle_name
            skill_entry["bundle_description"] = bundle_description
            skill_entry["repo_path"] = combined_path
            expanded.append(skill_entry)
        if expanded:
            return expanded
    return [base_entry]


def _parse_plugin_source(
    source: Any, plugin_root: str | None
) -> tuple[str | None, str | None, str | None]:
    repo_url = None
    repo_ref = None
    repo_path = None
    plugin_root_applied = False

    if isinstance(source, str) and source.strip():
        if _is_probable_url(source):
            repo_url = source.strip()
        else:
            repo_path = _join_relative_paths(plugin_root, source)
            plugin_root_applied = True
    elif isinstance(source, dict):
        source_kind = source.get("source")
        if source_kind == "github":
            repo = _first_str(source, "repo")
            if repo:
                repo_url = f"https://github.com/{repo}"
            repo_ref = _first_str(source, "ref", "branch", "tag", "revision", "commit")
            repo_path = _first_str(source, "path", "directory", "dir", "location")
        elif source_kind in {"url", "git"}:
            repo_url = _first_str(source, "url", "repo", "repository")
            repo_ref = _first_str(source, "ref", "branch", "tag", "revision", "commit")
            repo_path = _first_str(source, "path", "directory", "dir", "location")
        else:
            repo_url = _first_str(source, "url", "repo", "repository")
            repo_ref = _first_str(source, "ref", "branch", "tag", "revision", "commit")
            repo_path = _first_str(source, "path", "directory", "dir", "location")

    if repo_path and plugin_root and not plugin_root_applied and not _is_probable_url(repo_path):
        repo_path = _join_relative_paths(plugin_root, repo_path)

    return repo_url, repo_ref, repo_path


def _join_relative_paths(base: str | None, leaf: str | None) -> str | None:
    base_clean = _clean_relative_path(base)
    leaf_clean = _clean_relative_path(leaf)
    if not base_clean:
        return leaf_clean
    if not leaf_clean:
        return base_clean
    return str(PurePosixPath(base_clean) / PurePosixPath(leaf_clean))


def _clean_relative_path(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = str(value).strip().replace("\\", "/")
    cleaned = cleaned.lstrip("./").strip("/")
    if cleaned in {"", "."}:
        return None
    return cleaned


def _parse_github_url(url: str | None) -> tuple[str, str | None, str] | None:
    return marketplace_source_utils.parse_github_url(url)


def _is_probable_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)


def _normalize_source_path(source: str, entry: dict[str, Any]) -> str | None:
    if not source:
        return None
    source_path = source.strip().lstrip("./")
    if not source_path:
        return None

    name = _first_str(entry, "name", "id", "slug", "title")
    if "/skills/" in source_path:
        return source_path
    if source_path.endswith("/skills"):
        if name:
            return f"{source_path}/{name}"
        return source_path
    if name:
        return f"{source_path}/skills/{name}"
    return source_path


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


def _install_marketplace_skill_sync(skill: MarketplaceSkill, destination_root: Path) -> Path:
    destination_root = destination_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    install_dir = destination_root / skill.install_dir_name
    if install_dir.exists():
        raise FileExistsError(f"Skill already exists: {install_dir}")

    local_repo = _resolve_local_repo(skill.repo_url)
    installed_commit: str | None = None
    source_origin: SkillSourceOrigin = "remote"

    if local_repo is not None:
        source_origin = "local"
        source_dir = _resolve_repo_subdir(local_repo, skill.repo_subdir)
        source_dir = _resolve_skill_source_dir(source_dir, skill.name)
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Skill path not found in repository: {skill.repo_subdir}"
            )
        _copy_skill_source(source_dir, install_dir)
        installed_commit = _resolve_git_commit(local_repo, skill.repo_ref)
        installed_path_oid = None
        if installed_commit is not None:
            installed_path_oid = _resolve_git_path_oid(local_repo, installed_commit, skill.repo_path)
        fingerprint = compute_skill_content_fingerprint(install_dir)
        write_installed_skill_source(
            install_dir,
            _build_installed_skill_source(
                skill=skill,
                source_origin=source_origin,
                installed_commit=installed_commit,
                installed_path_oid=installed_path_oid,
                fingerprint=fingerprint,
            ),
        )
        return install_dir

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
        _run_git(["git", "-C", str(tmp_path), "checkout"])

        installed_commit = _resolve_git_commit(tmp_path, "HEAD")
        installed_path_oid = None
        if installed_commit is not None:
            installed_path_oid = _resolve_git_path_oid(tmp_path, installed_commit, skill.repo_path)

        source_dir = _resolve_repo_subdir(tmp_path, skill.repo_subdir)
        source_dir = _resolve_skill_source_dir(source_dir, skill.name)
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Skill path not found in repository: {skill.repo_subdir}"
            )

        _copy_skill_source(source_dir, install_dir)

    fingerprint = compute_skill_content_fingerprint(install_dir)
    write_installed_skill_source(
        install_dir,
        _build_installed_skill_source(
            skill=skill,
            source_origin=source_origin,
            installed_commit=installed_commit,
            installed_path_oid=installed_path_oid,
            fingerprint=fingerprint,
        ),
    )

    return install_dir


def _check_skill_updates(*, destination_root: Path) -> list[SkillUpdateInfo]:
    destination_root = destination_root.resolve()
    if not destination_root.exists() or not destination_root.is_dir():
        return []

    manifests, parse_errors = SkillRegistry.load_directory_with_errors(destination_root)
    manifests_by_dir = {
        (manifest.path.parent if manifest.path.is_file() else manifest.path): manifest
        for manifest in manifests
    }
    head_cache: dict[tuple[str, str | None], tuple[str | None, SkillUpdateStatus | None, str | None]] = {}
    path_cache: dict[
        tuple[str, str | None, str, str],
        tuple[str | None, SkillUpdateStatus | None, str | None],
    ] = {}
    updates: list[SkillUpdateInfo] = []

    skill_dirs = [entry for entry in sorted(destination_root.iterdir()) if entry.is_dir()]
    index = 0
    for skill_dir in skill_dirs:
        index += 1
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
    head_cache: dict[tuple[str, str | None], tuple[str | None, SkillUpdateStatus | None, str | None]],
    path_cache: dict[
        tuple[str, str | None, str, str],
        tuple[str | None, SkillUpdateStatus | None, str | None],
    ],
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

    available_revision, resolve_status, resolve_error = _resolve_source_revision(source, head_cache)
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


def _resolve_source_revision(
    source: InstalledSkillSource,
    head_cache: dict[tuple[str, str | None], tuple[str | None, SkillUpdateStatus | None, str | None]],
) -> tuple[str | None, SkillUpdateStatus | None, str | None]:
    cache_key = (source.repo_url, source.repo_ref)
    cached = head_cache.get(cache_key)
    if cached is not None:
        return cached

    local_repo = _resolve_local_repo(source.repo_url)
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
    if source.repo_ref:
        ls_remote_args.append(source.repo_ref)
    else:
        ls_remote_args.append("HEAD")

    result = subprocess.run(ls_remote_args, capture_output=True, text=True, check=False)
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

    commit = _parse_ls_remote_commit(output)
    if commit is None:
        resolved = (None, "source_unreachable", "unable to resolve source revision")
        head_cache[cache_key] = resolved
        return resolved

    resolved = (commit, None, None)
    head_cache[cache_key] = resolved
    return resolved


def _parse_ls_remote_commit(output: str) -> str | None:
    """Extract a commit hash from `git ls-remote` output.

    For annotated tags, prefer the peeled commit (`refs/tags/<tag>^{}`) when present.
    """
    return marketplace_source_utils.parse_ls_remote_commit(output)


def _resolve_source_path_oid(
    source: InstalledSkillSource,
    commit: str,
    path_cache: dict[
        tuple[str, str | None, str, str],
        tuple[str | None, SkillUpdateStatus | None, str | None],
    ],
) -> tuple[str | None, SkillUpdateStatus | None, str | None]:
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


def _parse_installed_skill_source(payload: dict[str, Any]) -> InstalledSkillSource:
    parsed = marketplace_source_utils.parse_installed_source_fields(
        payload,
        expected_schema_version=SKILL_SOURCE_SCHEMA_VERSION,
        normalize_repo_path=_normalize_repo_path,
    )

    return InstalledSkillSource(
        schema_version=SKILL_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin=parsed.source_origin,
        repo_url=parsed.repo_url,
        repo_ref=parsed.repo_ref,
        repo_path=parsed.repo_path,
        source_url=parsed.source_url,
        installed_commit=parsed.installed_commit,
        installed_path_oid=parsed.installed_path_oid,
        installed_revision=parsed.installed_revision,
        installed_at=parsed.installed_at,
        content_fingerprint=parsed.content_fingerprint,
    )


def _build_installed_skill_source(
    *,
    skill: MarketplaceSkill,
    source_origin: SkillSourceOrigin,
    installed_commit: str | None,
    installed_path_oid: str | None,
    fingerprint: str,
) -> InstalledSkillSource:
    installed_revision = installed_commit or LOCAL_REVISION
    return InstalledSkillSource(
        schema_version=SKILL_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin=source_origin,
        repo_url=skill.repo_url,
        repo_ref=skill.repo_ref,
        repo_path=skill.repo_path,
        source_url=skill.source_url,
        installed_commit=installed_commit,
        installed_path_oid=installed_path_oid,
        installed_revision=installed_revision,
        installed_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        content_fingerprint=fingerprint,
    )


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
        installed_commit, installed_path_oid = _copy_skill_from_marketplace_source(
            source_skill,
            destination_dir=staged_dir,
            pinned_revision=revision,
        )
        fingerprint = compute_skill_content_fingerprint(staged_dir)
        staged_source = InstalledSkillSource(
            schema_version=SKILL_SOURCE_SCHEMA_VERSION,
            installed_via="marketplace",
            source_origin=source.source_origin,
            repo_url=source.repo_url,
            repo_ref=source.repo_ref,
            repo_path=source.repo_path,
            source_url=source.source_url,
            installed_commit=installed_commit,
            installed_path_oid=installed_path_oid,
            installed_revision=installed_commit or LOCAL_REVISION,
            installed_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace(
                "+00:00", "Z"
            ),
            content_fingerprint=fingerprint,
        )
        write_installed_skill_source(staged_dir, staged_source)
        _atomic_replace_directory(existing_dir=skill_dir, staged_dir=staged_dir)
        return staged_source


def _copy_skill_from_marketplace_source(
    skill: MarketplaceSkill,
    *,
    destination_dir: Path,
    pinned_revision: str | None,
) -> tuple[str | None, str | None]:
    local_repo = _resolve_local_repo(skill.repo_url)
    if local_repo is not None:
        source_dir = _resolve_repo_subdir(local_repo, skill.repo_subdir)
        source_dir = _resolve_skill_source_dir(source_dir, skill.name)
        if not source_dir.exists():
            raise FileNotFoundError(f"Skill path not found in repository: {skill.repo_subdir}")
        _copy_skill_source(source_dir, destination_dir)
        commit = None
        if skill.repo_ref:
            commit = _resolve_git_commit(local_repo, skill.repo_ref)
        else:
            commit = _resolve_git_commit(local_repo, "HEAD")
        path_oid = None
        if commit is not None:
            path_oid = _resolve_git_path_oid(local_repo, commit, skill.repo_path)
        return commit, path_oid

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
        return commit, path_oid


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


def _read_json_file(path: Path) -> Any:
    return marketplace_source_utils.read_json_file(path)


def _resolve_local_repo(repo_url: str) -> Path | None:
    return marketplace_source_utils.resolve_local_repo(repo_url)


def _derive_local_repo_root(source_url: str) -> str | None:
    """Derive the local repo root from a marketplace source path/URL."""
    return marketplace_source_utils.derive_local_repo_root(source_url)


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
