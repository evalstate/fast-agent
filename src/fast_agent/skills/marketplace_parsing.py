"""Skills marketplace payload parsing helpers."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fast_agent.core.logging.logger import get_logger
from fast_agent.marketplace import source_utils as marketplace_source_utils
from fast_agent.skills.models import MarketplaceSkill

logger = get_logger(__name__)


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


def parse_marketplace_payload(
    payload: Any,
    *,
    source_url: str | None = None,
) -> list[MarketplaceSkill]:
    repo_url = None
    repo_ref = None
    if source_url:
        parsed = marketplace_source_utils.parse_github_url(source_url)
        if parsed:
            repo_url, repo_ref, _ = parsed
        else:
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


def normalize_repo_path(path: str) -> str | None:
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

    repo_path = normalize_repo_path(model.repo_path)
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
    source: Any,
    plugin_root: str | None,
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
