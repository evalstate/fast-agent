"""Skills marketplace payload parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from fast_agent.core.logging.logger import get_logger
from fast_agent.marketplace import fetch as marketplace_fetch
from fast_agent.marketplace import provenance_io as marketplace_provenance_io
from fast_agent.marketplace import source_urls as marketplace_source_urls
from fast_agent.marketplace.models import MarketplaceEntryFieldsModel
from fast_agent.skills.models import SKILL_MANIFEST_FILENAME_LOWER, MarketplaceSkill
from fast_agent.utils.text import strip_str_to_none, strip_to_none

if TYPE_CHECKING:
    from pydantic import ValidationInfo

logger = get_logger(__name__)
_ENTRY_REPO_PATH_KEYS = ("path", "skill_path", "directory", "dir", "location", "repo_path")
_PLUGIN_SOURCE_PATH_KEYS = ("path", "directory", "dir", "location")
_EXPLICIT_MARKETPLACE_ENTRY_KEYS = ("skills", "items", "entries", "marketplace")


@dataclass(frozen=True, slots=True)
class _ParsedPluginSource:
    repo_url: str | None = None
    repo_ref: str | None = None
    repo_path: str | None = None
    invalid_path: bool = False


class MarketplaceEntryModel(MarketplaceEntryFieldsModel):
    bundle_description: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_entry(cls, data: Any, info: "ValidationInfo") -> Any:
        if not isinstance(data, dict):
            return data

        context = info.context or {}
        default_repo_url = context.get("repo_url")
        default_repo_ref = context.get("repo_ref")
        default_source_url = context.get("source_url")

        repo_fields = marketplace_source_urls.marketplace_repo_fields(
            data,
            repo_path_keys=_ENTRY_REPO_PATH_KEYS,
        )
        repo_url = repo_fields.repo_url
        repo_ref = repo_fields.repo_ref
        repo_path = repo_fields.repo_path
        source_value = marketplace_source_urls.first_nonempty_str(
            data,
            "url",
            "skill_url",
            "source",
            "skill_source",
        )
        source_value_is_url = marketplace_source_urls.is_git_source_url(source_value)
        source_url = source_value if source_value_is_url else None
        source_url_value = marketplace_source_urls.explicit_entry_source_url(
            data,
            "source_url",
            default_source_url=default_source_url,
        )
        if source_url is None and source_url_value:
            source_url = source_url_value

        resolved_fields = marketplace_source_urls.repo_fields_from_source_url(
            repo_url=repo_url,
            repo_ref=repo_ref,
            repo_path=repo_path,
            source_url=source_url,
            default_repo_url=default_repo_url,
        )
        repo_url = resolved_fields.repo_url
        repo_ref = resolved_fields.repo_ref
        repo_path = resolved_fields.repo_path
        if source_value and not source_value_is_url and not repo_path:
            repo_path = _normalize_source_path(source_value, data)

        name = marketplace_source_urls.first_nonempty_str(data, "name", "id", "slug", "title")
        description = marketplace_source_urls.first_nonempty_str(data, "description", "summary")
        bundle_name = marketplace_source_urls.first_nonempty_str(data, "bundle_name")
        bundle_description = marketplace_source_urls.first_nonempty_str(
            data,
            "bundle_description",
        )
        if not name and repo_path:
            name = _skill_name_from_plugin_skill_path(repo_path)

        repo_url = repo_url or default_repo_url
        repo_ref = repo_ref or default_repo_ref

        return {
            "name": name,
            "description": description,
            "repo_url": repo_url,
            "repo_ref": repo_ref,
            "repo_path": repo_path,
            "source_url": source_url or default_source_url,
            "bundle_name": bundle_name,
            "bundle_description": bundle_description,
        }


class MarketplacePayloadModel(BaseModel):
    entries: list[MarketplaceEntryModel] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any, info: "ValidationInfo") -> Any:
        return marketplace_fetch.normalize_marketplace_payload(
            data,
            info,
            extract_entries=_extract_marketplace_entries,
        )


def parse_marketplace_payload(
    payload: Any,
    *,
    source_url: str | None = None,
) -> list[MarketplaceSkill]:
    source_context = marketplace_fetch.marketplace_source_context(source_url)
    try:
        model = MarketplacePayloadModel.model_validate(
            payload,
            context=source_context.as_validation_context(),
        )
    except ValidationError as exc:
        logger.warning(
            "Failed to parse marketplace payload",
            data={"error": str(exc)},
        )
        return []

    skills: list[MarketplaceSkill] = []
    for entry in model.entries:
        skill = _skill_from_entry_model(entry)
        if skill:
            skills.append(skill)
    return skills


def normalize_repo_path(path: str) -> str | None:
    return marketplace_provenance_io.normalize_relative_repo_path(path, allow_current_dir=True)


def _extract_marketplace_entries(payload: Any) -> list[dict[str, Any]]:
    plugins = payload.get("plugins") if isinstance(payload, dict) else None
    if isinstance(payload, dict) and isinstance(plugins, list):
        entries = _extract_optional_dict_entries(payload, _EXPLICIT_MARKETPLACE_ENTRY_KEYS)
        plugin_root = None
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            plugin_root = metadata.get("pluginRoot") or metadata.get("plugin_root")
        for entry in plugins:
            if isinstance(entry, dict):
                entries.extend(_expand_plugin_entry(entry, plugin_root))
        return entries

    return marketplace_fetch.extract_dict_entries(
        payload,
        ("skills", "items", "entries", "marketplace", "plugins"),
        allow_mapping_values=True,
    )


def _extract_optional_dict_entries(payload: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        return marketplace_fetch.extract_dict_entries(
            payload,
            keys,
            allow_mapping_values=True,
        )
    except ValueError:
        return []


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
    if _is_invalid_relative_path(plugin_root):
        return []
    source = entry.get("source")
    parsed_source = _parse_plugin_source(source, plugin_root)
    if parsed_source.invalid_path:
        return []
    skills = entry.get("skills")
    bundle_name = entry.get("name")
    bundle_description = entry.get("description")
    base_entry = dict(entry)
    base_entry.pop("skills", None)
    if parsed_source.repo_url and not base_entry.get("repo_url"):
        base_entry["repo_url"] = parsed_source.repo_url
    if parsed_source.repo_ref and not base_entry.get("repo_ref"):
        base_entry["repo_ref"] = parsed_source.repo_ref
    if parsed_source.repo_path and not base_entry.get("repo_path"):
        base_entry["repo_path"] = parsed_source.repo_path

    if isinstance(skills, list) and skills:
        expanded: list[dict[str, Any]] = []
        for skill in skills:
            normalized_skill = strip_str_to_none(skill)
            if normalized_skill is None:
                continue
            if _is_invalid_relative_path(normalized_skill):
                continue
            combined_path = _join_relative_paths(parsed_source.repo_path, normalized_skill)
            skill_name = _skill_name_from_plugin_skill_path(
                normalized_skill,
                fallback_path=combined_path,
            )
            skill_entry = dict(base_entry)
            skill_entry["name"] = skill_name
            skill_entry["description"] = None
            skill_entry["bundle_name"] = bundle_name
            skill_entry["bundle_description"] = bundle_description
            skill_entry["repo_path"] = combined_path
            expanded.append(skill_entry)
        if expanded:
            return expanded
        return []
    return [base_entry]


def _skill_name_from_plugin_skill_path(
    skill_path: str,
    *,
    fallback_path: str | None = None,
) -> str:
    normalized_path = (
        _clean_relative_path(skill_path)
        or _clean_relative_path(fallback_path)
        or strip_to_none(skill_path)
        or skill_path
    )
    return marketplace_provenance_io.repo_name_for_manifest_path(
        normalized_path,
        SKILL_MANIFEST_FILENAME_LOWER,
        allow_current_dir=True,
    )


def _parse_plugin_source(
    source: Any,
    plugin_root: str | None,
) -> _ParsedPluginSource:
    repo_url = None
    repo_ref = None
    repo_path = None
    plugin_root_applied = False

    source_text = strip_str_to_none(source)
    if source_text is not None:
        parsed_github_source = marketplace_source_urls.parse_github_url(source_text)
        if parsed_github_source is not None:
            repo_url = parsed_github_source.repo_url
            repo_ref = parsed_github_source.repo_ref
            repo_path = parsed_github_source.repo_path
        elif marketplace_source_urls.is_git_source_url(source_text):
            repo_url = source_text
        else:
            if _is_invalid_relative_path(source_text):
                return _ParsedPluginSource(invalid_path=True)
            repo_path = _join_relative_paths(plugin_root, source_text)
            plugin_root_applied = True
    elif isinstance(source, dict):
        source_kind = source.get("source")
        repo_ref, repo_path = _parse_source_ref_and_path(source)
        if _is_invalid_relative_path(repo_path):
            return _ParsedPluginSource(invalid_path=True)
        if source_kind == "github":
            repo = marketplace_source_urls.first_nonempty_str(source, "repo")
            if repo:
                repo_url = _github_repo_url(repo)
        else:
            repo_url = marketplace_source_urls.first_nonempty_str(
                source,
                "url",
                "repo",
                "repository",
            )

    if (
        repo_path
        and plugin_root
        and not plugin_root_applied
        and not marketplace_source_urls.is_git_source_url(repo_path)
    ):
        repo_path = _join_relative_paths(plugin_root, repo_path)

    return _ParsedPluginSource(repo_url=repo_url, repo_ref=repo_ref, repo_path=repo_path)


def _github_repo_url(repo: str) -> str:
    normalized_repo = strip_to_none(repo) or repo
    if marketplace_source_urls.is_git_source_url(normalized_repo):
        return normalized_repo
    return f"https://github.com/{normalized_repo.strip('/')}"


def _parse_source_ref_and_path(source: dict[str, Any]) -> tuple[str | None, str | None]:
    return (
        marketplace_source_urls.first_nonempty_str(
            source,
            *marketplace_source_urls.MARKETPLACE_REPO_REF_KEYS,
        ),
        marketplace_source_urls.first_nonempty_str(source, *_PLUGIN_SOURCE_PATH_KEYS),
    )


def _join_relative_paths(base: str | None, leaf: str | None) -> str | None:
    base_clean = _clean_relative_path(base)
    leaf_clean = _clean_relative_path(leaf)
    if not base_clean:
        return leaf_clean
    if not leaf_clean:
        return base_clean
    return str(PurePosixPath(base_clean) / PurePosixPath(leaf_clean))


def _clean_relative_path(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = marketplace_provenance_io.normalize_relative_repo_path(
        value,
        allow_current_dir=True,
    )
    if cleaned == ".":
        return None
    return cleaned


def _is_invalid_relative_path(value: str | None) -> bool:
    if not value:
        return False
    return marketplace_provenance_io.normalize_relative_repo_path(
        value,
        allow_current_dir=True,
    ) is None


def _skill_scoped_source_path(source_path: str, name: str | None) -> str:
    if not name:
        return source_path
    path = PurePosixPath(source_path)
    if "skills" in path.parts:
        return source_path
    if path.name == "skills":
        return f"{source_path}/{name}"
    return f"{source_path}/skills/{name}"


def _normalize_source_path(source: str, entry: dict[str, Any]) -> str | None:
    if not source:
        return None
    source_path = _clean_relative_path(source)
    if not source_path:
        return None

    name = marketplace_source_urls.first_nonempty_str(entry, "name", "id", "slug", "title")
    return _skill_scoped_source_path(source_path, name)
