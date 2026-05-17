"""Plugin marketplace payload parsing."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fast_agent.marketplace import source_utils as marketplace_source_utils
from fast_agent.plugins.models import MarketplacePlugin
from fast_agent.plugins.provenance import normalize_repo_path


class MarketplacePluginEntryModel(BaseModel):
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
        repo_url = _first_str(data, "repo", "repository", "git", "repo_url")
        repo_ref = _first_str(data, "repo_ref", "ref", "branch", "tag", "revision", "commit")
        repo_path = _first_str(data, "path", "plugin_path", "directory", "dir", "repo_path")
        source_url = _first_str(data, "source_url", "url", "source")

        parsed = marketplace_source_utils.parse_github_url(repo_url) if repo_url else None
        if parsed and not repo_path:
            repo_url, repo_ref, repo_path = parsed
        elif parsed:
            repo_url = parsed[0]
            repo_ref = repo_ref or parsed[1]

        if source_url and (not repo_url or not repo_path):
            parsed_source = marketplace_source_utils.parse_github_url(source_url)
            if parsed_source:
                repo_url, repo_ref, repo_path = parsed_source

        repo_url = repo_url or context.get("repo_url")
        repo_ref = repo_ref or context.get("repo_ref")
        name = _first_str(data, "name", "id", "slug", "title")
        if not name and repo_path:
            name = PurePosixPath(repo_path).name or repo_path

        return {
            "name": name,
            "description": _first_str(data, "description", "summary"),
            "kind": _first_str(data, "kind", "type"),
            "repo_url": repo_url,
            "repo_ref": repo_ref,
            "repo_path": repo_path,
            "source_url": source_url or context.get("source_url"),
            "bundle_name": _first_str(data, "bundle_name"),
        }


class MarketplacePluginPayloadModel(BaseModel):
    entries: list[MarketplacePluginEntryModel] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any, info: Any) -> Any:
        return marketplace_source_utils.normalize_marketplace_payload(
            data,
            info,
            extract_entries=_extract_marketplace_entries,
        )


def parse_marketplace_plugins(
    payload: Any,
    *,
    source_url: str | None = None,
) -> list[MarketplacePlugin]:
    repo_url = None
    repo_ref = None
    if source_url:
        parsed = marketplace_source_utils.parse_github_url(source_url)
        if parsed:
            repo_url, repo_ref, _ = parsed
        else:
            repo_url = marketplace_source_utils.derive_local_repo_root(source_url)

    model = MarketplacePluginPayloadModel.model_validate(
        payload,
        context={"source_url": source_url, "repo_url": repo_url, "repo_ref": repo_ref},
    )
    plugins: list[MarketplacePlugin] = []
    for entry in model.entries:
        if (entry.kind or "").strip().lower() in {"card", "card_pack", "card-pack", "bundle"}:
            continue
        if not entry.repo_url or not entry.repo_path:
            continue
        repo_path = normalize_repo_path(entry.repo_path)
        if not repo_path:
            continue
        plugins.append(
            MarketplacePlugin(
                name=entry.name or repo_path,
                description=entry.description,
                repo_url=entry.repo_url,
                repo_ref=entry.repo_ref,
                repo_path=repo_path,
                source_url=entry.source_url,
                bundle_name=entry.bundle_name,
            )
        )
    return plugins


def _extract_marketplace_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if isinstance(payload, dict):
        for key in ("command_plugins", "fast_agent_plugins", "plugins"):
            value = payload.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
        if isinstance(payload.get("entries"), list):
            return [entry for entry in payload["entries"] if isinstance(entry, dict)]
    raise ValueError("Unsupported plugin marketplace payload format.")


def _first_str(data: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
