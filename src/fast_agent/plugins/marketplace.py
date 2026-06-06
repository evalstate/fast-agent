"""Plugin marketplace payload parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fast_agent.marketplace import fetch as marketplace_fetch
from fast_agent.marketplace import provenance_io as marketplace_provenance_io
from fast_agent.marketplace import source_urls as marketplace_source_urls
from fast_agent.marketplace.models import MarketplaceEntryFieldsModel
from fast_agent.plugins.models import PLUGIN_MANIFEST_FILENAME, MarketplacePlugin
from fast_agent.plugins.provenance import normalize_repo_path
from fast_agent.utils.action_normalization import normalize_action_token

if TYPE_CHECKING:
    from pydantic import ValidationInfo

_CARD_PACK_ENTRY_KINDS = frozenset(("card", "card_pack", "card-pack", "bundle"))


def _is_card_pack_marketplace_entry(kind: str | None) -> bool:
    return normalize_action_token(kind) in _CARD_PACK_ENTRY_KINDS


class MarketplacePluginEntryModel(MarketplaceEntryFieldsModel):
    @model_validator(mode="before")
    @classmethod
    def _normalize_entry(cls, data: Any, info: "ValidationInfo") -> Any:
        if not isinstance(data, dict):
            return data
        context = info.context or {}
        default_repo_url = context.get("repo_url")
        default_source_url = context.get("source_url")
        repo_fields = marketplace_source_urls.marketplace_repo_fields(
            data,
            repo_path_keys=("path", "plugin_path", "directory", "dir", "repo_path"),
        )
        repo_url = repo_fields.repo_url
        repo_ref = repo_fields.repo_ref
        repo_path = repo_fields.repo_path
        source_url_value = marketplace_source_urls.explicit_entry_source_url(
            data,
            "source_url",
            "url",
            default_source_url=default_source_url,
        )
        source_value = marketplace_source_urls.first_nonempty_str(data, "source")
        source_url_value_is_url = marketplace_source_urls.is_git_source_url(source_url_value)
        source_value_is_url = marketplace_source_urls.is_git_source_url(source_value)
        source_url = (
            source_value
            if source_value_is_url
            else source_url_value
            if source_url_value_is_url
            else None
        )
        if (
            source_url_value
            and repo_path
            and not source_url
            and not repo_url
        ):
            repo_url = source_url_value
            source_url = source_url_value
        source_path = (
            source_value
            if not source_value_is_url
            else None
        )

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
        if source_path and not repo_path:
            repo_path = source_path

        repo_url = repo_url or default_repo_url
        repo_ref = repo_ref or context.get("repo_ref")
        name = marketplace_source_urls.first_nonempty_str(data, "name", "id", "slug", "title")
        if not name and repo_path:
            name = _plugin_name_from_repo_path(repo_path)

        return {
            "name": name,
            "description": marketplace_source_urls.first_nonempty_str(
                data,
                "description",
                "summary",
            ),
            "kind": marketplace_source_urls.first_nonempty_str(data, "kind", "type"),
            "repo_url": repo_url,
            "repo_ref": repo_ref,
            "repo_path": repo_path,
            "source_url": source_url or context.get("source_url"),
            "bundle_name": marketplace_source_urls.first_nonempty_str(data, "bundle_name"),
        }


class MarketplacePluginPayloadModel(BaseModel):
    entries: list[MarketplacePluginEntryModel] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any, info: "ValidationInfo") -> Any:
        return marketplace_fetch.normalize_marketplace_payload(
            data,
            info,
            extract_entries=_extract_marketplace_entries,
        )


def parse_marketplace_plugins(
    payload: Any,
    *,
    source_url: str | None = None,
) -> list[MarketplacePlugin]:
    source_context = marketplace_fetch.marketplace_source_context(source_url)

    model = MarketplacePluginPayloadModel.model_validate(
        payload,
        context=source_context.as_validation_context(),
    )
    plugins: list[MarketplacePlugin] = []
    for entry in model.entries:
        if _is_card_pack_marketplace_entry(entry.kind):
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
    try:
        return marketplace_fetch.extract_dict_entries(
            payload,
            ("command_plugins", "fast_agent_plugins", "plugins", "entries"),
        )
    except ValueError as exc:
        raise ValueError("Unsupported plugin marketplace payload format.") from exc


def _plugin_name_from_repo_path(repo_path: str) -> str:
    return marketplace_provenance_io.repo_name_for_manifest_path(
        repo_path,
        PLUGIN_MANIFEST_FILENAME,
    )
