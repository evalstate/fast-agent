from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from fast_agent.config import OpenAIWebSearchSettings


@dataclass(frozen=True)
class ResolvedOpenAIWebSearch:
    enabled: bool
    settings: "OpenAIWebSearchSettings"


def resolve_web_search(
    openai_settings: object | None,
    *,
    web_search_override: bool | None,
) -> ResolvedOpenAIWebSearch:
    from fast_agent.config import OpenAIWebSearchSettings

    raw_settings = getattr(openai_settings, "web_search", None) if openai_settings else None
    settings = (
        raw_settings
        if isinstance(raw_settings, OpenAIWebSearchSettings)
        else OpenAIWebSearchSettings()
    )

    enabled = web_search_override
    if enabled is None:
        enabled = bool(settings.enabled)

    return ResolvedOpenAIWebSearch(enabled=enabled, settings=settings)


def build_web_search_tool(
    resolved: ResolvedOpenAIWebSearch,
) -> dict[str, Any] | None:
    if not resolved.enabled:
        return None

    settings = resolved.settings

    payload: dict[str, Any] = {
        "type": settings.tool_type,
    }

    if settings.search_context_size is not None:
        payload["search_context_size"] = settings.search_context_size

    if settings.allowed_domains:
        payload["filters"] = {
            "allowed_domains": list(settings.allowed_domains),
        }

    if settings.user_location is not None:
        payload["user_location"] = settings.user_location.model_dump(
            exclude_none=True,
        )

    # Only GA web_search currently honors external_web_access.
    if settings.tool_type == "web_search" and settings.external_web_access is not None:
        payload["external_web_access"] = settings.external_web_access

    return payload


def _as_payload(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if hasattr(value, "model_dump"):
        model_dump = getattr(value, "model_dump")
        try:
            payload = model_dump(mode="json", exclude_none=False)
        except TypeError:
            payload = model_dump()
        if isinstance(payload, Mapping):
            return {str(key): item for key, item in payload.items()}
    payload: dict[str, Any] = {}
    for field_name in ("type", "id", "name", "status", "action", "query", "queries", "sources"):
        field_value = getattr(value, field_name, None)
        if field_value is not None:
            payload[field_name] = field_value
    return payload


def _mapping_view(value: object | None) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return cast("Mapping[str, Any]", value)
    return None


def _action_type(action: object | None) -> str | None:
    if action is None:
        return None
    action_map = _mapping_view(action)
    if action_map is not None:
        action_type = action_map.get("type")
        return action_type if isinstance(action_type, str) else None
    action_type = getattr(action, "type", None)
    return action_type if isinstance(action_type, str) else None


def _action_queries(action: object | None) -> list[str] | None:
    if action is None:
        return None

    raw_queries: object | None
    action_map = _mapping_view(action)
    if action_map is not None:
        raw_queries = action_map.get("queries")
        if raw_queries is None:
            raw_query = action_map.get("query")
            raw_queries = [raw_query] if isinstance(raw_query, str) and raw_query else None
    else:
        raw_queries = getattr(action, "queries", None)
        if raw_queries is None:
            raw_query = getattr(action, "query", None)
            raw_queries = [raw_query] if isinstance(raw_query, str) and raw_query else None

    if raw_queries is None:
        return None
    if not isinstance(raw_queries, Sequence) or isinstance(raw_queries, str):
        return None

    queries: list[str] = []
    for query in raw_queries:
        if isinstance(query, str) and query:
            queries.append(query)
    return queries or None


def _extract_source_payload(source: object) -> dict[str, str] | None:
    source_map = _mapping_view(source)
    if source_map is not None:
        source_type = source_map.get("type")
        source_url = source_map.get("url")
    else:
        source_type = getattr(source, "type", None)
        source_url = getattr(source, "url", None)

    if isinstance(source_url, str) and source_url:
        payload: dict[str, str] = {
            "type": "web_search_result_location",
            "url": source_url,
        }
        if isinstance(source_type, str) and source_type and source_type != "url":
            payload["source"] = source_type
        return payload

    if isinstance(source_type, str) and source_type:
        return {
            "type": "web_search_result_location",
            "source": source_type,
        }

    return None


def normalize_web_search_call_payload(item: object) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    payload = _as_payload(item)
    item_type = payload.get("type")
    if item_type != "web_search_call":
        return None, []

    normalized: dict[str, Any] = {
        "type": "server_tool_use",
        "name": "web_search",
    }

    item_id = payload.get("id")
    if isinstance(item_id, str) and item_id:
        normalized["id"] = item_id

    status = payload.get("status")
    if isinstance(status, str) and status:
        normalized["status"] = status

    action = payload.get("action")
    action_type = _action_type(cast("object | None", action))
    if action_type:
        normalized["action"] = action_type

    queries = _action_queries(cast("object | None", action))
    if queries:
        normalized["queries"] = queries

    citation_payloads: list[dict[str, str]] = []
    raw_sources: object | None
    action_map = _mapping_view(cast("object | None", action))
    if action_map is not None:
        raw_sources = action_map.get("sources")
    else:
        raw_sources = getattr(action, "sources", None)

    if isinstance(raw_sources, Sequence) and not isinstance(raw_sources, str):
        for source in raw_sources:
            source_payload = _extract_source_payload(source)
            if source_payload is not None:
                citation_payloads.append(source_payload)

    return normalized, citation_payloads


def extract_url_citation_payload(annotation: object) -> dict[str, Any] | None:
    annotation_map = _mapping_view(annotation)
    if annotation_map is not None:
        annotation_type = annotation_map.get("type")
        title = annotation_map.get("title")
        url = annotation_map.get("url")
        start_index = annotation_map.get("start_index")
        end_index = annotation_map.get("end_index")
    else:
        annotation_type = getattr(annotation, "type", None)
        title = getattr(annotation, "title", None)
        url = getattr(annotation, "url", None)
        start_index = getattr(annotation, "start_index", None)
        end_index = getattr(annotation, "end_index", None)

    if annotation_type != "url_citation":
        return None

    payload: dict[str, Any] = {
        "type": "web_search_result_location",
    }
    if isinstance(title, str) and title:
        payload["title"] = title
    if isinstance(url, str) and url:
        payload["url"] = url
    if isinstance(start_index, int):
        payload["start_index"] = start_index
    if isinstance(end_index, int):
        payload["end_index"] = end_index

    return payload if len(payload) > 1 else None
