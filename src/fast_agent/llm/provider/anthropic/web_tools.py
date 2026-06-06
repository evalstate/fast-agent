from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from fast_agent.config import (
    AnthropicSettings,
    AnthropicWebFetchSettings,
    AnthropicWebSearchSettings,
)
from fast_agent.utils.collections import unique_preserve_order

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anthropic.types.beta import BetaToolParam


@runtime_checkable
class _SupportsModelDump(Protocol):
    def model_dump(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ResolvedAnthropicWebTools:
    search_enabled: bool
    fetch_enabled: bool
    search_settings: AnthropicWebSearchSettings
    fetch_settings: AnthropicWebFetchSettings


def resolve_web_tools(
    anthropic_settings: AnthropicSettings | None,
    *,
    web_search_override: bool | None,
    web_fetch_override: bool | None,
) -> ResolvedAnthropicWebTools:
    search_settings = (
        anthropic_settings.web_search if anthropic_settings else AnthropicWebSearchSettings()
    )
    fetch_settings = (
        anthropic_settings.web_fetch if anthropic_settings else AnthropicWebFetchSettings()
    )

    search_enabled = web_search_override
    if search_enabled is None:
        search_enabled = bool(search_settings.enabled)

    fetch_enabled = web_fetch_override
    if fetch_enabled is None:
        fetch_enabled = bool(fetch_settings.enabled)

    return ResolvedAnthropicWebTools(
        search_enabled=search_enabled,
        fetch_enabled=fetch_enabled,
        search_settings=search_settings,
        fetch_settings=fetch_settings,
    )


def _validate_domain_xor(
    *,
    allowed_domains: Sequence[str] | None,
    blocked_domains: Sequence[str] | None,
    tool_name: str,
) -> None:
    if allowed_domains and blocked_domains:
        raise ValueError(
            f"{tool_name}: allowed_domains and blocked_domains are mutually exclusive."
        )


def _validate_positive(value: int | None, field_name: str, tool_name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{tool_name}: {field_name} must be greater than zero when provided.")


def _base_web_tool_payload(
    *,
    tool_name: str,
    tool_type: str,
    max_uses: int | None,
    allowed_domains: Sequence[str] | None,
    blocked_domains: Sequence[str] | None,
) -> dict[str, Any]:
    _validate_domain_xor(
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        tool_name=tool_name,
    )
    _validate_positive(max_uses, "max_uses", tool_name)
    payload: dict[str, Any] = {
        "name": tool_name,
        "type": tool_type,
    }
    if max_uses is not None:
        payload["max_uses"] = max_uses
    if allowed_domains:
        payload["allowed_domains"] = list(allowed_domains)
    if blocked_domains:
        payload["blocked_domains"] = list(blocked_domains)
    return payload


def _web_search_payload(
    settings: AnthropicWebSearchSettings,
    version: str,
) -> dict[str, Any]:
    payload = _base_web_tool_payload(
        tool_name="web_search",
        tool_type=version,
        max_uses=settings.max_uses,
        allowed_domains=settings.allowed_domains,
        blocked_domains=settings.blocked_domains,
    )
    if settings.user_location is not None:
        payload["user_location"] = settings.user_location.model_dump(exclude_none=True)
    return payload


def _web_fetch_payload(
    settings: AnthropicWebFetchSettings,
    version: str,
) -> dict[str, Any]:
    payload = _base_web_tool_payload(
        tool_name="web_fetch",
        tool_type=version,
        max_uses=settings.max_uses,
        allowed_domains=settings.allowed_domains,
        blocked_domains=settings.blocked_domains,
    )
    _validate_positive(settings.max_content_tokens, "max_content_tokens", "web_fetch")
    if settings.max_content_tokens is not None:
        payload["max_content_tokens"] = settings.max_content_tokens
    if settings.citations_enabled:
        payload["citations"] = {"enabled": True}
    return payload


def build_web_tool_params(
    *,
    resolved_tools: ResolvedAnthropicWebTools,
    search_version: str | None,
    fetch_version: str | None,
    required_betas: tuple[str, ...] | None,
) -> tuple[list["BetaToolParam"], tuple[str, ...]]:
    tools: list["BetaToolParam"] = []

    if resolved_tools.search_enabled and search_version:
        tools.append(
            cast(
                "BetaToolParam",
                _web_search_payload(resolved_tools.search_settings, search_version),
            )
        )

    if resolved_tools.fetch_enabled and fetch_version:
        tools.append(
            cast(
                "BetaToolParam",
                _web_fetch_payload(resolved_tools.fetch_settings, fetch_version),
            )
        )

    return tools, tuple(required_betas or ()) if tools else ()


def dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    return unique_preserve_order(values)


def serialize_anthropic_block_payload(block: object) -> dict[str, Any] | None:
    payload = _raw_anthropic_block_payload(block)
    if not payload:
        return None

    if payload.get("type") == "text":
        text_value = payload.get("text")
        if isinstance(text_value, Mapping):
            nested_text = text_value.get("text")
            if isinstance(nested_text, str):
                payload["text"] = nested_text
        payload.pop("parsed_output", None)

    return payload


def _raw_anthropic_block_payload(block: object) -> dict[str, Any] | None:
    if isinstance(block, _SupportsModelDump):
        try:
            dumped = block.model_dump(mode="json", exclude_none=False)
        except TypeError:
            dumped = block.model_dump()
        return dumped if isinstance(dumped, dict) else None

    if not isinstance(block, Mapping):
        return None

    return {key: value for key, value in block.items() if isinstance(key, str)}


def is_server_tool_trace_payload(payload: Mapping[str, Any] | None) -> bool:
    """Return True when payload represents provider-managed server tool activity.

    Anthropic can emit server tool use blocks plus many `*_tool_result` variants
    (web search/fetch, code execution, tool search, etc). We persist these in
    assistant channels and replay them on subsequent turns so the conversation
    remains valid.
    """

    if not isinstance(payload, Mapping):
        return False

    block_type = payload.get("type")
    if not isinstance(block_type, str):
        return False

    if block_type == "server_tool_use":
        return True
    if block_type == "mcp_tool_use":
        return True
    if block_type == "mcp_tool_result":
        return True

    return block_type.endswith("_tool_result")


def extract_citation_payloads(citations: Sequence[object] | None) -> list[dict[str, str]]:
    if not citations:
        return []

    payloads: list[dict[str, str]] = []
    for citation in citations:
        raw = serialize_anthropic_block_payload(citation)
        if not raw:
            continue

        citation_type = raw.get("type")
        if not isinstance(citation_type, str) or not citation_type:
            continue

        payload: dict[str, str] = {"type": citation_type}

        title = raw.get("title")
        if not isinstance(title, str) or not title:
            title = raw.get("document_title")
        if isinstance(title, str) and title:
            payload["title"] = title

        url = raw.get("url")
        if isinstance(url, str) and url:
            payload["url"] = url

        source = raw.get("source")
        if isinstance(source, str) and source:
            payload["source"] = source

        cited_text = raw.get("cited_text")
        if isinstance(cited_text, str) and cited_text:
            payload["cited_text"] = cited_text

        payloads.append(payload)

    return payloads


def web_tool_progress_label(tool_name: str) -> str:
    if tool_name == "web_search":
        return "Searching the web"
    if tool_name == "web_fetch":
        return "Fetching URL"
    return f"Running {tool_name}"
