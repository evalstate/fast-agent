from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.web_tools import (
    ResolvedOpenAIWebSearch,
    build_web_search_tool,
    resolve_web_search,
)
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from mcp import Tool

    from fast_agent.llm.provider.openai.responses import ResponsesTransport
    from fast_agent.types import RequestParams

DEFAULT_META_AI_MODEL = "muse-spark-1.1"
META_AI_BASE_URL = "https://api.meta.ai/v1"
RESPONSE_INCLUDE_WEB_SEARCH_RESULTS = "web_search_call.results"


class MetaAIResponsesLLM(ResponsesLLM):
    """LLM implementation for MetaAI's Responses-compatible API."""

    config_section: str | None = "metaai"

    def __init__(self, provider: Provider = Provider.META_AI, **kwargs: Any) -> None:
        provider = kwargs.pop("provider", provider)
        self.config_section = "metaai"
        super().__init__(provider=provider, **kwargs)

    def _initialize_default_params(self, kwargs: dict[str, Any]) -> RequestParams:
        return self._initialize_default_params_with_model_fallback(
            kwargs,
            DEFAULT_META_AI_MODEL,
        )

    def _provider_config_fallback_sections(self) -> tuple[str, ...]:
        return ()

    def _default_transport_setting(self) -> ResponsesTransport:
        return "sse"

    @property
    def web_search_supported(self) -> bool:
        return True

    @property
    def service_tier_supported(self) -> bool:
        return False

    def _provider_base_url(self) -> str | None:
        base_url: str | None = os.getenv("META_AI_BASE_URL", META_AI_BASE_URL)
        settings = self._get_provider_config()
        if settings and getattr(settings, "base_url", None):
            base_url = settings.base_url
        return base_url

    def _provider_default_headers(self) -> dict[str, str] | None:
        settings = self._get_provider_config()
        return getattr(settings, "default_headers", None) if settings else None

    def _build_web_search_tool(
        self,
        resolved_web_search: ResolvedOpenAIWebSearch,
    ) -> dict[str, Any] | None:
        payload = build_web_search_tool(resolved_web_search)
        if payload is None:
            return None
        # Meta documents type + search_context_size + user_location only.
        sanitized: dict[str, Any] = {"type": "web_search"}
        if "search_context_size" in payload:
            sanitized["search_context_size"] = payload["search_context_size"]
        if "user_location" in payload:
            sanitized["user_location"] = payload["user_location"]
        return sanitized

    def _append_web_search_tool(self, base_args: dict[str, Any]) -> None:
        resolved_web_search = resolve_web_search(
            self._openai_settings(),
            web_search_override=self._web_search_override,
        )
        web_search_tool = self._build_web_search_tool(resolved_web_search)
        if web_search_tool is None:
            return

        self._tools_payload(base_args).append(web_search_tool)
        include_payload = base_args.get("include")
        if not isinstance(include_payload, list):
            include_payload = []
            base_args["include"] = include_payload
        if RESPONSE_INCLUDE_WEB_SEARCH_RESULTS not in include_payload:
            include_payload.append(RESPONSE_INCLUDE_WEB_SEARCH_RESULTS)

    def _build_response_args(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
    ) -> dict[str, Any]:
        args = super()._build_response_args(input_items, request_params, tools)
        # MetaAI documents portable Responses fields only; strip OpenAI extensions.
        args.pop("service_tier", None)
        include_payload = args.get("include")
        if isinstance(include_payload, list):
            include_payload = [
                value
                for value in include_payload
                if value
                not in {
                    "reasoning.encrypted_content",
                    "web_search_call.action.sources",
                }
            ]
            if include_payload:
                args["include"] = include_payload
            else:
                args.pop("include", None)
        return args
