from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from fast_agent.llm.provider.openai.responses import ResponsesTransport
    from fast_agent.types import RequestParams

DEFAULT_META_AI_MODEL = "muse-spark-1.1"
META_AI_BASE_URL = "https://api.meta.ai/v1"


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
        return False

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
