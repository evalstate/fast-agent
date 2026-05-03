from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from fast_agent.llm.provider.openai.llm_xai import XAI_BASE_URL
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.responses_websocket import (
    ResponsesWsRequestPlanner,
    StatelessResponsesWsPlanner,
)
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from mcp import Tool

    from fast_agent.llm.provider.openai.responses import ResponsesTransport
    from fast_agent.types import RequestParams

DEFAULT_XAI_RESPONSES_MODEL = "grok-4.3"


class XAIResponsesLLM(ResponsesLLM):
    """LLM implementation for xAI's Responses-compatible API."""

    config_section: str | None = "xai"

    def __init__(self, provider: Provider = Provider.XAI, **kwargs: Any) -> None:
        provider = kwargs.pop("provider", provider)
        self.config_section = "xairesponses" if provider == Provider.XAI_RESPONSES else "xai"
        super().__init__(provider=provider, **kwargs)

    def _initialize_default_params(self, kwargs: dict[str, Any]) -> RequestParams:
        params = self._initialize_default_params_with_model_fallback(
            kwargs,
            DEFAULT_XAI_RESPONSES_MODEL,
        )
        params.parallel_tool_calls = False
        return params

    def _provider_config_fallback_sections(self) -> tuple[str, ...]:
        return ("xai",)

    def _default_transport_setting(self) -> ResponsesTransport:
        return "auto"

    @property
    def web_search_supported(self) -> bool:
        return False

    @property
    def service_tier_supported(self) -> bool:
        return False

    def _provider_base_url(self) -> str | None:
        base_url: str | None = os.getenv("XAI_BASE_URL", XAI_BASE_URL)
        settings = self._get_provider_config()
        if settings and getattr(settings, "base_url", None):
            base_url = settings.base_url
        return base_url

    def _provider_default_headers(self) -> dict[str, str] | None:
        settings = self._get_provider_config()
        return getattr(settings, "default_headers", None) if settings else None

    def _build_websocket_headers(self) -> dict[str, str]:
        headers = dict(self._default_headers() or {})
        headers.setdefault("Authorization", f"Bearer {self._api_key()}")
        return headers

    def _new_ws_request_planner(self) -> ResponsesWsRequestPlanner:
        # Live xAI websocket smoke tests currently hang on store=false
        # `previous_response_id` continuations. Keep ZDR/store=false semantics
        # by replaying full context on each websocket turn until xAI's in-memory
        # continuation path behaves as documented.
        return StatelessResponsesWsPlanner()

    def _build_response_args(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
    ) -> dict[str, Any]:
        args = super()._build_response_args(input_items, request_params, tools)
        # Keep the first pass xAI payload conservative; these are OpenAI-specific
        # Responses extensions and xAI's websocket docs show the portable core.
        args.pop("include", None)
        args.pop("service_tier", None)
        args.pop("reasoning", None)
        return args


class XAIExplicitResponsesLLM(XAIResponsesLLM):
    """Compatibility provider for the explicit `xairesponses` provider name."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(provider=Provider.XAI_RESPONSES, **kwargs)
