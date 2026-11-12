from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from fast_agent.llm.provider.openai.openresponses_streaming import OpenResponsesStreamingMixin
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.responses_raw_stream import open_raw_responses_stream
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import TurnUsage, usage_from_responses_compatible

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.responses import ResponseUsage

    from fast_agent.types import RequestParams

DEFAULT_OPENRESPONSES_API_KEY = ""
DEFAULT_OPENRESPONSES_BASE_URL = "http://localhost:8080/v1"


class OpenResponsesLLM(OpenResponsesStreamingMixin, ResponsesLLM):
    """LLM implementation for Open Responses-compatible APIs."""

    config_section: str | None = "openresponses"
    _OPENRESPONSES_EXTRA_BODY_SAMPLING_KEYS = (
        "top_k",
        "min_p",
        "repetition_penalty",
    )

    def __init__(self, provider: Provider = Provider.OPENRESPONSES, **kwargs: Any) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=provider, **kwargs)

    def _translate_responses_usage(
        self,
        usage: ResponseUsage,
        *,
        provider: Provider,
        model: str,
    ) -> TurnUsage:
        return usage_from_responses_compatible(
            usage.model_dump(mode="json"),
            provider=provider,
            model=model,
        )

    def _provider_api_key(self) -> str:
        from fast_agent.llm.provider_key_manager import ProviderKeyManager

        api_key = ProviderKeyManager.get_config_file_key(
            self.provider.config_name,
            self.context.config,
        )
        if api_key is not None:
            return api_key

        env_key = ProviderKeyManager.get_env_var(self.provider.config_name)
        if env_key is not None:
            return env_key

        return DEFAULT_OPENRESPONSES_API_KEY

    def _openresponses_settings(self) -> Any | None:
        config = getattr(self.context, "config", None)
        return getattr(config, "openresponses", None) if config is not None else None

    def _provider_config_fallback_sections(self) -> tuple[str, ...]:
        """OpenResponses settings should not inherit OpenAI provider defaults."""
        return ()

    @property
    def web_search_supported(self) -> bool:
        """OpenResponses backends vary; don't advertise interactive web search controls."""
        return False

    @property
    def service_tier_supported(self) -> bool:
        """OpenResponses backends vary; don't advertise interactive service-tier controls."""
        return False

    def _provider_base_url(self) -> str | None:
        base_url = os.getenv("OPENRESPONSES_BASE_URL", DEFAULT_OPENRESPONSES_BASE_URL)
        settings = self._openresponses_settings()
        if settings and settings.base_url:
            return settings.base_url
        return base_url

    def _responses_client(self) -> AsyncOpenAI:
        api_key = self._api_key()
        if api_key:
            return super()._responses_client()

        original_api_key = self._init_api_key
        self._init_api_key = DEFAULT_OPENRESPONSES_API_KEY or "openresponses-local"
        try:
            return super()._responses_client()
        finally:
            self._init_api_key = original_api_key

    def _build_response_args(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list | None,
    ) -> dict[str, Any]:
        arguments = super()._build_response_args(input_items, request_params, tools)
        self._move_openresponses_sampling_fields_to_extra_body(arguments)
        return arguments

    def _move_openresponses_sampling_fields_to_extra_body(self, arguments: dict[str, Any]) -> None:
        extra_body_raw = arguments.get("extra_body", {})
        extra_body: dict[str, Any] = extra_body_raw if isinstance(extra_body_raw, dict) else {}

        moved = False
        for key in self._OPENRESPONSES_EXTRA_BODY_SAMPLING_KEYS:
            if key not in arguments:
                continue
            value = arguments.pop(key)
            if value is None:
                continue
            extra_body[key] = value
            moved = True

        if moved or extra_body:
            arguments["extra_body"] = extra_body

    def _response_sse_stream(
        self,
        *,
        client: AsyncOpenAI,
        arguments: dict[str, Any],
        timeout_seconds: float | None = None,
    ):
        return open_raw_responses_stream(
            client=client,
            arguments=arguments,
            timeout_seconds=timeout_seconds,
            label="OpenResponses",
        )
