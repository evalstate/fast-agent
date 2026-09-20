"""Copilot routing for the existing Anthropic Messages turn implementation."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient, omit

from fast_agent.config import AnthropicSettings, CopilotSettings
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM, CacheTTL
from fast_agent.llm.provider.copilot.models import get_copilot_model
from fast_agent.llm.provider.copilot.policy import apply_policy, reject_overrides
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from fast_agent.llm.provider.copilot.endpoint import CopilotEndpoint
    from fast_agent.llm.structured_output_mode import StructuredOutputMode
    from fast_agent.types import RequestParams


class CopilotMessagesLLM(AnthropicLLM):
    config_section = "copilot"

    def __init__(self, **kwargs: Any) -> None:
        reject_overrides(kwargs)
        if kwargs.get("web_search") or kwargs.get("web_fetch"):
            raise ValueError("Copilot provider web tools are not supported.")
        if kwargs.get("transport") not in (None, "sse"):
            raise ValueError("Copilot Messages only supports SSE.")
        self._copilot_owner_id = uuid4().hex
        self._copilot_endpoint: ContextVar[CopilotEndpoint] = ContextVar(
            "copilot_messages_endpoint"
        )
        super().__init__(**kwargs)
        spec = get_copilot_model(self.default_request_params.model or "")
        if spec.wire_api != "messages":
            raise ValueError("Copilot Messages requires a Messages model.")

    @classmethod
    def provider_identity(cls) -> Provider:
        return Provider.COPILOT

    def _get_provider_config(self) -> None:
        return None

    def _anthropic_settings(self) -> None:
        return None

    def _provider_config_fallback_sections(self) -> tuple[str, ...]:
        return ()

    def validate_provider_credentials(self) -> None:
        # Credentials exist only after an asynchronous broker resolution.
        return None

    async def _prepare_anthropic_client(self, model: str) -> None:
        if self.provider_managed_mcp_state.has_servers():
            raise ValueError("Copilot native MCP is not supported.")
        from fast_agent.llm.provider.copilot.broker import get_copilot_broker

        endpoint = await get_copilot_broker(self.context).resolve(
            model, owner_id=self._copilot_owner_id, transport="sse"
        )
        self._copilot_endpoint.set(endpoint)

    def _provider_base_url(self) -> str:
        return self._copilot_endpoint.get().base_url

    def _provider_default_headers(self) -> dict[str, str]:
        return dict(self._copilot_endpoint.get().headers)

    def _initialize_anthropic_client(self) -> AsyncAnthropic:
        endpoint = self._copilot_endpoint.get()
        # The broker authenticates with a bearer header. Omit the SDK's API-key
        # header so it sends none; the omit sentinel also satisfies SDK auth
        # validation without borrowing environment credentials.
        headers: dict[str, Any] = {**endpoint.headers, "X-Api-Key": omit}
        http_client = DefaultAsyncHttpxClient(trust_env=False, follow_redirects=False)
        # The pinned Anthropic SDK installs environment proxy mounts even when
        # trust_env=False. Keep its default transport/limits, but remove those routes.
        http_client._mounts.clear()
        client = AsyncAnthropic(
            api_key="",
            base_url=endpoint.base_url.rstrip("/").removesuffix("/v1"),
            default_headers=headers,
            max_retries=0,
            http_client=http_client,
        )
        # The SDK merges ANTHROPIC_CUSTOM_HEADERS during construction. Replace,
        # rather than merge, with broker headers (including the auth omit sentinel).
        client._custom_headers = headers
        return client

    def supports_files_api(self) -> bool:
        return False

    def supports_web_tools(self) -> bool:
        return False

    def supports_direct_anthropic_beta(self, feature: str) -> bool:
        return False

    def _get_cache_mode(self) -> str:
        settings = self.context.config.copilot if self.context.config else CopilotSettings()
        return settings.cache_mode

    def _get_cache_ttl(self) -> CacheTTL:
        # Reuse model defaults/planning, not direct Anthropic account configuration.
        settings = self.context.config.copilot if self.context.config else CopilotSettings()
        return settings.cache_ttl or self.resolved_model.cache_ttl or AnthropicSettings().cache_ttl

    def _cache_diagnostics_enabled(self) -> bool:
        return False

    @property
    def web_tools_enabled(self) -> tuple[bool, bool]:
        return False, False

    def set_web_search_enabled(self, value: bool | None) -> None:
        if value:
            raise ValueError("Copilot provider web tools are not supported.")

    def set_web_fetch_enabled(self, value: bool | None) -> None:
        if value:
            raise ValueError("Copilot provider web tools are not supported.")

    def _resolve_thinking_arguments(
        self, model: str, max_tokens: int | None, structured_mode: StructuredOutputMode | None
    ) -> tuple[dict[str, Any], bool]:
        arguments, enabled = super()._resolve_thinking_arguments(model, max_tokens, structured_mode)
        if max_tokens is not None:
            thinking = arguments.get("thinking")
            if enabled and isinstance(thinking, dict) and thinking.get("type") == "enabled":
                budget = thinking.get("budget_tokens")
                if isinstance(budget, int) and max_tokens <= budget:
                    raise ValueError(
                        f"Copilot max_tokens ({max_tokens}) must exceed the enabled thinking "
                        f"budget ({budget}). Increase max_tokens or disable thinking with "
                        "?reasoning=off."
                    )
            arguments["max_tokens"] = max_tokens
        return arguments, enabled

    def prepare_provider_arguments(
        self,
        base_args: dict,
        request_params: RequestParams,
        exclude_fields: set | None = None,
    ) -> dict:
        if request_params.service_tier is not None:
            raise ValueError("Copilot does not support service_tier selection; omit it.")
        arguments = super().prepare_provider_arguments(base_args, request_params, exclude_fields)
        spec = get_copilot_model(self.default_request_params.model or "")
        endpoint = self._copilot_endpoint.get()
        return apply_policy(arguments, spec, endpoint.headers)
