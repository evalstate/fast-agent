"""Copilot routing for the existing Anthropic Messages turn implementation."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient, omit

from fast_agent.config import AnthropicSettings
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.compaction_support import CompactionSupport
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM, CacheTTL
from fast_agent.llm.provider.copilot import broker
from fast_agent.llm.provider.copilot.broker import CopilotEndpoint, copilot_settings
from fast_agent.llm.provider.copilot.images import MAX_MESSAGES_URL_IMAGES, CopilotImageUploads
from fast_agent.llm.provider.copilot.models import get_copilot_model
from fast_agent.llm.provider.copilot.policy import apply_policy, contains_type, reject_overrides
from fast_agent.llm.provider.request_images import ImageBudget, ImageFitReport, request_bytes
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from anthropic.types.beta import BetaMessageParam

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
        self._image_uploads = CopilotImageUploads()
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
        endpoint = await broker.get_copilot_broker(self.context).resolve(
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

    def _anthropic_image_budget(self) -> ImageBudget:
        return replace(
            super()._anthropic_image_budget(),
            url_image_slots=MAX_MESSAGES_URL_IMAGES,
            max_url_images=MAX_MESSAGES_URL_IMAGES,
        )

    def _anthropic_413_retry_budget(self, sent_bytes: int) -> ImageBudget:
        return replace(
            super()._anthropic_413_retry_budget(sent_bytes),
            url_image_slots=MAX_MESSAGES_URL_IMAGES,
            max_url_images=MAX_MESSAGES_URL_IMAGES,
        )

    async def _upload_messages_with_fallback(
        self,
        arguments: dict[str, Any],
        budget: ImageBudget,
    ) -> tuple[dict[str, Any], int]:
        endpoint = self._copilot_endpoint.get()
        normalized = await self._image_uploads.normalize(
            {"messages": arguments["messages"]},
            endpoint,
            on_progress=partial(self._log_upload_progress, model=endpoint.model_id, noun="image"),
        )
        updated = {**arguments, "messages": cast("list[BetaMessageParam]", normalized["messages"])}
        if (
            budget.max_request_bytes is not None
            and request_bytes(updated) > budget.max_request_bytes
        ):
            updated, fallback = self._image_fitter.fit(
                updated,
                replace(budget, url_image_slots=0),
                update_watermark=False,
            )
            return updated, fallback.omitted
        return updated, 0

    async def _fit_anthropic_request_images(
        self,
        base_args: dict[str, Any],
        *,
        model: str,
    ) -> tuple[dict[str, Any], ImageFitReport]:
        fitted, report = await super()._fit_anthropic_request_images(base_args, model=model)
        if not report.images:
            return fitted, report
        fitted, fallback_omitted = await self._upload_messages_with_fallback(
            fitted, self._anthropic_image_budget()
        )
        if fallback_omitted:
            report.omitted += fallback_omitted
            report.request_bytes_after = request_bytes(fitted)
        return fitted, report

    async def _prepare_anthropic_413_retry_arguments(
        self,
        arguments: dict[str, Any],
        retry_budget: ImageBudget,
        omitted: int,
    ) -> tuple[dict[str, Any], int]:
        arguments, fallback_omitted = await self._upload_messages_with_fallback(
            arguments, retry_budget
        )
        omitted += fallback_omitted
        extra_headers = arguments.get("extra_headers")
        if (
            isinstance(extra_headers, dict)
            and "copilot-vision-request" in extra_headers
            and not contains_type(arguments.get("messages"), {"image", "input_image", "image_url"})
        ):
            updated_headers = {
                k: v for k, v in extra_headers.items() if k != "copilot-vision-request"
            }
            arguments = {**arguments, "extra_headers": updated_headers}
        return arguments, omitted

    def supports_files_api(self) -> bool:
        return False

    def supports_web_tools(self) -> bool:
        return False

    @property
    def compaction_support(self) -> CompactionSupport:
        # The gateway forwards the threshold beta but strips the on-demand one.
        spec = get_copilot_model(self.default_request_params.model or "")
        if spec.supports_threshold_compaction and self.compaction_mode != "client":
            return CompactionSupport.IN_REQUEST
        return CompactionSupport.CLIENT

    def supports_direct_anthropic_beta(self, feature: str) -> bool:
        # Verified against the gateway; other betas are accepted but untested or inert.
        return feature == "interleaved_thinking"

    def _supports_native_json_schema(self, model: str) -> bool:
        # These models cannot use the legacy forced-tool structured-output fallback.
        # Native output_config.format does not require direct Anthropic beta headers.
        return model in {
            "claude-fable-5.1",
            "claude-fable-5.2",
            "claude-opus-5.5",
            "claude-sonnet-5.5",
            "claude-haiku-5.5",
        }

    def _get_cache_mode(self) -> str:
        return copilot_settings(self.context).cache_mode

    def _get_cache_ttl(self) -> CacheTTL:
        # Reuse model defaults/planning, not direct Anthropic account configuration.
        settings = copilot_settings(self.context)
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
        return apply_policy(
            arguments,
            spec,
            endpoint.headers,
            agent_initiated=self.verb == ProgressAction.WARMING_CACHE,
        )
