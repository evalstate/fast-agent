from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from fast_agent.llm.model_database import ModelDatabase, ModelParameters
from fast_agent.llm.model_display_name import resolve_resolved_model_display_name
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams

if TYPE_CHECKING:
    from fast_agent.llm.model_factory import ModelConfig
    from fast_agent.llm.model_info import ModelInfo
    from fast_agent.llm.model_overlays import LoadedModelOverlay
    from fast_agent.llm.reasoning_effort import ReasoningEffortSpec
    from fast_agent.llm.text_verbosity import TextVerbositySpec


ModelResolutionSource = Literal["overlay", "preset", "direct"]


@dataclass(frozen=True, slots=True)
class ResolvedModelSpec:
    """Hydrated model selection with dispatch, display, and metadata details."""

    raw_input: str
    selected_model_name: str
    source: ModelResolutionSource
    model_config: "ModelConfig"
    provider: Provider
    wire_model_name: str
    overlay: "LoadedModelOverlay | None" = None
    model_params: ModelParameters | None = None

    @property
    def overlay_name(self) -> str | None:
        overlay = self.overlay
        if overlay is None:
            return None
        return overlay.name

    @property
    def overlay_display_name(self) -> str | None:
        overlay = self.overlay
        if overlay is None:
            return None
        return overlay.display_label

    @property
    def selected_model_token(self) -> str:
        selected_model_name = self.selected_model_name.strip()
        selected_token = selected_model_name.partition("?")[0].strip()
        return selected_token or selected_model_name

    @property
    def display_name(self) -> str:
        return resolve_resolved_model_display_name(self) or self.wire_model_name

    @property
    def llm_init_kwargs(self) -> dict[str, object]:
        overlay = self.overlay
        if overlay is None:
            return {}
        return overlay.llm_init_kwargs()

    @property
    def context_window(self) -> int | None:
        model_info = self.build_model_info()
        return None if model_info is None else model_info.context_window

    @property
    def max_output_tokens(self) -> int | None:
        model_info = self.build_model_info()
        return None if model_info is None else model_info.max_output_tokens

    @property
    def default_max_tokens(self) -> int | None:
        overlay = self.overlay
        if overlay is None:
            return None
        return overlay.manifest.defaults.max_tokens

    @property
    def json_mode(self) -> str | None:
        model_params = self.model_params
        return model_params.json_mode if model_params is not None else None

    @property
    def reasoning_mode(self) -> str | None:
        model_params = self.model_params
        return model_params.reasoning if model_params is not None else None

    @property
    def reasoning_effort_spec(self) -> ReasoningEffortSpec | None:
        model_params = self.model_params
        return model_params.reasoning_effort_spec if model_params is not None else None

    @property
    def text_verbosity_spec(self) -> TextVerbositySpec | None:
        model_params = self.model_params
        return model_params.text_verbosity_spec if model_params is not None else None

    @property
    def long_context_window(self) -> int | None:
        model_params = self.model_params
        return model_params.long_context_window if model_params is not None else None

    @property
    def response_transports(self) -> tuple[Literal["sse", "websocket"], ...] | None:
        model_params = self.model_params
        return model_params.response_transports if model_params is not None else None

    @property
    def response_websocket_providers(self) -> tuple[Provider, ...] | None:
        model_params = self.model_params
        return model_params.response_websocket_providers if model_params is not None else None

    @property
    def response_service_tiers(self) -> tuple[Literal["fast", "flex"], ...] | None:
        model_params = self.model_params
        return model_params.response_service_tiers if model_params is not None else None

    @property
    def stream_mode(self) -> Literal["openai", "manual"]:
        model_params = self.model_params
        if model_params is None:
            return "openai"
        return model_params.stream_mode

    @property
    def cache_ttl(self) -> Literal["5m", "1h"] | None:
        model_params = self.model_params
        return model_params.cache_ttl if model_params is not None else None

    @property
    def anthropic_web_search_version(self) -> str | None:
        model_params = self.model_params
        return model_params.anthropic_web_search_version if model_params is not None else None

    @property
    def anthropic_web_fetch_version(self) -> str | None:
        model_params = self.model_params
        return model_params.anthropic_web_fetch_version if model_params is not None else None

    @property
    def anthropic_required_betas(self) -> tuple[str, ...] | None:
        model_params = self.model_params
        return model_params.anthropic_required_betas if model_params is not None else None

    def apply_request_defaults(
        self,
        request_params: RequestParams | None,
    ) -> RequestParams | None:
        """Apply model-selection defaults to the provided request params."""
        effective_request_params = request_params
        config = self.model_config

        sampling_overrides: dict[str, float | int] = {}
        if config.temperature is not None:
            sampling_overrides["temperature"] = config.temperature
        if config.top_p is not None:
            sampling_overrides["top_p"] = config.top_p
        if config.top_k is not None:
            sampling_overrides["top_k"] = config.top_k
        if config.min_p is not None:
            sampling_overrides["min_p"] = config.min_p
        if config.presence_penalty is not None:
            sampling_overrides["presence_penalty"] = config.presence_penalty
        if config.repetition_penalty is not None:
            sampling_overrides["repetition_penalty"] = config.repetition_penalty

        if sampling_overrides:
            if effective_request_params is None:
                effective_request_params = RequestParams().model_copy(update=sampling_overrides)
            else:
                effective_request_params = effective_request_params.model_copy(
                    update=sampling_overrides
                )

        if config.service_tier is not None:
            has_explicit_service_tier = (
                effective_request_params is not None
                and "service_tier" in effective_request_params.model_fields_set
            )
            if not has_explicit_service_tier:
                if effective_request_params is None:
                    effective_request_params = RequestParams(service_tier=config.service_tier)
                else:
                    effective_request_params = effective_request_params.model_copy(
                        update={"service_tier": config.service_tier}
                    )

        default_max_tokens = self.default_max_tokens
        if default_max_tokens is not None:
            has_explicit_max_tokens = (
                effective_request_params is not None
                and "maxTokens" in effective_request_params.model_fields_set
            )
            if not has_explicit_max_tokens:
                if effective_request_params is None:
                    effective_request_params = RequestParams(maxTokens=default_max_tokens)
                else:
                    effective_request_params = effective_request_params.model_copy(
                        update={"maxTokens": default_max_tokens}
                    )

        return effective_request_params

    def build_llm_kwargs(self) -> dict[str, object]:
        """Build constructor kwargs implied by the resolved model selection."""
        config = self.model_config
        kwargs: dict[str, object] = {}

        if config.reasoning_effort:
            kwargs["reasoning_effort"] = config.reasoning_effort
        if config.text_verbosity:
            kwargs["text_verbosity"] = config.text_verbosity
        if config.structured_output_mode:
            kwargs["structured_output_mode"] = config.structured_output_mode
        if config.long_context:
            kwargs["long_context"] = True
        if config.transport:
            kwargs["transport"] = config.transport
        if config.web_search is not None and self.provider in {
            Provider.ANTHROPIC,
            Provider.RESPONSES,
            Provider.OPENRESPONSES,
            Provider.CODEX_RESPONSES,
        }:
            kwargs["web_search"] = config.web_search
        if config.web_fetch is not None and self.provider == Provider.ANTHROPIC:
            kwargs["web_fetch"] = config.web_fetch

        return kwargs

    def build_model_info(
        self, *, context_window_override: int | None = None
    ) -> "ModelInfo | None":
        """Return effective model info for UI/reporting."""
        from fast_agent.llm.model_info import ModelInfo

        model_params = self.model_params
        if model_params is None:
            info = ModelInfo.from_name(self.wire_model_name, self.provider)
            overlay_metadata = self.overlay.manifest.metadata if self.overlay is not None else None
            if info is None and overlay_metadata is None:
                return None
            if info is not None and context_window_override is None and overlay_metadata is None:
                return info

            context_window = context_window_override
            if context_window is None and overlay_metadata is not None:
                context_window = overlay_metadata.context_window
            if context_window is None and info is not None:
                context_window = info.context_window

            max_output_tokens = None
            if overlay_metadata is not None:
                max_output_tokens = overlay_metadata.max_output_tokens
            if max_output_tokens is None and info is not None:
                max_output_tokens = info.max_output_tokens

            tokenizes = None
            if overlay_metadata is not None:
                tokenizes = overlay_metadata.tokenizes
            if tokenizes is None and info is not None:
                tokenizes = info.tokenizes

            return ModelInfo(
                name=info.name if info is not None else self.wire_model_name,
                provider=info.provider if info is not None else self.provider,
                context_window=context_window,
                max_output_tokens=max_output_tokens,
                tokenizes=tokenizes or ["text/plain"],
                json_mode=info.json_mode if info is not None else None,
                reasoning=info.reasoning if info is not None else None,
            )

        context_window = (
            context_window_override
            if context_window_override is not None
            else model_params.context_window
        )
        return ModelInfo(
            name=self.wire_model_name,
            provider=self.provider,
            context_window=context_window,
            max_output_tokens=model_params.max_output_tokens,
            tokenizes=model_params.tokenizes,
            json_mode=model_params.json_mode,
            reasoning=model_params.reasoning,
        )


def resolve_base_model_params(
    *,
    provider: Provider,
    model_name: str,
) -> ModelParameters | None:
    """Resolve base model metadata without preferring overlay runtime mutations."""
    return ModelDatabase.get_model_params(model_name, provider=provider)
