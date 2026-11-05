import os
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

from openai import APIStatusError

from fast_agent.config import HUGGINGFACE_DEFAULT_MODEL as DEFAULT_HUGGINGFACE_MODEL
from fast_agent.config import HuggingFaceSettings
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.openai.huggingface_router_profiles import (
    GENERIC_REASONING_TOGGLE,
    HUGGINGFACE_CUSTOM_ENDPOINT_BACKEND,
    HUGGINGFACE_ROUTE_PROFILES,
    HuggingFaceRouteProfile,
)
from fast_agent.llm.provider.openai.llm_openai import (
    _OpenAICompletionRequest,
    _OpenAICompletionResponse,
)
from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider.openai.request_images import (
    ChatRequestImageFitter,
    ImageBudget,
    ImageFitReport,
    request_bytes,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.llm.router_profiles import RouterRoute
from fast_agent.types import RequestParams

HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"


class HuggingFaceLLM(OpenAICompatibleLLM):
    _HF_EXTRA_BODY_SAMPLING_KEYS = (
        "top_k",
        "min_p",
        "repetition_penalty",
    )

    def __init__(self, **kwargs) -> None:
        explicit_reasoning_effort = "reasoning_effort" in kwargs
        self._hf_provider_suffix: str | None = None
        self._image_fitter = ChatRequestImageFitter()
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.HUGGINGFACE, **kwargs)
        self._apply_prompt_context_window()
        if not explicit_reasoning_effort:
            # HuggingFace inherits the OpenAI-compatible transport, but not the
            # OpenAI provider's default reasoning_effort. When no HF model query
            # or preset supplied reasoning explicitly, use the model metadata
            # default during request shaping.
            self.set_reasoning_effort(None)

    def _apply_prompt_context_window(self) -> None:
        profile = self._route_profile(self.default_request_params.model)
        if profile is None:
            return
        if profile.context_window is not None:
            self._usage_accumulator.set_context_window_size(profile.context_window)
        if profile.prompt_context_window is None:
            return
        prompt_context_window = profile.prompt_context_window
        max_tokens = self.default_request_params.max_tokens
        model_context_window = profile.context_window or self._resolved_model_spec.context_window
        if max_tokens is not None and model_context_window is not None:
            prompt_context_window = max(model_context_window - max_tokens, 1)
        self._usage_accumulator.set_context_window_size(prompt_context_window)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize HuggingFace-specific default parameters"""
        kwargs = kwargs.copy()
        requested_model = self._resolve_default_model_name(
            kwargs.get("model"),
            DEFAULT_HUGGINGFACE_MODEL,
        )
        base_model, explicit_provider = self._split_provider_suffix(requested_model)
        base_model = base_model or requested_model
        kwargs["model"] = base_model

        # Determine which provider suffix to use
        provider_suffix = explicit_provider or self._resolve_default_provider()
        self._hf_provider_suffix = provider_suffix

        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(kwargs)
        profile = self._route_profile(base_model)
        if profile and profile.omit_default_max_tokens:
            base_params.max_tokens = None
        elif profile and profile.default_max_tokens is not None:
            base_params.max_tokens = profile.default_max_tokens
        elif profile and profile.max_output_tokens is not None:
            if base_params.max_tokens is not None:
                base_params.max_tokens = min(base_params.max_tokens, profile.max_output_tokens)

        # Override with HuggingFace-specific settings
        base_params.model = base_model
        base_params.parallel_tool_calls = True

        return base_params

    def _hf_settings(self) -> HuggingFaceSettings:
        config = self.context.config
        return config.hf if config and config.hf else HuggingFaceSettings()

    def _image_budget(self) -> ImageBudget:
        settings = self._hf_settings()
        return ImageBudget(
            max_dimension=settings.image_max_dimension,
            reencode_min_bytes=settings.image_reencode_min_bytes,
            max_request_bytes=settings.max_request_bytes,
        )

    def _fit_request_images(
        self, request: _OpenAICompletionRequest, budget: ImageBudget
    ) -> tuple[_OpenAICompletionRequest, ImageFitReport]:
        arguments, report = self._image_fitter.fit(request.arguments, budget)
        if report.changed:
            self.logger.info(
                "Fitted inline images to the HuggingFace request limits",
                data={
                    "model": request.model_name,
                    "images": report.images,
                    "resized": report.resized,
                    "reencoded": report.reencoded,
                    "omitted": report.omitted,
                    "request_bytes_before": report.request_bytes_before,
                    "request_bytes_after": report.request_bytes_after,
                    "max_request_bytes": budget.max_request_bytes,
                    "dimensions": report.dimensions,
                },
            )
        return replace(request, arguments=arguments), report

    async def _run_openai_completion_request(
        self, request: _OpenAICompletionRequest
    ) -> _OpenAICompletionResponse:
        if not self._uses_huggingface_router():
            # The size limit is the router's; custom endpoints get requests unchanged.
            return await super()._run_openai_completion_request(request)
        budget = self._image_budget()
        fitted, report = self._fit_request_images(request, budget)
        try:
            return await super()._run_openai_completion_request(fitted)
        except APIStatusError as error:
            if error.status_code != 413 or not report.images:
                raise
            # The router's limit is on the body (~5 MB): shrink the image share and resend once.
            sent = request_bytes(fitted.arguments)
            retry_budget = replace(budget, max_request_bytes=sent // 2)
            retry, retry_report = self._fit_request_images(fitted, retry_budget)
            if not retry_report.changed:
                raise
            self.logger.warning(
                "HuggingFace rejected the request as too large (413); resending with fewer "
                "or smaller images",
                data={
                    "model": request.model_name,
                    "request_bytes": sent,
                    "retry_request_bytes": retry_report.request_bytes_after,
                    "omitted": retry_report.omitted,
                },
            )
            return await super()._run_openai_completion_request(retry)

    def _provider_base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.hf:
            base_url = self.context.config.hf.base_url

        return base_url if base_url else HUGGINGFACE_BASE_URL

    def _prepare_api_request(
        self, messages, tools: list | None, request_params: RequestParams
    ) -> dict[str, Any]:
        arguments = super()._prepare_api_request(messages, tools, request_params)
        self._omit_empty_tools(arguments)
        self._move_hf_sampling_fields_to_extra_body(arguments)
        self._apply_reasoning_toggle(arguments)
        profile = self._route_profile(arguments.get("model"))
        if profile and profile.use_legacy_json_schema and self._uses_huggingface_router():
            self._use_legacy_json_schema(arguments)
        if profile and profile.max_output_tokens is not None:
            max_tokens = arguments.get("max_tokens")
            if isinstance(max_tokens, int):
                arguments["max_tokens"] = min(max_tokens, profile.max_output_tokens)
        model_name = arguments.get("model")
        base_model, explicit_provider = self._split_provider_suffix(model_name)
        base_model = base_model or model_name
        if not base_model:
            return arguments

        provider_suffix = explicit_provider or self._hf_provider_suffix
        if provider_suffix:
            arguments["model"] = f"{base_model}:{provider_suffix}"
        else:
            arguments["model"] = base_model
        return arguments

    def _resolve_usage_attribution(
        self,
        model_name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, str | None]:
        wire_model = arguments.get("model")
        if not isinstance(wire_model, str):
            return model_name, None
        base_model, upstream_provider = self._split_provider_suffix(wire_model)
        return base_model or model_name, upstream_provider

    @staticmethod
    def _omit_empty_tools(arguments: dict[str, Any]) -> None:
        if arguments.get("tools") == []:
            arguments.pop("tools")
            if arguments.get("tool_choice") == "none":
                arguments.pop("tool_choice")

    @staticmethod
    def _use_legacy_json_schema(arguments: dict[str, Any]) -> None:
        response_format = arguments.get("response_format")
        if not isinstance(response_format, dict) or response_format.get("type") != "json_schema":
            return
        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict) or "schema" not in json_schema:
            return
        # HF's Together adapter flattens json_schema to {type: json_schema, schema: ...},
        # which this route rejects. Together's legacy json_object + schema envelope
        # avoids that translation. Retain the schema; this is NOT bare JSON-object mode.
        arguments["response_format"] = {"type": "json_object", "schema": json_schema["schema"]}

    def _move_hf_sampling_fields_to_extra_body(self, arguments: dict[str, Any]) -> None:
        extra_body_raw = arguments.get("extra_body", {})
        extra_body: dict[str, Any] = (
            dict(extra_body_raw) if isinstance(extra_body_raw, dict) else {}
        )

        moved = False
        for key in self._HF_EXTRA_BODY_SAMPLING_KEYS:
            if key not in arguments:
                continue
            value = arguments.pop(key)
            if value is None:
                continue
            extra_body[key] = value
            moved = True

        if moved or extra_body:
            arguments["extra_body"] = extra_body

    def _apply_reasoning_toggle(self, arguments: dict[str, Any]) -> None:
        spec = self.reasoning_effort_spec
        profile = self._route_profile(arguments.get("model"))
        reasoning_profile = profile.reasoning if profile else None
        if reasoning_profile is None:
            if spec is None or spec.kind != "toggle":
                return
            reasoning_profile = GENERIC_REASONING_TOGGLE
        reasoning_profile.apply(
            arguments,
            setting=self.reasoning_effort,
            spec=spec,
        )

    def _preserves_raw_reasoning_history(self, model: str) -> bool:
        profile = self._route_profile(model)
        return profile is not None and profile.preserve_raw_reasoning

    def _should_emit_reasoning_stream(self, reasoning_mode: str | None) -> bool:
        if reasoning_mode not in {"stream", "reasoning_content", "tags", "gpt_oss"}:
            return True
        return self._reasoning_display_enabled()

    def _structured_json_mode(self, request_params: RequestParams | None = None) -> str | None:
        model_name = (
            request_params.model
            if request_params and request_params.model
            else self.default_request_params.model
        )
        profile = self._route_profile(model_name)
        if profile and profile.structured_json_mode:
            return profile.structured_json_mode
        return super()._structured_json_mode(request_params)

    def _reasoning_display_enabled(self) -> bool:
        spec = self.reasoning_effort_spec
        if spec is None or spec.kind != "toggle":
            return True

        effective = self.reasoning_effort or spec.default
        if effective is None:
            return True
        if isinstance(effective, ReasoningEffortSetting) and effective.kind == "toggle":
            return bool(effective.value)
        return True

    def _route_profile(self, model: str | None) -> HuggingFaceRouteProfile | None:
        route = self._router_route(model)
        return HUGGINGFACE_ROUTE_PROFILES.resolve(route) if route else None

    def _router_route(self, model: str | None) -> RouterRoute | None:
        if not model:
            return None
        base_model, explicit_provider = self._split_provider_suffix(model)
        normalized_model = ModelDatabase.normalize_model_name(base_model or model)
        if not normalized_model:
            return None
        backend = explicit_provider or self._hf_provider_suffix
        if backend is None and not self._uses_huggingface_router():
            # A dedicated HF endpoint does not use a router-provider suffix in
            # its wire model. Use an internal marker solely for route-profile
            # selection so known model contracts still apply.
            backend = HUGGINGFACE_CUSTOM_ENDPOINT_BACKEND
        return RouterRoute(
            model=normalized_model,
            backend=backend,
        )

    def _uses_huggingface_router(self) -> bool:
        effective = urlsplit(self._base_url() or "")
        default = urlsplit(HUGGINGFACE_BASE_URL)
        effective_port = effective.port or (443 if effective.scheme.casefold() == "https" else 80)
        default_port = default.port or (443 if default.scheme.casefold() == "https" else 80)
        return (
            effective.scheme.casefold() == default.scheme.casefold()
            and effective.hostname == default.hostname
            and effective_port == default_port
            and effective.path.rstrip("/") == default.path.rstrip("/")
            and effective.query == default.query
        )

    def _resolve_default_provider(self) -> str | None:
        config_provider = None
        if self.context and self.context.config and self.context.config.hf:
            config_provider = self.context.config.hf.default_provider
        env_provider = os.getenv("HF_DEFAULT_PROVIDER")
        return config_provider or env_provider

    @staticmethod
    def _split_provider_suffix(model: str | None) -> tuple[str | None, str | None]:
        if not model or ":" not in model:
            return model, None
        base, suffix = model.rsplit(":", 1)
        if not base:
            return model, None
        return base, suffix or None

    def get_hf_display_info(self) -> dict[str, str]:
        """Return display information for HuggingFace model and provider.

        Returns:
            dict with 'model' and 'provider' keys
        """
        model = self.default_request_params.model if self.default_request_params else None
        provider = self._hf_provider_suffix or "auto-routing"
        return {"model": model or DEFAULT_HUGGINGFACE_MODEL, "provider": provider}
