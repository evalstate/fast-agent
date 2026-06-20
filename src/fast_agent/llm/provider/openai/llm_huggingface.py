import os
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.types import RequestParams

HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_HUGGINGFACE_MODEL = "moonshotai/Kimi-K2-Instruct-0905"


@dataclass(frozen=True)
class _HFRouteProfile:
    """Provider-route request-shaping profile for HF-hosted model deployments.

    Keep canonical model capabilities in ModelDatabase; use route profiles for
    provider-specific wire contracts such as reasoning payload shape and JSON
    mode quirks.
    """

    structured_json_mode: Literal["schema", "object"] | None = None
    reasoning_api: Literal["thinking_with_effort", "reasoning_effort"] = "thinking_with_effort"
    effort_map: dict[str, str] | None = None


class HuggingFaceLLM(OpenAICompatibleLLM):
    _HF_EXTRA_BODY_SAMPLING_KEYS = (
        "top_k",
        "min_p",
        "repetition_penalty",
    )
    _GLM_52_ROUTE_PROFILES: ClassVar[dict[str, _HFRouteProfile]] = {
        # Z.ai-compatible HF router payload:
        # thinking.enabled/disabled plus reasoning_effort when enabled.
        "zai-org": _HFRouteProfile(),
        "novita": _HFRouteProfile(),
        "together": _HFRouteProfile(),
        "deepinfra": _HFRouteProfile(reasoning_api="reasoning_effort", effort_map={"max": "xhigh"}),
        "fireworks-ai": _HFRouteProfile(reasoning_api="reasoning_effort"),
    }

    def __init__(self, **kwargs) -> None:
        explicit_reasoning_effort = "reasoning_effort" in kwargs
        self._hf_provider_suffix: str | None = None
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.HUGGINGFACE, **kwargs)
        if not explicit_reasoning_effort:
            # HuggingFace inherits the OpenAI-compatible transport, but not the
            # OpenAI provider's default reasoning_effort. When no HF model query
            # or preset supplied reasoning explicitly, use the model metadata
            # default during request shaping.
            self.set_reasoning_effort(None)

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

        # Override with HuggingFace-specific settings
        base_params.model = base_model
        base_params.parallel_tool_calls = True

        return base_params

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

    @staticmethod
    def _omit_empty_tools(arguments: dict[str, Any]) -> None:
        if arguments.get("tools") == []:
            arguments.pop("tools")
            if arguments.get("tool_choice") == "none":
                arguments.pop("tool_choice")

    def _move_hf_sampling_fields_to_extra_body(self, arguments: dict[str, Any]) -> None:
        extra_body_raw = arguments.get("extra_body", {})
        extra_body: dict[str, Any] = extra_body_raw if isinstance(extra_body_raw, dict) else {}

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

    @staticmethod
    def _set_chat_template_kwarg(
        extra_body: dict[str, Any],
        key: str,
        value: bool,
    ) -> None:
        chat_kwargs_raw = extra_body.get("chat_template_kwargs", {})
        chat_kwargs = chat_kwargs_raw if isinstance(chat_kwargs_raw, dict) else {}
        chat_kwargs[key] = value
        extra_body["chat_template_kwargs"] = chat_kwargs

    def _apply_reasoning_toggle(self, arguments: dict[str, Any]) -> None:
        if self._uses_glm_52_reasoning_effort(arguments.get("model")):
            self._apply_glm_52_reasoning_effort(arguments)
            return

        spec = self.reasoning_effort_spec
        if not spec or spec.kind != "toggle":
            return
        effective = self.reasoning_effort or spec.default
        if not effective or effective.kind != "toggle":
            return

        disable_reasoning = not bool(effective.value)
        uses_kimi_25_chat_toggle = self._uses_kimi_25_chat_toggle(arguments.get("model"))
        uses_kimi_26_chat_toggle = self._uses_kimi_26_chat_toggle(arguments.get("model"))
        uses_chat_template_enable_thinking = self._uses_enable_thinking_chat_template_toggle(
            arguments.get("model")
        )
        if (
            not uses_kimi_25_chat_toggle
            and not uses_kimi_26_chat_toggle
            and not uses_chat_template_enable_thinking
            and not disable_reasoning
            and self.reasoning_effort is None
        ):
            return

        extra_body_raw = arguments.get("extra_body", {})
        extra_body: dict[str, Any] = extra_body_raw if isinstance(extra_body_raw, dict) else {}
        if uses_kimi_25_chat_toggle:
            # Kimi 2.5 defaults to thinking-enabled on the provider side.
            # Only send the explicit instant-mode disable flag when reasoning is off.
            # Hugging Face's router expects Moonshot's official API thinking config,
            # not vLLM/SGLang's chat_template_kwargs override.
            if disable_reasoning:
                extra_body["thinking"] = {"type": "disabled"}
                arguments["extra_body"] = extra_body
            return
        if uses_kimi_26_chat_toggle:
            # Kimi 2.6 also defaults to thinking-enabled on the provider side.
            # Instant mode is exposed via the model chat template toggle.
            if disable_reasoning:
                self._set_chat_template_kwarg(extra_body, "thinking", False)
                arguments["extra_body"] = extra_body
            return
        if uses_chat_template_enable_thinking:
            self._set_chat_template_kwarg(
                extra_body,
                "enable_thinking",
                not disable_reasoning,
            )
        else:
            extra_body["disable_reasoning"] = disable_reasoning
        arguments["extra_body"] = extra_body

    def _apply_glm_52_reasoning_effort(self, arguments: dict[str, Any]) -> None:
        spec = self.reasoning_effort_spec
        setting = self.reasoning_effort or (spec.default if spec else None)
        profile = self._glm_52_route_profile(arguments.get("model"))

        extra_body_raw = arguments.get("extra_body", {})
        extra_body: dict[str, Any] = extra_body_raw if isinstance(extra_body_raw, dict) else {}

        disabled = False
        effort = "max"
        if setting is not None:
            if setting.kind == "toggle":
                disabled = setting.value is False
            elif setting.kind == "effort" and isinstance(setting.value, str):
                disabled = setting.value in {"none", "minimal"}
                effort = setting.value

        if profile and profile.effort_map:
            effort = profile.effort_map.get(effort, effort)

        if profile and profile.reasoning_api == "reasoning_effort":
            extra_body.pop("thinking", None)
            arguments["reasoning_effort"] = "none" if disabled else effort
            if extra_body:
                arguments["extra_body"] = extra_body
            else:
                arguments.pop("extra_body", None)
            return

        if disabled:
            extra_body["thinking"] = {"type": "disabled"}
            arguments.pop("reasoning_effort", None)
        else:
            extra_body["thinking"] = {"type": "enabled", "clear_thinking": False}
            arguments["reasoning_effort"] = effort
        arguments["extra_body"] = extra_body

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
        base_model, provider = self._split_provider_suffix(model_name)
        profile = self._glm_52_route_profile_for_parts(
            base_model, provider or self._hf_provider_suffix
        )
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

    @staticmethod
    def _uses_glm_52_reasoning_effort(model: str | None) -> bool:
        if not model:
            return False
        return ModelDatabase.normalize_model_name(model) == "zai-org/glm-5.2"

    def _glm_52_route_profile(self, model: str | None) -> _HFRouteProfile | None:
        base_model, provider = self._split_provider_suffix(model)
        provider = provider or self._hf_provider_suffix
        return self._glm_52_route_profile_for_parts(base_model, provider)

    @staticmethod
    def _glm_52_route_profile_for_parts(
        base_model: str | None, provider: str | None
    ) -> _HFRouteProfile | None:
        if not HuggingFaceLLM._uses_glm_52_reasoning_effort(base_model):
            return None
        if not provider:
            return None
        return HuggingFaceLLM._GLM_52_ROUTE_PROFILES.get(provider)

    @staticmethod
    def _uses_kimi_25_chat_toggle(model: str | None) -> bool:
        if not model:
            return False
        return ModelDatabase.normalize_model_name(model) == "moonshotai/kimi-k2.5"

    @staticmethod
    def _uses_kimi_26_chat_toggle(model: str | None) -> bool:
        if not model:
            return False
        return ModelDatabase.normalize_model_name(model) == "moonshotai/kimi-k2.6"

    @staticmethod
    def _uses_enable_thinking_chat_template_toggle(model: str | None) -> bool:
        if not model:
            return False
        return ModelDatabase.normalize_model_name(model) in {
            "qwen/qwen3.5-397b-a17b",
            "qwen/qwen3.6-35b-a3b",
            "google/gemma-4-31b-it",
        }

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
