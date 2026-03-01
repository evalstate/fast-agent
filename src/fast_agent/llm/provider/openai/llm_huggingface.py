import os
from typing import Any

from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.types import RequestParams

HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_HUGGINGFACE_MODEL = "moonshotai/Kimi-K2-Instruct-0905"


class HuggingFaceLLM(OpenAICompatibleLLM):
    _HF_EXTRA_BODY_SAMPLING_KEYS = (
        "top_k",
        "min_p",
        "repetition_penalty",
    )

    def __init__(self, **kwargs) -> None:
        self._hf_provider_suffix: str | None = None
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.HUGGINGFACE, **kwargs)

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

    def _base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.hf:
            base_url = self.context.config.hf.base_url

        return base_url if base_url else HUGGINGFACE_BASE_URL

    def _prepare_api_request(
        self, messages, tools: list | None, request_params: RequestParams
    ) -> dict[str, Any]:
        arguments = super()._prepare_api_request(messages, tools, request_params)
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

    def _apply_reasoning_toggle(self, arguments: dict[str, Any]) -> None:
        spec = self.reasoning_effort_spec
        if not spec or spec.kind != "toggle":
            return
        effective = self.reasoning_effort or spec.default
        if not effective or effective.kind != "toggle":
            return

        disable_reasoning = not bool(effective.value)
        uses_kimi_toggle = self._uses_kimi_thinking_toggle(arguments.get("model"))
        uses_qwen_chat_toggle = self._uses_qwen_chat_template_toggle(arguments.get("model"))
        if (
            not uses_kimi_toggle
            and not uses_qwen_chat_toggle
            and not disable_reasoning
            and self.reasoning_effort is None
        ):
            return

        extra_body_raw = arguments.get("extra_body", {})
        extra_body: dict[str, Any] = extra_body_raw if isinstance(extra_body_raw, dict) else {}
        if uses_kimi_toggle:
            thinking_type = "disabled" if disable_reasoning else "enabled"
            extra_body["thinking"] = {"type": thinking_type}
        elif uses_qwen_chat_toggle:
            chat_kwargs_raw = extra_body.get("chat_template_kwargs", {})
            chat_kwargs = chat_kwargs_raw if isinstance(chat_kwargs_raw, dict) else {}
            chat_kwargs["enable_thinking"] = not disable_reasoning
            extra_body["chat_template_kwargs"] = chat_kwargs
        else:
            extra_body["disable_reasoning"] = disable_reasoning
        arguments["extra_body"] = extra_body

    def _should_emit_reasoning_stream(self, reasoning_mode: str | None) -> bool:
        if reasoning_mode not in {"stream", "reasoning_content", "tags", "gpt_oss"}:
            return True
        return self._reasoning_display_enabled()

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
    def _uses_kimi_thinking_toggle(model: str | None) -> bool:
        if not model:
            return False
        return ModelDatabase.normalize_model_name(model) == "moonshotai/kimi-k2.5"

    @staticmethod
    def _uses_qwen_chat_template_toggle(model: str | None) -> bool:
        if not model:
            return False
        return ModelDatabase.normalize_model_name(model) == "qwen/qwen3.5-397b-a17b"

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
