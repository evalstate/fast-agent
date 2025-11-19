import os

from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_HUGGINGFACE_MODEL = "moonshotai/Kimi-K2-Instruct-0905"


class HuggingFaceLLM(OpenAICompatibleLLM):
    def __init__(self, *args, **kwargs) -> None:
        self._hf_provider_suffix: str | None = None
        super().__init__(*args, provider=Provider.HUGGINGFACE, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize HuggingFace-specific default parameters"""
        kwargs = kwargs.copy()
        requested_model = kwargs.get("model") or DEFAULT_HUGGINGFACE_MODEL
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
    ) -> dict[str, str]:
        arguments = super()._prepare_api_request(messages, tools, request_params)
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
