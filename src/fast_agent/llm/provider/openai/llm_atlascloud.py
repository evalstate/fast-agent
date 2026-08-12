from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

ATLASCLOUD_BASE_URL = "https://api.atlascloud.ai/v1"
DEFAULT_ATLASCLOUD_MODEL = "qwen/qwen3.8-max"


class AtlasCloudLLM(OpenAICompatibleLLM):
    """Atlas Cloud provider using its OpenAI-compatible Chat Completions API."""

    def __init__(self, **kwargs) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.ATLASCLOUD, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        return self._initialize_default_params_with_model_fallback(kwargs, DEFAULT_ATLASCLOUD_MODEL)

    def _provider_base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.atlascloud:
            base_url = self.context.config.atlascloud.base_url

        return base_url if base_url else ATLASCLOUD_BASE_URL
