from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_HUGGINGFACE_MODEL = "moonshotai/Kimi-K2-Instruct-0905"


class HuggingFaceLLM(OpenAICompatibleLLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.HUGGINGFACE, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize HuggingFace-specific default parameters"""
        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(kwargs)

        # Override with HuggingFace-specific settings
        chosen_model = kwargs.get("model", DEFAULT_HUGGINGFACE_MODEL)
        base_params.model = chosen_model
        base_params.parallel_tool_calls = True

        return base_params

    def _base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.huggingface:
            base_url = self.context.config.huggingface.base_url

        return base_url if base_url else HUGGINGFACE_BASE_URL
