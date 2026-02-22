from fast_agent.llm.provider.openai.llm_groq import GroqLLM
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

ALIYUN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-turbo"


class AliyunLLM(GroqLLM):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("provider", None)
        OpenAILLM.__init__(self, provider=Provider.ALIYUN, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Aliyun-specific default parameters"""
        chosen_model = self._resolve_default_model_name(kwargs.get("model"), DEFAULT_QWEN_MODEL)
        resolved_kwargs = dict(kwargs)
        if chosen_model is not None:
            resolved_kwargs["model"] = chosen_model

        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(resolved_kwargs)

        # Override with Aliyun-specific settings
        base_params.model = chosen_model
        base_params.parallel_tool_calls = True

        return base_params

    def _base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.aliyun:
            base_url = self.context.config.aliyun.base_url

        return base_url if base_url else ALIYUN_BASE_URL
