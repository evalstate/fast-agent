from fast_agent.llm.provider.openai.llm_groq import GroqLLM
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-turbo"


class AliyunLLM(GroqLLM):
    def __init__(self, *args, **kwargs) -> None:
        OpenAILLM.__init__(self, *args, provider=Provider.ALIYUN, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Aliyun-specific default parameters"""
        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(kwargs)

        # Override with Aliyun-specific settings
        chosen_model = kwargs.get("model", DEFAULT_QWEN_MODEL)
        base_params.model = chosen_model
        base_params.parallel_tool_calls = True

        return base_params

    def _base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.aliyun:
            base_url = self.context.config.aliyun.base_url

        return base_url if base_url else ALIYUN_BASE_URL
