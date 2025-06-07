from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM

ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-turbo"


class AliyunAugmentedLLM(OpenAIAugmentedLLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.ALIYUN, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Aliyun-specific default parameters"""
        chosen_model = kwargs.get("model", DEFAULT_QWEN_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.aliyun:
            base_url = self.context.config.aliyun.base_url

        return base_url if base_url else ALIYUN_BASE_URL
