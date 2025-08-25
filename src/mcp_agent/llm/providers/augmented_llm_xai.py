import os

from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM

XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_XAI_MODEL = "grok-3"


class XAIAugmentedLLM(OpenAIAugmentedLLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args, provider=Provider.XAI, **kwargs
        )  # Properly pass args and kwargs to parent

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize xAI parameters"""
        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(kwargs)
        
        # Override with xAI-specific settings
        chosen_model = kwargs.get("model", DEFAULT_XAI_MODEL)
        base_params.model = chosen_model
        base_params.parallel_tool_calls = False
        
        return base_params

    def _base_url(self) -> str:
        base_url = os.getenv("XAI_BASE_URL", XAI_BASE_URL)
        if self.context.config and self.context.config.xai:
            base_url = self.context.config.xai.base_url

        return base_url

    async def _is_tool_stop_reason(self, finish_reason: str) -> bool:
        # grok uses Null as the finish reason for tool calls?
        return await super()._is_tool_stop_reason(finish_reason) or finish_reason is None
