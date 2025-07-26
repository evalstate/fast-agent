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
        chosen_model = kwargs.get("model", DEFAULT_XAI_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=False,
            max_iterations=20,
            use_history=True,
        )

    def _base_url(self) -> str:
        base_url = os.getenv("XAI_BASE_URL", XAI_BASE_URL)
        if self.context.config and self.context.config.xai:
            base_url = self.context.config.xai.base_url

        return base_url

    async def _is_tool_stop_reason(self, finish_reason: str) -> bool:
        # grok uses Null as the finish reason for tool calls?
        return await super()._is_tool_stop_reason(finish_reason) or finish_reason is None
