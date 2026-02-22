import os

from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_XAI_MODEL = "grok-4-1-fast-reasoning"


class XAILLM(OpenAILLM):
    def __init__(self, **kwargs) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.XAI, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize xAI parameters"""
        base_params = self._initialize_default_params_with_model_fallback(kwargs, DEFAULT_XAI_MODEL)
        base_params.parallel_tool_calls = False

        return base_params

    def _base_url(self) -> str | None:
        base_url: str | None = os.getenv("XAI_BASE_URL", XAI_BASE_URL)
        if self.context.config and self.context.config.xai:
            base_url = self.context.config.xai.base_url

        return base_url

    async def _is_tool_stop_reason(self, finish_reason: str) -> bool:
        # grok uses Null as the finish reason for tool calls?
        return await super()._is_tool_stop_reason(finish_reason) or finish_reason is None
