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
        chosen_model = self._resolve_default_model_name(kwargs.get("model"), DEFAULT_XAI_MODEL)
        resolved_kwargs = dict(kwargs)
        if chosen_model is not None:
            resolved_kwargs["model"] = chosen_model

        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(resolved_kwargs)

        # Override with xAI-specific settings
        base_params.model = chosen_model
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
