from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash"


class GoogleOaiAugmentedLLM(OpenAIAugmentedLLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.GOOGLE_OAI, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Google OpenAI Compatibility default parameters"""
        chosen_model = kwargs.get("model", DEFAULT_GOOGLE_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=False,
            max_iterations=10,
            use_history=True,
        )

    def _base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.google:
            base_url = self.context.config.google.base_url

        return base_url if base_url else GOOGLE_BASE_URL
