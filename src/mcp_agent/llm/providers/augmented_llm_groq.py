from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = ""


class GroqAugmentedLLM(OpenAIAugmentedLLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.GROQ, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Groq default parameters"""
        chosen_model = kwargs.get("model", DEFAULT_GROQ_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=False,
            max_iterations=20,
            use_history=True,
        )

    def _base_url(self) -> str:
        base_url = None
        if self.context.config and self.context.config.groq:
            base_url = self.context.config.groq.base_url

        return base_url if base_url else GROQ_BASE_URL
