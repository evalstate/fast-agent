import os

from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
DEFAULT_OLLAMA_API_KEY = "ollama"


class GenericAugmentedLLM(OpenAIAugmentedLLM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args, provider=Provider.GENERIC, **kwargs
        )  # Properly pass args and kwargs to parent

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Generic  parameters"""
        chosen_model = kwargs.get("model", DEFAULT_OLLAMA_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _base_url(self) -> str:
        base_url = os.getenv("GENERIC_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        if self.context.config and self.context.config.generic:
            base_url = self.context.config.generic.base_url

        return base_url
