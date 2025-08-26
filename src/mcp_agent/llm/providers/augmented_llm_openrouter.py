import os

from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# No single default model for OpenRouter, users must specify full path
DEFAULT_OPENROUTER_MODEL = None


class OpenRouterAugmentedLLM(OpenAIAugmentedLLM):
    """Augmented LLM provider for OpenRouter, using an OpenAI-compatible API."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.OPENROUTER, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize OpenRouter-specific default parameters."""
        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(kwargs)
        
        # Override with OpenRouter-specific settings
        # OpenRouter model names include the provider, e.g., "google/gemini-flash-1.5"
        # The model should be passed in the 'model' kwarg during factory creation.
        chosen_model = kwargs.get("model", DEFAULT_OPENROUTER_MODEL)
        if chosen_model:
            base_params.model = chosen_model
        # If it's still None here, it indicates an issue upstream (factory or user input).
        # However, the base class _get_model handles the error if model is None.
        
        return base_params

    def _base_url(self) -> str:
        """Retrieve the OpenRouter base URL from config or use the default."""
        base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)  # Default
        config = self.context.config

        # Check config file for override
        if config and hasattr(config, "openrouter") and config.openrouter:
            config_base_url = getattr(config.openrouter, "base_url", None)
            if config_base_url:
                base_url = config_base_url

        return base_url
