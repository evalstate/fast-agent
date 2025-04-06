import os

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM


# OpenRouter uses an OpenAI compatible API, so we can inherit most functionality
class AugmentedOpenRouterLLM(OpenAIAugmentedLLM):
    """AugmentedLLM implementation for OpenRouter API."""

    def __init__(self, *args, **kwargs) -> None:
        # Ensure the provider name is set correctly for logging/identification
        super().__init__(provider_name="OpenRouter", *args, **kwargs)

    def _api_key(self) -> str:
        """Retrieve OpenRouter API key from config or environment variables."""
        config = self.context.config
        api_key = None

        # Check config first (assuming similar structure to openai config)
        if hasattr(config, "openrouter") and config.openrouter:
            api_key = config.openrouter.api_key
            if api_key == "<your-api-key-here>" or not api_key:
                api_key = None

        # Fallback to environment variable
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ProviderKeyError(
                "OpenRouter API key not configured",
                "The OpenRouter API key is required but not set.\\n"
                "Add it to your configuration file under openrouter.api_key\\n"
                "Or set the OPENROUTER_API_KEY environment variable",
            )
        return api_key

    def _base_url(self) -> str:
        """Return the base URL for the OpenRouter API."""
        # Check config first (assuming similar structure to openai config)
        if (
            hasattr(self.context.config, "openrouter")
            and self.context.config.openrouter
            and self.context.config.openrouter.base_url
        ):
            return self.context.config.openrouter.base_url

        # Default OpenRouter base URL
        return "https://openrouter.ai/api/v1"

    # Inherits generate_internal, generate_str, structured, etc. from OpenAIAugmentedLLM
    # The model name passed during initialization (e.g., "google/gemini-flash")
    # will be used in the API calls made by the inherited methods. 