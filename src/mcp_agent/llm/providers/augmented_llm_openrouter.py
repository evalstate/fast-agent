import os

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# No single default model for OpenRouter, users must specify full path
DEFAULT_OPENROUTER_MODEL = None 


class OpenRouterAugmentedLLM(OpenAIAugmentedLLM):
    """Augmented LLM provider for OpenRouter, using an OpenAI-compatible API."""
    def __init__(self, *args, **kwargs) -> None:
        kwargs["provider_name"] = "OpenRouter"  # Set provider name
        super().__init__(*args, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize OpenRouter-specific default parameters."""
        # OpenRouter model names include the provider, e.g., "google/gemini-flash-1.5"
        # The model should be passed in the 'model' kwarg during factory creation.
        chosen_model = kwargs.get("model", DEFAULT_OPENROUTER_MODEL)
        if not chosen_model:
             # Unlike Deepseek, OpenRouter *requires* a model path in the identifier.
             # The factory should extract this before calling the constructor.
             # We rely on the model being passed correctly via kwargs.
             # If it's still None here, it indicates an issue upstream (factory or user input).
             # However, the base class _get_model handles the error if model is None.
             pass


        return RequestParams(
            model=chosen_model, # Will be validated by base class
            systemPrompt=self.instruction,
            parallel_tool_calls=True, # Default based on OpenAI provider
            max_iterations=10,      # Default based on OpenAI provider
            use_history=True,       # Default based on OpenAI provider
        )

    def _api_key(self) -> str:
        """Retrieve the OpenRouter API key from config or environment variables."""
        config = self.context.config
        api_key = None

        # Check config file first
        if config and hasattr(config, 'openrouter') and config.openrouter:
            api_key = getattr(config.openrouter, 'api_key', None)
            if api_key == "<your-openrouter-api-key-here>" or not api_key:
                api_key = None

        # Fallback to environment variable
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ProviderKeyError(
                "OpenRouter API key not configured",
                "The OpenRouter API key is required but not set.\n"
                "Add it to your configuration file under openrouter.api_key\n"
                "Or set the OPENROUTER_API_KEY environment variable.",
            )
        return api_key

    def _base_url(self) -> str:
        """Retrieve the OpenRouter base URL from config or use the default."""
        base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)  # Default
        config = self.context.config
        
        # Check config file for override
        if config and hasattr(config, 'openrouter') and config.openrouter:
            config_base_url = getattr(config.openrouter, 'base_url', None)
            if config_base_url:
                base_url = config_base_url

        return base_url

    # Other methods like _get_model, _send_request etc., are inherited from OpenAIAugmentedLLM
    # We may override them later if OpenRouter deviates significantly or offers unique features. 