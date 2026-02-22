import os

from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.openrouter_model_lookup import discover_openrouter_models_sync
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider_types import Provider
from fast_agent.types import RequestParams

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# No single default model for OpenRouter, users must specify full path
DEFAULT_OPENROUTER_MODEL = None


class OpenRouterLLM(OpenAILLM):
    """Augmented LLM provider for OpenRouter, using an OpenAI-compatible API."""

    def __init__(self, **kwargs) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=Provider.OPENROUTER, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize OpenRouter-specific default parameters."""
        chosen_model = self._resolve_default_model_name(
            kwargs.get("model"),
            DEFAULT_OPENROUTER_MODEL,
        )
        self._ensure_model_metadata(chosen_model)

        resolved_kwargs = dict(kwargs)
        if chosen_model is not None:
            resolved_kwargs["model"] = chosen_model

        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(resolved_kwargs)

        # Override with OpenRouter-specific settings
        # OpenRouter model names include the provider, e.g., "google/gemini-flash-1.5"
        # The model should be passed in the 'model' kwarg during factory creation.
        if chosen_model:
            base_params.model = chosen_model
        # If it's still None here, it indicates an issue upstream (factory or user input).
        # However, the base class _get_model handles the error if model is None.

        return base_params

    def _ensure_model_metadata(self, model_name: str | None) -> None:
        """Populate runtime model metadata for OpenRouter-discovered models."""
        if not model_name:
            return

        if ModelDatabase.get_model_params(model_name) is not None:
            return

        try:
            api_key = self._api_key()
        except ProviderKeyError:
            return

        try:
            discover_openrouter_models_sync(api_key=api_key, base_url=self._base_url())
        except Exception:
            # Non-fatal: unknown models fall back to conservative defaults.
            return

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
