"""
Provider API key management for various LLM providers.
Centralizes API key handling logic to make provider implementations more generic.
"""

import os
from typing import Any, Dict

from pydantic import BaseModel

from mcp_agent.core.exceptions import ProviderKeyError

PROVIDER_ENVIRONMENT_MAP: Dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "generic": "GENERIC_API_KEY",
    "huggingface": "HF_TOKEN",
}
API_KEY_HINT_TEXT = "<your-api-key-here>"


class ProviderKeyManager:
    """
    Manages API keys for different providers centrally.
    This class abstracts away the provider-specific key access logic,
    making the provider implementations more generic.
    """

    @staticmethod
    def get_env_var(provider_name: str) -> str | None:
        return os.getenv(ProviderKeyManager.get_env_key_name(provider_name))

    @staticmethod
    def get_env_key_name(provider_name: str) -> str:
        return PROVIDER_ENVIRONMENT_MAP.get(provider_name, f"{provider_name.upper()}_API_KEY")

    @staticmethod
    def get_config_file_key(provider_name: str, config: Any) -> str | None:
        api_key = None
        if isinstance(config, BaseModel):
            config = config.model_dump()
        provider_settings = config.get(provider_name)
        if provider_settings:
            api_key = provider_settings.get("api_key", API_KEY_HINT_TEXT)
            if api_key == API_KEY_HINT_TEXT:
                api_key = None

        return api_key

    @staticmethod
    def get_api_key(provider_name: str, config: Any) -> str:
        """
        Gets the API key for the specified provider.

        Args:
            provider_name: Name of the provider (e.g., "anthropic", "openai")
            config: The application configuration object

        Returns:
            The API key as a string

        Raises:
            ProviderKeyError: If the API key is not found or is invalid
        """

        provider_name = provider_name.lower()
        api_key = ProviderKeyManager.get_config_file_key(provider_name, config)
        if not api_key:
            api_key = ProviderKeyManager.get_env_var(provider_name)

        if not api_key and provider_name == "generic":
            api_key = "ollama"  # Default for generic provider

        if not api_key:
            raise ProviderKeyError(
                f"{provider_name.title()} API key not configured",
                f"The {provider_name.title()} API key is required but not set.\n"
                f"Add it to your configuration file under {provider_name}.api_key "
                f"or set the {ProviderKeyManager.get_env_key_name(provider_name)} environment variable.",
            )

        return api_key
