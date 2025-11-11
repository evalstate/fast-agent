"""
Provider API key management for various LLM providers.
Centralizes API key handling logic to make provider implementations more generic.
"""

import os
from typing import Any, Dict

from pydantic import BaseModel

from fast_agent.core.exceptions import ProviderKeyError

PROVIDER_ENVIRONMENT_MAP: Dict[str, str] = {
    # default behaviour in _get_env_key_name is to capitalize the
    # provider name and suffix "_API_KEY" - so no specific mapping needed unless overriding
    "hf": "HF_TOKEN",
    "responses": "OPENAI_API_KEY",  # Temporary workaround
}
PROVIDER_CONFIG_KEY_ALIASES: Dict[str, tuple[str, ...]] = {
    # HuggingFace historically used "huggingface" (full name) in config files,
    # while the provider id is "hf". Support both spellings.
    "hf": ("hf", "huggingface"),
    "huggingface": ("huggingface", "hf"),
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
        provider_name = provider_name.lower()
        provider_keys = ProviderKeyManager._get_provider_config_keys(provider_name)
        for key in provider_keys:
            provider_settings = config.get(key)
            if not provider_settings:
                continue
            api_key = provider_settings.get("api_key", API_KEY_HINT_TEXT)
            if api_key == API_KEY_HINT_TEXT:
                api_key = None
            break

        return api_key

    @staticmethod
    def _get_provider_config_keys(provider_name: str) -> list[str]:
        """Return config key candidates for a provider (provider id + aliases)."""
        keys = [provider_name]
        for alias in PROVIDER_CONFIG_KEY_ALIASES.get(provider_name, ()):
            if alias not in keys:
                keys.append(alias)
        return keys

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

        from fast_agent.llm.provider_types import Provider

        provider_name = provider_name.lower()

        # Fast-agent provider doesn't need external API keys
        if provider_name == "fast-agent":
            return ""

        api_key = ProviderKeyManager.get_config_file_key(provider_name, config)
        if not api_key:
            api_key = ProviderKeyManager.get_env_var(provider_name)

        if not api_key and provider_name == "generic":
            api_key = "ollama"  # Default for generic provider

        if not api_key:
            # Get proper display name for error message
            try:
                provider_enum = Provider(provider_name)
                display_name = provider_enum.display_name
            except ValueError:
                # Invalid provider name
                raise ProviderKeyError(
                    f"Invalid provider: {provider_name}",
                    f"'{provider_name}' is not a valid provider name.",
                )

            raise ProviderKeyError(
                f"{display_name} API key not configured",
                f"The {display_name} API key is required but not set.\n"
                f"Add it to your configuration file under {provider_name}.api_key "
                f"or set the {ProviderKeyManager.get_env_key_name(provider_name)} environment variable.",
            )

        return api_key
