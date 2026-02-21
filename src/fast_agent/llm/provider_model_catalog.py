"""Provider-specific model catalog adapters.

This module keeps provider-specific model discovery logic separate from the
provider-agnostic model selection helpers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_types import Provider


@dataclass(frozen=True)
class ProviderModelInventory:
    """Dynamic model inventory returned by provider-specific adapters."""

    current_models: tuple[str, ...] = ()
    all_models: tuple[str, ...] = ()


class ProviderModelCatalogAdapter(Protocol):
    """Protocol for provider-specific model discovery adapters."""

    provider: Provider

    def discover(self, config: dict[str, Any]) -> ProviderModelInventory:
        """Discover provider models from runtime/config context."""


class OpenRouterModelCatalogAdapter:
    """OpenRouter dynamic model discovery via key-scoped model listing."""

    provider = Provider.OPENROUTER

    def discover(self, config: dict[str, Any]) -> ProviderModelInventory:
        api_key = ProviderKeyManager.get_config_file_key("openrouter", config)
        if not api_key:
            api_key = ProviderKeyManager.get_env_var("openrouter")
        if not api_key:
            return ProviderModelInventory()

        base_url = os.getenv("OPENROUTER_BASE_URL")
        openrouter_cfg = config.get("openrouter")
        if isinstance(openrouter_cfg, dict):
            cfg_base_url = openrouter_cfg.get("base_url")
            if isinstance(cfg_base_url, str) and cfg_base_url.strip():
                base_url = cfg_base_url.strip()

        try:
            from fast_agent.llm.openrouter_model_lookup import list_openrouter_model_specs_sync

            discovered = tuple(list_openrouter_model_specs_sync(api_key=api_key, base_url=base_url))
            return ProviderModelInventory(current_models=discovered, all_models=discovered)
        except Exception:
            return ProviderModelInventory()


class ProviderModelCatalogRegistry:
    """Registry for provider-specific model discovery adapters."""

    _ADAPTERS: dict[Provider, ProviderModelCatalogAdapter] = {
        Provider.OPENROUTER: OpenRouterModelCatalogAdapter(),
    }

    @classmethod
    def discover(cls, provider: Provider, config: dict[str, Any]) -> ProviderModelInventory:
        adapter = cls._ADAPTERS.get(provider)
        if adapter is None:
            return ProviderModelInventory()
        return adapter.discover(config)

