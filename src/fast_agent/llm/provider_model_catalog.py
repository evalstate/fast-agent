"""Provider-specific model catalog adapters.

This module keeps provider-specific model discovery logic separate from the
provider-agnostic model selection helpers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_types import Provider
from fast_agent.utils.text import strip_str_to_none


@dataclass(frozen=True, slots=True)
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
            normalized_base_url = strip_str_to_none(cfg_base_url)
            if normalized_base_url is not None:
                base_url = normalized_base_url

        try:
            from fast_agent.llm.openrouter_model_lookup import list_openrouter_model_specs_sync

            discovered = tuple(list_openrouter_model_specs_sync(api_key=api_key, base_url=base_url))
            return ProviderModelInventory(current_models=discovered, all_models=discovered)
        except Exception:
            return ProviderModelInventory()


class LiteLLMModelCatalogAdapter:
    """LiteLLM model discovery via the SDK's bundled `models_by_provider` registry.

    Returns every model LiteLLM knows about, prefixed with `litellm.` and the
    underlying provider key (e.g. `litellm.anthropic/claude-3-5-sonnet`,
    `litellm.bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0`). The same set
    is returned for both `current_models` and `all_models` since LiteLLM does
    not distinguish a "current" subset.
    """

    provider = Provider.LITELLM

    def discover(self, config: dict[str, Any]) -> ProviderModelInventory:  # noqa: ARG002
        try:
            import litellm
        except ImportError:
            return ProviderModelInventory()

        specs: list[str] = []
        seen: set[str] = set()
        models_by_provider = getattr(litellm, "models_by_provider", {})
        for backing_provider, models in models_by_provider.items():
            prefix = f"{backing_provider}/"
            for model in models:
                # Some LiteLLM model strings already include the provider prefix
                # (e.g. `gemini/gemini-exp-1206` listed under the `gemini` key).
                # Strip it so the spec stays single-prefixed.
                model_id = model[len(prefix):] if model.startswith(prefix) else model
                spec = f"litellm.{backing_provider}/{model_id}"
                if spec in seen:
                    continue
                seen.add(spec)
                specs.append(spec)

        if not specs:
            return ProviderModelInventory()

        specs_tuple = tuple(specs)
        return ProviderModelInventory(current_models=specs_tuple, all_models=specs_tuple)


class ProviderModelCatalogRegistry:
    """Registry for provider-specific model discovery adapters."""

    _ADAPTERS: ClassVar[dict[Provider, ProviderModelCatalogAdapter]] = {
        Provider.OPENROUTER: OpenRouterModelCatalogAdapter(),
        Provider.LITELLM: LiteLLMModelCatalogAdapter(),
    }

    @classmethod
    def discover(cls, provider: Provider, config: dict[str, Any]) -> ProviderModelInventory:
        adapter = cls._ADAPTERS.get(provider)
        if adapter is None:
            return ProviderModelInventory()
        return adapter.discover(config)
