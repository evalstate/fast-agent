"""Model selection helpers for current, listed, and fast model recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pydantic import BaseModel

from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_model_catalog import ProviderModelCatalogRegistry
from fast_agent.llm.provider_types import Provider


@dataclass(frozen=True)
class ProviderModelSuggestions:
    """Current/listed and fast model suggestions for a provider."""

    provider: Provider
    current_models: tuple[str, ...]
    current_aliases: tuple[str, ...]
    non_current_aliases: tuple[str, ...]
    fast_models: tuple[str, ...]
    all_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogModelEntry:
    """An explicit model catalog entry for a provider."""

    alias: str
    model: str
    current: bool = True
    fast: bool = False


class ModelSelectionCatalog:
    """Catalog of current/listed and fast model aliases."""

    CATALOG_ENTRIES_BY_PROVIDER: dict[Provider, tuple[CatalogModelEntry, ...]] = {
        Provider.RESPONSES: (
            CatalogModelEntry(
                alias="gpt-5-mini",
                model="responses.gpt-5-mini?reasoning=low",
                fast=True,
            ),
            CatalogModelEntry(alias="gpt-5.2", model="responses.gpt-5.2?reasoning=medium"),
        ),
        Provider.OPENAI: (
            CatalogModelEntry(alias="gpt-4.1-mini", model="openai.gpt-4.1-mini", fast=True),
            CatalogModelEntry(alias="gpt-4.1", model="openai.gpt-4.1"),
        ),
        Provider.ANTHROPIC: (
            CatalogModelEntry(alias="sonnet", model="claude-sonnet-4-6"),
            CatalogModelEntry(alias="haiku", model="claude-haiku-4-5", fast=True),
            CatalogModelEntry(alias="opus", model="claude-opus-4-6"),
        ),
        Provider.GOOGLE: (
            CatalogModelEntry(
                alias="gemini3-flash",
                model="google.gemini-3-flash-preview",
                fast=True,
            ),
            CatalogModelEntry(alias="gemini3", model="google.gemini-3-pro-preview"),
            CatalogModelEntry(alias="gemini3.1", model="google.gemini-3.1-pro-preview"),
        ),
        Provider.XAI: (
            CatalogModelEntry(alias="grok41fast", model="grok-4-1-fast-non-reasoning", fast=True),
            CatalogModelEntry(alias="grok4", model="xai.grok-4"),
        ),
        Provider.DEEPSEEK: (
            CatalogModelEntry(alias="deepseek", model="deepseek.deepseek-chat", fast=True),
        ),
        Provider.OPENROUTER: (),
        Provider.ALIYUN: (
            CatalogModelEntry(alias="qwen-turbo", model="aliyun.qwen-turbo", fast=True),
            CatalogModelEntry(alias="qwen3-max", model="aliyun.qwen3-max"),
        ),
        Provider.HUGGINGFACE: (
            CatalogModelEntry(alias="kimi", model="hf.moonshotai/Kimi-K2-Instruct-0905:groq"),
            CatalogModelEntry(alias="minimax", model="hf.MiniMaxAI/MiniMax-M2.1:novita"),
            CatalogModelEntry(alias="gpt-oss", model="hf.openai/gpt-oss-120b:cerebras", fast=True),
            CatalogModelEntry(alias="gpt-oss-20b", model="hf.openai/gpt-oss-20b"),
            CatalogModelEntry(alias="glm5", model="hf.zai-org/GLM-5:novita"),
            CatalogModelEntry(alias="glm", model="hf.zai-org/GLM-5:novita"),
            CatalogModelEntry(alias="qwen3", model="hf.Qwen/Qwen3-Next-80B-A3B-Instruct:together"),
            CatalogModelEntry(
                alias="qwen35",
                model=(
                    "hf.Qwen/Qwen3.5-397B-A17B:novita"
                    "?temperature=0.6&top_p=0.95&top_k=20&min_p=0.0"
                    "&presence_penalty=0.0&repetition_penalty=1.0&reasoning=on"
                ),
            ),
            CatalogModelEntry(
                alias="qwen35instruct",
                model=(
                    "hf.Qwen/Qwen3.5-397B-A17B:novita"
                    "?temperature=0.7&top_p=0.8&top_k=20&min_p=0.0"
                    "&presence_penalty=1.5&repetition_penalty=1.0&reasoning=off"
                ),
            ),
            CatalogModelEntry(alias="deepseek31", model="hf.deepseek-ai/DeepSeek-V3.1"),
            CatalogModelEntry(
                alias="deepseek32",
                model="hf.deepseek-ai/DeepSeek-V3.2:fireworks-ai",
            ),
            CatalogModelEntry(
                alias="kimi25",
                model="hf.moonshotai/Kimi-K2.5:fireworks-ai",
                fast=True,
            ),
            CatalogModelEntry(
                alias="glm47",
                model="hf.zai-org/GLM-4.7:cerebras",
                current=False,
            ),
        ),
        Provider.CODEX_RESPONSES: (
            CatalogModelEntry(
                alias="codexplan",
                model="codexresponses.gpt-5.3-codex?transport=ws&reasoning=high",
            ),
            CatalogModelEntry(
                alias="codexspark",
                model="codexresponses.gpt-5.3-codex-spark?transport=ws",
                fast=True,
            ),
        ),
    }

    @staticmethod
    def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    @classmethod
    def _entries_by_provider(cls) -> dict[Provider, tuple[CatalogModelEntry, ...]]:
        return cls.CATALOG_ENTRIES_BY_PROVIDER

    @classmethod
    def list_entries(
        cls,
        provider: Provider | None = None,
        *,
        current: bool | None = None,
    ) -> list[CatalogModelEntry]:
        """Return catalog entries, optionally filtered by provider and current flag."""
        provider_map = cls._entries_by_provider()
        if provider is not None:
            entries = list(provider_map.get(provider, ()))
            if current is None:
                return entries
            return [entry for entry in entries if entry.current is current]

        entries: list[CatalogModelEntry] = []
        for provider_entries in provider_map.values():
            entries.extend(provider_entries)
        if current is None:
            return entries
        return [entry for entry in entries if entry.current is current]

    @classmethod
    def list_current_entries(cls, provider: Provider | None = None) -> list[CatalogModelEntry]:
        """Return current entries for one provider, or all providers."""
        return cls.list_entries(provider=provider, current=True)

    @classmethod
    def list_non_current_entries(cls, provider: Provider | None = None) -> list[CatalogModelEntry]:
        """Return listed but non-current entries for one provider, or all providers."""
        return cls.list_entries(provider=provider, current=False)

    @classmethod
    def list_current_models(cls, provider: Provider | None = None) -> list[str]:
        """Return current models for one provider, or all providers."""
        entries = cls.list_current_entries(provider)
        return cls._dedupe_preserve_order(entry.model for entry in entries)

    @classmethod
    def list_current_aliases(cls, provider: Provider | None = None) -> list[str]:
        """Return current aliases for one provider, or all providers."""
        entries = cls.list_current_entries(provider)
        return cls._dedupe_preserve_order(entry.alias for entry in entries)

    @classmethod
    def list_non_current_aliases(cls, provider: Provider | None = None) -> list[str]:
        """Return listed aliases that are intentionally not current."""
        entries = cls.list_non_current_entries(provider)
        return cls._dedupe_preserve_order(entry.alias for entry in entries)

    @classmethod
    def list_fast_models(cls, provider: Provider | None = None) -> list[str]:
        """Return explicit fast models from current catalog entries."""
        entries = cls.list_current_entries(provider)
        return cls._dedupe_preserve_order(entry.model for entry in entries if entry.fast)

    # Backward-compatible aliases
    @classmethod
    def list_curated_entries(cls, provider: Provider | None = None) -> list[CatalogModelEntry]:
        """Backward-compatible alias for current entries."""
        return cls.list_current_entries(provider)

    @classmethod
    def list_curated_models(cls, provider: Provider | None = None) -> list[str]:
        """Backward-compatible alias for current models."""
        return cls.list_current_models(provider)

    @classmethod
    def list_curated_aliases(cls, provider: Provider | None = None) -> list[str]:
        """Backward-compatible alias for current aliases."""
        return cls.list_current_aliases(provider)

    @classmethod
    def list_legacy_aliases(cls, provider: Provider | None = None) -> list[str]:
        """Backward-compatible alias for non-current aliases."""
        return cls.list_non_current_aliases(provider)

    @classmethod
    def list_all_models(
        cls, provider: Provider | None = None, config: Any | None = None
    ) -> list[str]:
        """Return all known models, optionally constrained to one provider."""
        config_payload = cls._as_mapping(config)
        if provider is None:
            return ModelDatabase.list_models()

        static_models = cls._list_static_models_for_provider(provider)
        discovered = ProviderModelCatalogRegistry.discover(provider, config_payload)
        if not discovered.all_models:
            return static_models

        return cls._dedupe_preserve_order([*static_models, *discovered.all_models])

    @classmethod
    def is_fast_model(cls, model: str) -> bool:
        """Return True when the provided model spec belongs to the fast catalog."""
        return ModelDatabase.is_fast_model(model)

    @classmethod
    def suggestions_for_providers(
        cls,
        providers: Iterable[Provider],
        *,
        config: Any | None = None,
    ) -> list[ProviderModelSuggestions]:
        """Build provider-specific current, non-current, and fast model suggestions."""
        config_payload = cls._as_mapping(config)
        suggestions: list[ProviderModelSuggestions] = []
        for provider in providers:
            discovered = ProviderModelCatalogRegistry.discover(provider, config_payload)

            current_models = tuple(
                cls._dedupe_preserve_order(
                    [*cls.list_current_models(provider), *discovered.current_models]
                )
            )
            current_aliases = tuple(cls.list_current_aliases(provider))
            non_current_aliases = tuple(cls.list_non_current_aliases(provider))
            fast = tuple(cls.list_fast_models(provider))
            all_models = tuple(
                cls._dedupe_preserve_order(
                    [*cls._list_static_models_for_provider(provider), *discovered.all_models]
                )
            )

            if (
                not current_models
                and not current_aliases
                and not non_current_aliases
                and not fast
                and not all_models
            ):
                continue
            suggestions.append(
                ProviderModelSuggestions(
                    provider=provider,
                    current_models=current_models,
                    current_aliases=current_aliases,
                    non_current_aliases=non_current_aliases,
                    fast_models=fast,
                    all_models=all_models,
                )
            )

        return suggestions

    @classmethod
    def configured_providers(cls, config: Any | None = None) -> list[Provider]:
        """Detect providers with configured credentials via config and environment."""
        config_payload = cls._as_mapping(config)

        providers: list[Provider] = []
        for provider in cls._entries_by_provider():
            provider_name = provider.config_name

            # Google Vertex can run without an API key.
            if provider == Provider.GOOGLE and cls._google_vertex_enabled(config_payload):
                providers.append(provider)
                continue

            config_key = ProviderKeyManager.get_config_file_key(provider_name, config_payload)
            env_key = ProviderKeyManager.get_env_var(provider_name)
            if config_key or env_key:
                providers.append(provider)

        return providers

    @staticmethod
    def _as_mapping(config: Any | None) -> dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, BaseModel):
            dumped = config.model_dump()
            if isinstance(dumped, dict):
                return dumped
            return {}
        if isinstance(config, dict):
            return config
        return {}

    @staticmethod
    def _google_vertex_enabled(config_payload: dict[str, Any]) -> bool:
        google_cfg = config_payload.get("google")
        if not isinstance(google_cfg, dict):
            return False

        vertex_cfg = google_cfg.get("vertex_ai")
        if not isinstance(vertex_cfg, dict):
            return False

        return bool(vertex_cfg.get("enabled"))

    @staticmethod
    def _list_static_models_for_provider(provider: Provider) -> list[str]:
        models = ModelDatabase.list_models()
        return [model for model in models if ModelDatabase.get_default_provider(model) == provider]
