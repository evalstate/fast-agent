"""Model selection helpers for current, listed, and fast model recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from fast_agent.llm.model_aliases import BUILTIN_MODEL_ALIASES
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_overlays import ModelOverlayRegistry, load_model_overlay_registry
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_model_catalog import ProviderModelCatalogRegistry
from fast_agent.llm.provider_types import Provider
from fast_agent.utils.collections import unique_preserve_order

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class CatalogModelEntry:
    """An explicit model catalog entry for a provider preset token."""

    alias: str
    model: str
    current: bool = True
    fast: bool = False
    local: bool = False
    display_label: str | None = None
    description: str | None = None


def _builtin_entry(
    alias: str,
    *,
    current: bool = True,
    fast: bool = False,
    display_label: str | None = None,
    description: str | None = None,
) -> CatalogModelEntry:
    """Create picker metadata for a canonical built-in alias."""
    return CatalogModelEntry(
        alias=alias,
        model=BUILTIN_MODEL_ALIASES[alias],
        current=current,
        fast=fast,
        display_label=display_label,
        description=description,
    )


class ModelSelectionCatalog:
    """Catalog of current/listed and fast model preset tokens."""

    CATALOG_ENTRIES_BY_PROVIDER: ClassVar[dict[Provider, tuple[CatalogModelEntry, ...]]] = {
        Provider.RESPONSES: (
            _builtin_entry("gpt-5.6-sol"),
            _builtin_entry("gpt-5.6-terra", fast=True),
            _builtin_entry("gpt-5.6-luna", fast=True),
            _builtin_entry("chat-latest"),
            _builtin_entry("gpt-5.5"),
            _builtin_entry("gpt-5.4"),
            _builtin_entry("gpt-5.4-mini", fast=True),
            _builtin_entry("gpt-5.4-nano", fast=True),
            _builtin_entry("gpt-5.3-codex"),
            _builtin_entry("gpt-5.2"),
        ),
        Provider.OPENAI: (
            _builtin_entry("gpt-4.1"),
            _builtin_entry("gpt-4o"),
            _builtin_entry("gpt-4.1-mini", fast=True),
            _builtin_entry("gpt-4.1-nano", fast=True),
        ),
        Provider.ANTHROPIC: (
            _builtin_entry("fable"),
            _builtin_entry("opus"),
            _builtin_entry("sonnet"),
            _builtin_entry("opus48"),
            _builtin_entry("opus46"),
            _builtin_entry("haiku", fast=True),
        ),
        Provider.ANTHROPIC_VERTEX: (
            CatalogModelEntry(alias="opus", model="anthropic-vertex.claude-opus-4-7"),
            CatalogModelEntry(alias="opus46", model="anthropic-vertex.claude-opus-4-6"),
            CatalogModelEntry(alias="sonnet", model="anthropic-vertex.claude-sonnet-5"),
            CatalogModelEntry(
                alias="haiku",
                model="anthropic-vertex.claude-haiku-4-5",
                fast=True,
            ),
        ),
        Provider.GOOGLE: (
            CatalogModelEntry(
                alias="gemini35flash",
                display_label="Gemini 3.5 Flash",
                model="google.gemini-3.5-flash",
                fast=True,
            ),
            CatalogModelEntry(
                alias="gemini3.1",
                display_label="Gemini 3.1 Pro",
                model="google.gemini-3.1-pro-preview",
            ),
            CatalogModelEntry(
                alias="gemini3.1flashlite",
                display_label="Gemini 3.1 Flash Lite",
                model="google.gemini-3.1-flash-lite-preview",
                fast=True,
            ),
            CatalogModelEntry(
                alias="gemini3flash",
                display_label="Gemini 3 Flash",
                model="google.gemini-3-flash-preview",
            ),
        ),
        Provider.XAI: (
            _builtin_entry("Grok 4.6"),
            _builtin_entry("Grok 4.6 (X Search)"),
            _builtin_entry("Grok 4.5"),
            _builtin_entry("Grok 4.5 (X Search)"),
            _builtin_entry("Grok 4.3"),
        ),
        Provider.META_AI: (
            _builtin_entry("Muse Spark 1.2"),
            _builtin_entry("Muse Spark 1.2 (Contributor)"),
            _builtin_entry("Muse Spark 1.1"),
        ),
        Provider.DEEPSEEK: (
            _builtin_entry(
                "deepseek",
                display_label="DeepSeek V4 Flash",
                fast=True,
            ),
            _builtin_entry(
                "DeepSeek V4 Pro",
                display_label="DeepSeek V4 Pro",
            ),
        ),
        Provider.ZAI: (_builtin_entry("zaiglm", display_label="GLM 5.2"),),
        Provider.MOONSHOT: (_builtin_entry("kimik3", display_label="Kimi K3"),),
        Provider.OPENROUTER: (),
        Provider.ALIYUN: (
            _builtin_entry("qwen-turbo", fast=True),
            _builtin_entry("qwen3-max"),
        ),
        Provider.HUGGINGFACE: (
            _builtin_entry(
                "Kimi K3 (fireworks-ai)",
                display_label="Kimi K3 (fireworks-ai)",
                description="image-only HF route",
            ),
            _builtin_entry(
                "Kimi K3 (together)",
                display_label="Kimi K3 (together)",
                description="image-only HF route",
            ),
            _builtin_entry(
                "GLM 5.2 (zai-org)",
                display_label="GLM 5.2 (zai-org)",
            ),
            _builtin_entry(
                "GLM 5.2 (fireworks-ai)",
                display_label="GLM 5.2 (fireworks-ai)",
            ),
            _builtin_entry(
                "GLM 5.2 (deepinfra)",
                display_label="GLM 5.2 (deepinfra)",
            ),
            _builtin_entry(
                "kimi27",
                display_label="Kimi 2.7-Code",
                description="thinking mode",
                fast=True,
            ),
            _builtin_entry("gemma4", display_label="Gemma 4 31B"),
            _builtin_entry("minimax3", display_label="Minimax 3.0"),
            _builtin_entry(
                "DeepSeek V4 Flash 0731 (baseten)",
                display_label="DeepSeek V4 Flash 0731 (baseten)",
            ),
            _builtin_entry(
                "DeepSeek V4 Flash 0731 (deepinfra)",
                display_label="DeepSeek V4 Flash 0731 (deepinfra)",
            ),
            _builtin_entry("deepseek-hf", display_label="DeepSeek V4 Pro (HF)"),
            _builtin_entry(
                "kimi26",
                display_label="Kimi 2.6",
                description="thinking mode",
            ),
            _builtin_entry(
                "kimi26instant",
                display_label="Kimi 2.6 (instant)",
                description="instant mode",
                fast=True,
            ),
            _builtin_entry("glm51", display_label="GLM 5.1"),
            _builtin_entry(
                "minimax27",
                display_label="Minimax 2.7",
                current=False,
            ),
            _builtin_entry(
                "qwen35",
                display_label="Qwen 3.5-397B-A17B",
            ),
            _builtin_entry(
                "qwen35instruct",
                display_label="Qwen 3.5-397B-A17B (instruct)",
            ),
            _builtin_entry(
                "qwen36",
                display_label="Qwen 3.6 35B-A3B",
            ),
            _builtin_entry(
                "qwen36instruct",
                display_label="Qwen 3.6 35B-A3B (instruct)",
            ),
            _builtin_entry(
                "minimax25",
                display_label="Minimax 2.5",
                current=False,
            ),
            _builtin_entry(
                "kimi25",
                display_label="Kimi 2.5",
                fast=True,
                current=False,
            ),
            _builtin_entry(
                "kimi25instant",
                display_label="Kimi 2.5 (instant)",
                fast=True,
                current=False,
            ),
            _builtin_entry("glm5", current=False),
            _builtin_entry("gpt-oss", fast=True),
            _builtin_entry("glm47", current=False),
            _builtin_entry("gpt-oss-20b"),
            #            CatalogModelEntry(alias="deepseek31", model="hf.deepseek-ai/DeepSeek-V3.1"),
            _builtin_entry("deepseek32", current=False),
        ),
        Provider.CODEX_RESPONSES: (
            _builtin_entry("sol"),
            _builtin_entry("terra"),
            _builtin_entry("luna"),
            _builtin_entry("codexplan"),
            _builtin_entry("codexplan55"),
            _builtin_entry("codexplan54"),
            _builtin_entry("codexplan53"),
            _builtin_entry("codexspark", fast=True),
            CatalogModelEntry(
                alias="gpt-5.4-mini",
                model="codexresponses.gpt-5.4-mini?reasoning=medium",
                fast=True,
            ),
        ),
        Provider.GROQ: (
            _builtin_entry("qwen3.6-27b", fast=True),
            _builtin_entry("qwen3-32b", fast=True),
        ),
        Provider.FAST_AGENT: (
            _builtin_entry("passthrough"),
            _builtin_entry("playback"),
        ),
    }

    @staticmethod
    def _resolve_overlay_registry(
        overlay_registry: ModelOverlayRegistry | None = None,
        *,
        start_path: Path | None = None,
        home: str | Path | None = None,
    ) -> ModelOverlayRegistry:
        if overlay_registry is not None:
            return overlay_registry
        return load_model_overlay_registry(start_path=start_path, home=home)

    @classmethod
    def _entries_by_provider(
        cls,
        overlay_registry: ModelOverlayRegistry | None = None,
        *,
        start_path: Path | None = None,
        home: str | Path | None = None,
    ) -> dict[Provider, tuple[CatalogModelEntry, ...]]:
        provider_map = {
            provider: list(entries) for provider, entries in cls.CATALOG_ENTRIES_BY_PROVIDER.items()
        }
        overlay_registry = cls._resolve_overlay_registry(
            overlay_registry,
            start_path=start_path,
            home=home,
        )
        overlay_entries_by_provider: dict[Provider, list[CatalogModelEntry]] = {}
        overlay_aliases_by_provider: dict[Provider, set[str]] = {}

        for overlay in overlay_registry.overlays:
            overlay_aliases_by_provider.setdefault(overlay.provider, set()).add(overlay.name)
            overlay_entries_by_provider.setdefault(overlay.provider, []).append(
                CatalogModelEntry(
                    alias=overlay.name,
                    model=overlay.compiled_model_spec,
                    current=overlay.current,
                    fast=overlay.fast,
                    local=True,
                    display_label=overlay.display_label,
                    description=overlay.description,
                )
            )

        merged: dict[Provider, tuple[CatalogModelEntry, ...]] = {}
        ordered_providers = list(provider_map.keys())
        ordered_providers.extend(
            provider for provider in overlay_entries_by_provider if provider not in provider_map
        )

        for provider in ordered_providers:
            overlay_entries = overlay_entries_by_provider.get(provider, [])
            overlay_aliases = overlay_aliases_by_provider.get(provider, set())
            static_entries = [
                entry
                for entry in provider_map.get(provider, [])
                if entry.alias not in overlay_aliases
            ]
            merged[provider] = (*overlay_entries, *static_entries)
        return merged

    @classmethod
    def list_entries(
        cls,
        provider: Provider | None = None,
        *,
        current: bool | None = None,
        overlay_registry: ModelOverlayRegistry | None = None,
        start_path: Path | None = None,
        home: str | Path | None = None,
    ) -> list[CatalogModelEntry]:
        """Return catalog entries, optionally filtered by provider and current flag."""
        provider_map = cls._entries_by_provider(
            overlay_registry=overlay_registry,
            start_path=start_path,
            home=home,
        )
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
    def list_current_entries(
        cls,
        provider: Provider | None = None,
        *,
        overlay_registry: ModelOverlayRegistry | None = None,
        start_path: Path | None = None,
        home: str | Path | None = None,
    ) -> list[CatalogModelEntry]:
        """Return current entries for one provider, or all providers."""
        return cls.list_entries(
            provider=provider,
            current=True,
            overlay_registry=overlay_registry,
            start_path=start_path,
            home=home,
        )

    @classmethod
    def list_non_current_entries(
        cls,
        provider: Provider | None = None,
        *,
        overlay_registry: ModelOverlayRegistry | None = None,
        start_path: Path | None = None,
        home: str | Path | None = None,
    ) -> list[CatalogModelEntry]:
        """Return listed but non-current entries for one provider, or all providers."""
        return cls.list_entries(
            provider=provider,
            current=False,
            overlay_registry=overlay_registry,
            start_path=start_path,
            home=home,
        )

    @classmethod
    def list_current_models(
        cls,
        provider: Provider | None = None,
        *,
        overlay_registry: ModelOverlayRegistry | None = None,
        start_path: Path | None = None,
        home: str | Path | None = None,
    ) -> list[str]:
        """Return current models for one provider, or all providers."""
        entries = cls.list_current_entries(
            provider,
            overlay_registry=overlay_registry,
            start_path=start_path,
            home=home,
        )
        return unique_preserve_order(entry.model for entry in entries)

    @classmethod
    def list_current_aliases(
        cls,
        provider: Provider | None = None,
        *,
        overlay_registry: ModelOverlayRegistry | None = None,
        start_path: Path | None = None,
        home: str | Path | None = None,
    ) -> list[str]:
        """Return current aliases for one provider, or all providers."""
        entries = cls.list_current_entries(
            provider,
            overlay_registry=overlay_registry,
            start_path=start_path,
            home=home,
        )
        return unique_preserve_order(entry.alias for entry in entries)

    @classmethod
    def list_non_current_aliases(
        cls,
        provider: Provider | None = None,
        *,
        overlay_registry: ModelOverlayRegistry | None = None,
        start_path: Path | None = None,
        home: str | Path | None = None,
    ) -> list[str]:
        """Return listed aliases that are intentionally not current."""
        entries = cls.list_non_current_entries(
            provider,
            overlay_registry=overlay_registry,
            start_path=start_path,
            home=home,
        )
        return unique_preserve_order(entry.alias for entry in entries)

    @classmethod
    def list_fast_models(
        cls,
        provider: Provider | None = None,
        *,
        overlay_registry: ModelOverlayRegistry | None = None,
        start_path: Path | None = None,
        home: str | Path | None = None,
    ) -> list[str]:
        """Return explicit fast models from current catalog entries."""
        entries = cls.list_current_entries(
            provider,
            overlay_registry=overlay_registry,
            start_path=start_path,
            home=home,
        )
        return unique_preserve_order(entry.model for entry in entries if entry.fast)

    @classmethod
    def list_all_models(
        cls,
        provider: Provider | None = None,
        config: Any | None = None,
        *,
        overlay_registry: ModelOverlayRegistry | None = None,
        start_path: Path | None = None,
        home: str | Path | None = None,
    ) -> list[str]:
        """Return all known models, optionally constrained to one provider."""
        config_payload = cls._as_mapping(config)
        if provider is None:
            return ModelDatabase.list_models()

        static_models = cls._list_static_models_for_provider(
            provider,
            overlay_registry=overlay_registry,
            start_path=start_path,
            home=home,
        )
        discovered = ProviderModelCatalogRegistry.discover(provider, config_payload)
        if not discovered.all_models:
            return static_models

        return unique_preserve_order([*static_models, *discovered.all_models])

    @classmethod
    def is_fast_model(cls, model: str) -> bool:
        """Return True when the provided model spec belongs to the fast catalog."""
        return ModelDatabase.is_fast_model(model)

    @classmethod
    def configured_providers(
        cls,
        config: Any | None = None,
        *,
        overlay_registry: ModelOverlayRegistry | None = None,
        start_path: Path | None = None,
        home: str | Path | None = None,
    ) -> list[Provider]:
        """Detect providers with configured credentials via config and environment."""
        config_payload = cls._as_mapping(config)

        providers: list[Provider] = []
        for provider in cls._entries_by_provider(
            overlay_registry=overlay_registry,
            start_path=start_path,
            home=home,
        ):
            provider_name = provider.config_name

            if provider == Provider.ANTHROPIC_VERTEX:
                from fast_agent.llm.provider.anthropic.vertex_config import anthropic_vertex_ready

                ready, _ = anthropic_vertex_ready(config_payload)
                if ready:
                    providers.append(provider)
                continue

            # Google Vertex can run without an API key.
            if provider == Provider.GOOGLE and cls._google_vertex_enabled(config_payload):
                providers.append(provider)
                continue

            config_key = ProviderKeyManager.get_config_file_key(provider_name, config_payload)
            env_key = ProviderKeyManager.get_env_var(provider_name)
            if config_key or env_key:
                providers.append(provider)
                continue

            # OAuth / external credential stores are valid runtime sources for
            # providers that intentionally support them. Keep this narrow so
            # fallbacks like generic/ollama or optional HF hub tokens do not
            # mark every local provider "configured".
            if provider in {Provider.CODEX_RESPONSES, Provider.XAI}:
                if ProviderKeyManager._provider_specific_fallback_key(provider_name):
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
    def _list_static_models_for_provider(
        provider: Provider,
        *,
        overlay_registry: ModelOverlayRegistry | None = None,
        start_path: Path | None = None,
        home: str | Path | None = None,
    ) -> list[str]:
        overlay_models = [
            overlay.compiled_model_spec
            for overlay in ModelSelectionCatalog._resolve_overlay_registry(
                overlay_registry,
                start_path=start_path,
                home=home,
            ).entries_for_provider(provider)
        ]
        models = ModelDatabase.list_models()
        if provider == Provider.ANTHROPIC_VERTEX:
            static_models = [
                f"{provider.config_name}.{model}"
                for model in models
                if ModelDatabase.get_default_provider(model) == Provider.ANTHROPIC
            ]
            return unique_preserve_order([*overlay_models, *static_models])
        static_models = [
            model for model in models if ModelDatabase.get_default_provider(model) == provider
        ]
        return unique_preserve_order([*overlay_models, *static_models])
