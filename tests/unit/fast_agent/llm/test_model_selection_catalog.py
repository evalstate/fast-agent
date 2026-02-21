from __future__ import annotations

import os

from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider_types import Provider


def test_list_curated_models_for_provider() -> None:
    models = ModelSelectionCatalog.list_curated_models(Provider.ANTHROPIC)
    assert "claude-haiku-4-5" in models
    assert "claude-sonnet-4-6" in models
    assert "claude-opus-4-6" in models


def test_list_curated_aliases_for_provider() -> None:
    aliases = ModelSelectionCatalog.list_curated_aliases(Provider.ANTHROPIC)
    assert aliases == ["sonnet", "haiku", "opus"]


def test_legacy_aliases_are_listed_but_not_curated() -> None:
    curated_aliases = ModelSelectionCatalog.list_curated_aliases(Provider.HUGGINGFACE)
    legacy_aliases = ModelSelectionCatalog.list_non_current_aliases(Provider.HUGGINGFACE)

    assert "glm47" in legacy_aliases
    assert "glm47" not in curated_aliases


def test_list_fast_models_uses_explicit_curated_designation() -> None:
    anthropic_fast = ModelSelectionCatalog.list_fast_models(Provider.ANTHROPIC)
    assert anthropic_fast == ["claude-haiku-4-5"]

    codex_fast = ModelSelectionCatalog.list_fast_models(Provider.CODEX_RESPONSES)
    assert codex_fast == ["codexresponses.gpt-5.3-codex-spark?transport=ws"]

    hf_fast = ModelSelectionCatalog.list_fast_models(Provider.HUGGINGFACE)
    assert "hf.openai/gpt-oss-120b:cerebras" in hf_fast
    assert "hf.moonshotai/Kimi-K2.5:fireworks-ai" in hf_fast


def test_is_fast_model_normalizes_provider_prefix() -> None:
    assert ModelSelectionCatalog.is_fast_model("openai.gpt-4.1-mini")
    assert ModelSelectionCatalog.is_fast_model("gpt-4.1-mini")
    assert not ModelSelectionCatalog.is_fast_model("gpt-5")


def test_configured_providers_reads_config_keys() -> None:
    providers = ModelSelectionCatalog.configured_providers(
        {
            "anthropic": {"api_key": "sk-ant"},
            "openai": {"api_key": "sk-openai"},
        }
    )

    assert Provider.ANTHROPIC in providers
    assert Provider.OPENAI in providers
    assert Provider.RESPONSES in providers


def test_configured_providers_reads_environment_keys() -> None:
    original = os.environ.get("OPENAI_API_KEY")

    try:
        os.environ["OPENAI_API_KEY"] = "sk-openai-env"
        providers = ModelSelectionCatalog.configured_providers({})
    finally:
        if original is not None:
            os.environ["OPENAI_API_KEY"] = original
        else:
            os.environ.pop("OPENAI_API_KEY", None)

    assert Provider.OPENAI in providers
    assert Provider.RESPONSES in providers


def test_suggestions_for_providers_returns_curated_and_fast_models() -> None:
    suggestions = ModelSelectionCatalog.suggestions_for_providers([Provider.GOOGLE])

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.provider == Provider.GOOGLE
    assert suggestion.current_aliases
    assert suggestion.non_current_aliases == ()
    assert suggestion.current_models
    assert suggestion.fast_models
    assert suggestion.all_models


def test_suggestions_include_legacy_aliases_when_configured() -> None:
    suggestions = ModelSelectionCatalog.suggestions_for_providers([Provider.HUGGINGFACE])

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert "glm47" in suggestion.non_current_aliases
    assert "glm47" not in suggestion.current_aliases


def test_list_all_models_for_provider() -> None:
    openai_models = ModelSelectionCatalog.list_all_models(Provider.OPENAI)
    assert "gpt-4.1" in openai_models
    assert "claude-sonnet-4-6" not in openai_models


def test_codexresponses_curated_entries_use_explicit_transports() -> None:
    curated = ModelSelectionCatalog.list_curated_models(Provider.CODEX_RESPONSES)
    assert "codexresponses.gpt-5.3-codex?transport=ws&reasoning=high" in curated
    assert "codexresponses.gpt-5.3-codex-spark?transport=ws" in curated
