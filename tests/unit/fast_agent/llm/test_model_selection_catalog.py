"""
Testing notes:

- This module owns catalog-level invariants: current vs non-current aliases,
  fast-model flags, provider suggestions, and overlay/discovery interactions.
- Prefer invariant and sentinel tests over reproducing the full curated catalog
  as exact string-for-string assertions.
- A few promoted/legacy smoke tests are useful here when they validate the
  migration state users see in pickers and suggestions.
- Alias parsing semantics belong in test_model_factory.py; model capability
  lookups belong in test_model_database.py.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_overlays import load_model_overlay_registry
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider.anthropic.vertex_config import GoogleAdcStatus
from fast_agent.llm.provider_types import Provider
from fast_agent.utils.collections import unique_preserve_order

if TYPE_CHECKING:
    from pathlib import Path


def _write_overlay(home: "Path", name: str, *, provider: str, model: str) -> None:
    overlays_dir = home / "model-overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    (overlays_dir / f"{name}.yaml").write_text(
        "\n".join(
            [
                f"name: {name}",
                f"provider: {provider}",
                f"model: {model}",
            ]
        ),
        encoding="utf-8",
    )


def _static_current_entries(provider: Provider) -> list:
    return [
        entry
        for entry in ModelSelectionCatalog.CATALOG_ENTRIES_BY_PROVIDER[provider]
        if entry.current
    ]


def test_list_current_models_for_provider() -> None:
    models = ModelSelectionCatalog.list_current_models(Provider.ANTHROPIC)
    expected = unique_preserve_order(
        entry.model for entry in _static_current_entries(Provider.ANTHROPIC)
    )
    assert models == expected


def test_list_current_aliases_for_provider() -> None:
    aliases = ModelSelectionCatalog.list_current_aliases(Provider.ANTHROPIC)

    assert aliases == unique_preserve_order(aliases)
    assert {
        "fable",
        "haiku",
        "opus",
        "sonnet",
    }.issubset(aliases)
    assert aliases.index("opus") < aliases.index("opus46")


def test_anthropic_catalog_lists_user_facing_factory_aliases() -> None:
    aliases = ModelSelectionCatalog.list_current_aliases(Provider.ANTHROPIC)

    assert aliases
    for alias in aliases:
        assert alias in ModelFactory.MODEL_PRESETS
    assert ModelFactory.MODEL_PRESETS["sonnet"] == "claude-sonnet-5"
    assert ModelFactory.MODEL_PRESETS["sonnet5"] == "claude-sonnet-5"
    assert ModelFactory.MODEL_PRESETS["fable"] == "claude-fable-5"
    assert ModelFactory.MODEL_PRESETS["fable5"] == "claude-fable-5"


def test_current_catalog_entries_match_model_presets_for_shared_aliases() -> None:
    for entry in ModelSelectionCatalog.list_current_entries():
        preset = ModelFactory.MODEL_PRESETS.get(entry.alias)
        if preset is not None:
            parsed_entry = ModelFactory.parse_model_string(entry.model)
            parsed_preset = ModelFactory.parse_model_string(preset)
            if parsed_entry.provider != parsed_preset.provider:
                continue
            assert parsed_entry.provider == parsed_preset.provider
            assert parsed_entry.model_name == parsed_preset.model_name
            assert parsed_entry.model_dump(
                exclude={"provider", "model_name"}
            ) == parsed_preset.model_dump(exclude={"provider", "model_name"})


def test_deepseek_current_order_prefers_pro_above_flash() -> None:
    aliases = ModelSelectionCatalog.list_current_aliases(Provider.DEEPSEEK)
    assert aliases[:2] == ["deepseek", "deepseek4flash"]


def test_non_current_aliases_are_listed_but_not_current() -> None:
    for provider in Provider:
        current_aliases = ModelSelectionCatalog.list_current_aliases(provider)
        non_current_aliases = ModelSelectionCatalog.list_non_current_aliases(provider)

        assert set(current_aliases).isdisjoint(non_current_aliases)
        assert non_current_aliases == [
            entry.alias
            for entry in ModelSelectionCatalog.CATALOG_ENTRIES_BY_PROVIDER.get(provider, ())
            if not entry.current
        ]


def test_list_fast_models_uses_explicit_curated_designation() -> None:
    for provider in (
        Provider.ANTHROPIC,
        Provider.CODEX_RESPONSES,
        Provider.HUGGINGFACE,
        Provider.GROQ,
    ):
        assert ModelSelectionCatalog.list_fast_models(provider) == unique_preserve_order(
            entry.model for entry in _static_current_entries(provider) if entry.fast
        )


def test_groq_current_aliases_drop_deprecated_kimi_entry() -> None:
    aliases = ModelSelectionCatalog.list_current_aliases(Provider.GROQ)

    assert "kimigroq" not in aliases
    assert "qwen3-32b" in aliases


@pytest.mark.parametrize(
    "provider",
    (
        Provider.ANTHROPIC,
        Provider.GOOGLE,
        Provider.HUGGINGFACE,
        Provider.CODEX_RESPONSES,
    ),
)
def test_current_catalog_helpers_project_current_entries(provider: Provider) -> None:
    current_entries = _static_current_entries(provider)

    assert ModelSelectionCatalog.list_current_aliases(provider) == [
        entry.alias for entry in current_entries
    ]
    assert ModelSelectionCatalog.list_current_models(provider) == unique_preserve_order(
        entry.model for entry in current_entries
    )
    assert ModelSelectionCatalog.list_fast_models(provider) == unique_preserve_order(
        entry.model for entry in current_entries if entry.fast
    )


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


def test_configured_providers_does_not_treat_anthropic_vertex_as_base_provider(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "fast_agent.llm.provider.anthropic.vertex_config.detect_google_adc",
        lambda: GoogleAdcStatus(available=True, project_id="proj", credentials=object()),
    )

    providers = ModelSelectionCatalog.configured_providers(
        {
            "anthropic": {
                "vertex_ai": {
                    "enabled": True,
                    "project_id": "proj",
                    "location": "global",
                }
            }
        }
    )

    assert Provider.ANTHROPIC not in providers


def test_configured_providers_reads_anthropic_vertex_env_only_setup(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "fast_agent.llm.provider.anthropic.vertex_config.detect_google_adc",
        lambda: GoogleAdcStatus(available=True, project_id="proj", credentials=object()),
    )
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "proj")

    providers = ModelSelectionCatalog.configured_providers({})

    assert Provider.ANTHROPIC_VERTEX in providers


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


def test_configured_providers_does_not_treat_overlay_only_provider_as_ready(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".fast-agent"
    overlays_dir = home / "model-overlays"
    overlays_dir.mkdir(parents=True)
    (overlays_dir / "haikutiny.yaml").write_text(
        "\n".join(
            [
                "name: haikutiny",
                "provider: anthropic",
                "model: claude-haiku-4-5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    previous_home = os.environ.get("FAST_AGENT_HOME")
    os.environ["FAST_AGENT_HOME"] = str(home)
    try:
        providers = ModelSelectionCatalog.configured_providers({})
    finally:
        empty_home = tmp_path / ".empty-fast-agent-overlay-ready"
        empty_home.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, home=empty_home)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home

    assert Provider.ANTHROPIC not in providers


def test_google_catalog_exposes_curated_and_fast_models() -> None:
    assert ModelSelectionCatalog.list_current_aliases(Provider.GOOGLE)
    assert ModelSelectionCatalog.list_non_current_aliases(Provider.GOOGLE) == []
    assert ModelSelectionCatalog.list_current_models(Provider.GOOGLE)
    assert ModelSelectionCatalog.list_fast_models(Provider.GOOGLE)
    assert ModelSelectionCatalog.list_all_models(Provider.GOOGLE)


def test_google_picker_lists_gemini35_flash_first() -> None:
    entries = ModelSelectionCatalog.list_entries(Provider.GOOGLE)
    current_entries = [entry for entry in entries if entry.current]

    assert current_entries
    first = current_entries[0]
    assert first.alias == "gemini35flash"
    assert first.model == "google.gemini-3.5-flash"
    assert first.fast is True


def test_catalog_lists_legacy_aliases_when_configured() -> None:
    current_aliases = ModelSelectionCatalog.list_current_aliases(Provider.HUGGINGFACE)
    non_current_aliases = ModelSelectionCatalog.list_non_current_aliases(Provider.HUGGINGFACE)

    assert "glm51" in current_aliases
    assert "glm5" in non_current_aliases
    assert "glm47" in non_current_aliases
    assert "glm47" not in current_aliases


def test_list_all_models_for_provider() -> None:
    openai_models = ModelSelectionCatalog.list_all_models(Provider.OPENAI)
    assert "gpt-4.1" in openai_models
    assert "claude-sonnet-4-6" not in openai_models


def test_cross_provider_overlay_alias_does_not_hide_curated_model(tmp_path: Path) -> None:
    home = tmp_path / ".fast-agent"
    overlays_dir = home / "model-overlays"
    overlays_dir.mkdir(parents=True)
    (overlays_dir / "sonnet.yaml").write_text(
        "\n".join(
            [
                "name: sonnet",
                "provider: openresponses",
                "model: overlay-tests/Qwen-Sonnet",
            ]
        ),
        encoding="utf-8",
    )

    previous_home = os.environ.get("FAST_AGENT_HOME")
    os.environ["FAST_AGENT_HOME"] = str(home)
    try:
        aliases = ModelSelectionCatalog.list_current_aliases(Provider.ANTHROPIC)
        assert "sonnet" in aliases
    finally:
        empty_home = tmp_path / ".empty-fast-agent"
        empty_home.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, home=empty_home)
        if previous_home is None:
            os.environ.pop("FAST_AGENT_HOME", None)
        else:
            os.environ["FAST_AGENT_HOME"] = previous_home


def test_codexresponses_current_entries_use_explicit_transports() -> None:
    current = ModelSelectionCatalog.list_current_models(Provider.CODEX_RESPONSES)
    assert "codexresponses.gpt-5.4?reasoning=high" in current
    assert "codexresponses.gpt-5.5?reasoning=medium" in current
    assert "codexresponses.gpt-5.3-codex-spark" in current


def test_google_curated_models_exist_in_provider_catalog() -> None:
    known = {
        ModelDatabase.normalize_model_name(model)
        for model in ModelSelectionCatalog.list_all_models(Provider.GOOGLE)
    }
    for entry in ModelSelectionCatalog.list_current_entries(Provider.GOOGLE):
        assert ModelDatabase.normalize_model_name(entry.model) in known


def test_openrouter_list_all_models_uses_discovery(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    def _stub_openrouter_models(*, api_key: str, base_url: str | None = None):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return [
            "openrouter.openai/gpt-4.1-mini",
            "openrouter.anthropic/claude-sonnet-4",
        ]

    monkeypatch.setattr(
        "fast_agent.llm.openrouter_model_lookup.list_openrouter_model_specs_sync",
        _stub_openrouter_models,
    )

    models = ModelSelectionCatalog.list_all_models(
        Provider.OPENROUTER,
        config={
            "openrouter": {
                "api_key": "or-test-key",
                "base_url": "https://openrouter.ai/api/v1",
            }
        },
    )

    assert captured["api_key"] == "or-test-key"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert "openrouter.openai/gpt-4.1-mini" in models
    assert "openrouter.anthropic/claude-sonnet-4" in models


def test_openrouter_all_models_use_discovered_models_when_no_curated(monkeypatch) -> None:
    monkeypatch.setattr(
        "fast_agent.llm.openrouter_model_lookup.list_openrouter_model_specs_sync",
        lambda **kwargs: ["openrouter.openai/gpt-4.1-mini"],
    )

    models = ModelSelectionCatalog.list_all_models(
        Provider.OPENROUTER, config={"openrouter": {"api_key": "or-test-key"}}
    )

    assert models == ["openrouter.openai/gpt-4.1-mini"]


def test_overlay_catalog_uses_explicit_environment_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "project-env"
    ambient_home = tmp_path / "ambient-env"
    _write_overlay(
        home,
        "projectoverlay",
        provider="openresponses",
        model="overlay-tests/project",
    )
    _write_overlay(
        ambient_home,
        "ambientoverlay",
        provider="openresponses",
        model="overlay-tests/ambient",
    )

    monkeypatch.setenv("FAST_AGENT_HOME", str(ambient_home))

    models = ModelSelectionCatalog.list_all_models(
        Provider.OPENRESPONSES,
        start_path=tmp_path,
        home=home,
    )
    current_aliases = ModelSelectionCatalog.list_current_aliases(
        Provider.OPENRESPONSES,
        start_path=tmp_path,
        home=home,
    )

    assert "openresponses.overlay-tests/project" in models
    assert "openresponses.overlay-tests/ambient" not in models
    assert "projectoverlay" in current_aliases
    assert "ambientoverlay" not in current_aliases
