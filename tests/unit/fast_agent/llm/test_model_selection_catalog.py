from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_overlays import load_model_overlay_registry
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider.anthropic.vertex_config import GoogleAdcStatus
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from pathlib import Path


def _write_overlay(env_dir: "Path", name: str, *, provider: str, model: str) -> None:
    overlays_dir = env_dir / "model-overlays"
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
    curated_models = ModelSelectionCatalog.list_curated_models(Provider.HUGGINGFACE)
    legacy_aliases = ModelSelectionCatalog.list_non_current_aliases(Provider.HUGGINGFACE)

    assert "minimax25" in curated_aliases
    assert (
        "hf.MiniMaxAI/MiniMax-M2.5:fireworks-ai?temperature=1.0&top_p=0.95&top_k=40"
        in curated_models
    )
    assert "qwen35" in curated_aliases
    assert "qwen35instruct" in curated_aliases
    assert "glm47" in legacy_aliases
    assert "glm47" not in curated_aliases


def test_list_fast_models_uses_explicit_curated_designation() -> None:
    for provider in (Provider.ANTHROPIC, Provider.CODEX_RESPONSES, Provider.HUGGINGFACE):
        assert ModelSelectionCatalog.list_fast_models(provider) == [
            entry.model for entry in _static_current_entries(provider) if entry.fast
        ]


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
    assert ModelSelectionCatalog.list_current_models(provider) == [
        entry.model for entry in current_entries
    ]
    assert ModelSelectionCatalog.list_fast_models(provider) == [
        entry.model for entry in current_entries if entry.fast
    ]


@pytest.mark.parametrize(
    "provider",
    (
        Provider.ANTHROPIC,
        Provider.GOOGLE,
        Provider.HUGGINGFACE,
        Provider.CODEX_RESPONSES,
    ),
)
def test_curated_helpers_alias_current_helpers(provider: Provider) -> None:
    assert ModelSelectionCatalog.list_curated_aliases(
        provider
    ) == ModelSelectionCatalog.list_current_aliases(provider)
    assert ModelSelectionCatalog.list_curated_models(
        provider
    ) == ModelSelectionCatalog.list_current_models(provider)


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
    env_dir = tmp_path / ".fast-agent"
    overlays_dir = env_dir / "model-overlays"
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

    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)
    try:
        providers = ModelSelectionCatalog.configured_providers({})
    finally:
        empty_env_dir = tmp_path / ".empty-fast-agent-overlay-ready"
        empty_env_dir.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, env_dir=empty_env_dir)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir

    assert Provider.ANTHROPIC not in providers


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


def test_cross_provider_overlay_alias_does_not_hide_curated_model(tmp_path: Path) -> None:
    env_dir = tmp_path / ".fast-agent"
    overlays_dir = env_dir / "model-overlays"
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

    previous_env_dir = os.environ.get("ENVIRONMENT_DIR")
    os.environ["ENVIRONMENT_DIR"] = str(env_dir)
    try:
        aliases = ModelSelectionCatalog.list_curated_aliases(Provider.ANTHROPIC)
        assert "sonnet" in aliases
    finally:
        empty_env_dir = tmp_path / ".empty-fast-agent"
        empty_env_dir.mkdir(parents=True, exist_ok=True)
        load_model_overlay_registry(start_path=tmp_path, env_dir=empty_env_dir)
        if previous_env_dir is None:
            os.environ.pop("ENVIRONMENT_DIR", None)
        else:
            os.environ["ENVIRONMENT_DIR"] = previous_env_dir


def test_codexresponses_curated_entries_use_explicit_transports() -> None:
    curated = ModelSelectionCatalog.list_curated_models(Provider.CODEX_RESPONSES)
    assert "codexresponses.gpt-5.4?reasoning=high" in curated
    assert "codexresponses.gpt-5.3-codex-spark" in curated


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


def test_openrouter_suggestions_use_discovered_models_when_no_curated(monkeypatch) -> None:
    monkeypatch.setattr(
        "fast_agent.llm.openrouter_model_lookup.list_openrouter_model_specs_sync",
        lambda **kwargs: ["openrouter.openai/gpt-4.1-mini"],
    )

    suggestions = ModelSelectionCatalog.suggestions_for_providers(
        [Provider.OPENROUTER],
        config={"openrouter": {"api_key": "or-test-key"}},
    )

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.provider == Provider.OPENROUTER
    assert suggestion.current_models == ("openrouter.openai/gpt-4.1-mini",)
    assert suggestion.all_models == ("openrouter.openai/gpt-4.1-mini",)


def test_overlay_catalog_uses_explicit_environment_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "project-env"
    ambient_env_dir = tmp_path / "ambient-env"
    _write_overlay(
        env_dir,
        "projectoverlay",
        provider="openresponses",
        model="overlay-tests/project",
    )
    _write_overlay(
        ambient_env_dir,
        "ambientoverlay",
        provider="openresponses",
        model="overlay-tests/ambient",
    )

    monkeypatch.setenv("ENVIRONMENT_DIR", str(ambient_env_dir))

    models = ModelSelectionCatalog.list_all_models(
        Provider.OPENRESPONSES,
        start_path=tmp_path,
        env_dir=env_dir,
    )
    suggestions = ModelSelectionCatalog.suggestions_for_providers(
        [Provider.OPENRESPONSES],
        start_path=tmp_path,
        env_dir=env_dir,
    )

    assert "openresponses.overlay-tests/project" in models
    assert "openresponses.overlay-tests/ambient" not in models
    assert suggestions[0].current_aliases[0] == "projectoverlay"
    assert "ambientoverlay" not in suggestions[0].current_aliases
