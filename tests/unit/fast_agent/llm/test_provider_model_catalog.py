from __future__ import annotations

from fast_agent.llm.provider_model_catalog import (
    OpenRouterModelCatalogAdapter,
    ProviderModelCatalogRegistry,
)
from fast_agent.llm.provider_types import Provider


def test_openrouter_catalog_adapter_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    adapter = OpenRouterModelCatalogAdapter()
    inventory = adapter.discover({})

    assert inventory.current_models == ()
    assert inventory.all_models == ()


def test_openrouter_catalog_adapter_discovers_models(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    def _stub_openrouter_models(*, api_key: str, base_url: str | None = None):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return ["openrouter.openai/gpt-4.1-mini"]

    monkeypatch.setattr(
        "fast_agent.llm.openrouter_model_lookup.list_openrouter_model_specs_sync",
        _stub_openrouter_models,
    )

    adapter = OpenRouterModelCatalogAdapter()
    inventory = adapter.discover(
        {
            "openrouter": {
                "api_key": "or-test-key",
                "base_url": "https://example.com/v1",
            }
        }
    )

    assert captured["api_key"] == "or-test-key"
    assert captured["base_url"] == "https://example.com/v1"
    assert inventory.current_models == ("openrouter.openai/gpt-4.1-mini",)
    assert inventory.all_models == ("openrouter.openai/gpt-4.1-mini",)


def test_openrouter_catalog_adapter_prefers_config_base_url_over_env(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    def _stub_openrouter_models(*, api_key: str, base_url: str | None = None):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return []

    monkeypatch.setattr(
        "fast_agent.llm.openrouter_model_lookup.list_openrouter_model_specs_sync",
        _stub_openrouter_models,
    )
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://env-base-url.example/v1")

    adapter = OpenRouterModelCatalogAdapter()
    _ = adapter.discover(
        {
            "openrouter": {
                "api_key": "or-test-key",
                "base_url": "https://config-base-url.example/v1",
            }
        }
    )

    assert captured["api_key"] == "or-test-key"
    assert captured["base_url"] == "https://config-base-url.example/v1"


def test_provider_model_catalog_registry_returns_empty_for_static_provider() -> None:
    inventory = ProviderModelCatalogRegistry.discover(Provider.OPENAI, {})
    assert inventory.current_models == ()
    assert inventory.all_models == ()
