"""Unit tests for the LiteLLM provider integration."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_model_catalog import (
    LiteLLMModelCatalogAdapter,
    ProviderModelCatalogRegistry,
)
from fast_agent.llm.provider_types import Provider


def test_provider_enum_includes_litellm() -> None:
    assert Provider.LITELLM.config_name == "litellm"
    assert Provider.LITELLM.display_name == "LiteLLM"


def test_picker_provider_order_includes_litellm() -> None:
    from fast_agent.ui.model_picker_common import PICKER_PROVIDER_ORDER

    assert Provider.LITELLM in PICKER_PROVIDER_ORDER


def test_provider_is_active_true_when_litellm_importable() -> None:
    from fast_agent.ui.model_picker_common import _provider_is_active

    # litellm is in test deps; this should resolve to True
    assert _provider_is_active(Provider.LITELLM, {}) is True


def test_provider_is_active_false_when_litellm_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wizard should not advertise LiteLLM as available if the SDK isn't installed."""
    import builtins

    from fast_agent.ui.model_picker_common import _provider_is_active

    real_import = builtins.__import__

    def _no_litellm(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "litellm" or name.startswith("litellm."):
            raise ImportError("litellm not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_litellm)
    assert _provider_is_active(Provider.LITELLM, {}) is False


def test_litellm_is_keyless_at_fast_agent_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """LiteLLM resolves backing-provider creds itself; no LITELLM_API_KEY required."""
    from fast_agent.llm import provider_key_manager as pkm

    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    assert "litellm" in pkm.API_KEYLESS_PROVIDERS
    # api_key resolves to empty string when nothing is configured (does not raise)
    assert ProviderKeyManager.get_api_key("litellm", {}) == ""


def test_litellm_picks_up_proxy_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "proxy-key-xyz")
    assert ProviderKeyManager.get_api_key("litellm", {}) == "proxy-key-xyz"


def test_litellm_picks_up_proxy_api_key_from_config() -> None:
    config = {"litellm": {"api_key": "config-key-abc"}}
    assert ProviderKeyManager.get_api_key("litellm", config) == "config-key-abc"


def test_factory_dispatches_to_litellm_class() -> None:
    cls = ModelFactory._load_provider_class(Provider.LITELLM)  # noqa: SLF001
    from fast_agent.llm.provider.litellm.llm_litellm import LiteLLMLLM

    assert cls is LiteLLMLLM


def test_model_spec_parses_into_litellm_provider_and_path() -> None:
    parsed = ModelFactory.parse_model_string("litellm.anthropic/claude-sonnet-4-5")
    assert parsed.provider is Provider.LITELLM
    assert parsed.model_name == "anthropic/claude-sonnet-4-5"


def test_model_spec_parses_with_colons_and_at() -> None:
    """Bedrock and Vertex specs include `:` and `@` characters."""
    parsed = ModelFactory.parse_model_string(
        "litellm.bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    assert parsed.provider is Provider.LITELLM
    assert parsed.model_name == "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"

    parsed = ModelFactory.parse_model_string("litellm.vertex_ai/claude-sonnet-4-5@20250929")
    assert parsed.provider is Provider.LITELLM
    assert parsed.model_name == "vertex_ai/claude-sonnet-4-5@20250929"


def test_catalog_adapter_returns_full_litellm_inventory() -> None:
    adapter = LiteLLMModelCatalogAdapter()
    inventory = adapter.discover({})
    # Real LiteLLM SDK has 80+ provider keys and 1000+ models
    assert len(inventory.all_models) > 1000
    assert inventory.current_models == inventory.all_models
    assert all(spec.startswith("litellm.") for spec in inventory.all_models)


def test_catalog_adapter_collapses_redundant_provider_prefix() -> None:
    """When a model in `models_by_provider[X]` already starts with `X/`, our
    spec should collapse it to a single `litellm.X/...` rather than emitting
    `litellm.X/X/...`.

    Note: LiteLLM's data legitimately contains nested specs like
    `openrouter/openrouter/auto` (where the second `openrouter/` is part of
    the OpenRouter model name itself). Those are valid and must be preserved
    when round-tripping into `litellm.acompletion(model=...)`.
    """
    adapter = LiteLLMModelCatalogAdapter()
    inventory = adapter.discover({})

    # Common, well-known specs should round-trip cleanly without redundant prefixes
    expected = {
        "litellm.openai/gpt-4o",
        "litellm.anthropic/claude-sonnet-4-5",
        "litellm.gemini/gemini-2.5-pro",
        "litellm.groq/llama-3.3-70b-versatile",
    }
    seen = set(inventory.all_models)
    missing = expected - seen
    assert missing == set(), f"Expected popular LiteLLM specs missing: {missing}"


def test_catalog_adapter_returns_empty_when_litellm_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def _no_litellm(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "litellm":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_litellm)
    adapter = LiteLLMModelCatalogAdapter()
    inventory = adapter.discover({})
    assert inventory.all_models == ()
    assert inventory.current_models == ()


def test_registry_dispatches_to_litellm_adapter() -> None:
    inventory = ProviderModelCatalogRegistry.discover(Provider.LITELLM, {})
    assert len(inventory.all_models) > 1000


def test_curated_entries_present_in_picker_snapshot() -> None:
    """Wizard snapshot should include LiteLLM with at least 10 curated entries."""
    from fast_agent.ui.model_picker_common import build_snapshot, find_provider

    snapshot = build_snapshot(config_path=None)
    option = find_provider(snapshot, "litellm")
    assert option.active is True
    assert len(option.curated_entries) >= 10


def test_per_model_backing_available_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each LiteLLM curated row should reflect whether its backing provider has creds.

    This is what makes the wizard's ✓/✗ markers meaningful: a user without
    `OPENAI_API_KEY` set should see ✗ next to LiteLLM's OpenAI rows even
    though the LiteLLM SDK itself is installed.
    """
    from fast_agent.ui.model_picker_common import (
        build_snapshot,
        find_provider,
        model_options_for_option,
    )

    # Strip every *_API_KEY and *_BASE_URL so all backings start unconfigured
    for key in list(__import__("os").environ):
        if key.endswith("_API_KEY") or key.endswith("_BASE_URL"):
            monkeypatch.delenv(key, raising=False)

    snapshot = build_snapshot(config_path=None)
    option = find_provider(snapshot, "litellm")
    models = {m.spec: m for m in model_options_for_option(snapshot, option, source="curated")}

    # No creds: all should be False (LiteLLM-known) or None (unknown spec)
    openai_row = models["litellm.openai/gpt-4o"]
    assert openai_row.backing_available is False
    anthropic_row = models["litellm.anthropic/claude-sonnet-4-6"]
    assert anthropic_row.backing_available is False

    # Set ANTHROPIC_API_KEY -> only the anthropic rows flip to True
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    snapshot2 = build_snapshot(config_path=None)
    option2 = find_provider(snapshot2, "litellm")
    models2 = {m.spec: m for m in model_options_for_option(snapshot2, option2, source="curated")}

    assert models2["litellm.anthropic/claude-sonnet-4-6"].backing_available is True
    assert models2["litellm.openai/gpt-4o"].backing_available is False


def test_litellm_backing_creds_present_returns_none_for_non_litellm_spec() -> None:
    from fast_agent.ui.model_picker_common import litellm_backing_creds_present

    assert litellm_backing_creds_present("openai.gpt-4o") is None
    assert litellm_backing_creds_present("anthropic.claude-sonnet-4-5") is None


def test_all_scope_includes_dynamic_catalog() -> None:
    """The picker's `all` scope should pull in LiteLLM's full registry, not just curated."""
    from fast_agent.ui.model_picker_common import (
        build_snapshot,
        find_provider,
        model_options_for_option,
    )

    snapshot = build_snapshot(config_path=None)
    option = find_provider(snapshot, "litellm")
    curated = model_options_for_option(snapshot, option, source="curated")
    full = model_options_for_option(snapshot, option, source="all")
    assert len(full) > len(curated) + 100
    # Curated entries are still first
    curated_specs = {opt.spec for opt in curated}
    assert curated_specs.issubset({opt.spec for opt in full})


# --- Provider class shape tests --------------------------------------------------


def test_litellm_client_shim_is_async_context_manager() -> None:
    from fast_agent.llm.provider.litellm.llm_litellm import _LiteLLMClientShim

    shim = _LiteLLMClientShim(
        api_key=None,
        base_url=None,
        default_headers=None,
        drop_params=True,
        timeout=None,
    )
    assert hasattr(shim, "__aenter__")
    assert hasattr(shim, "__aexit__")
    assert hasattr(shim.chat, "completions")
    assert callable(shim.chat.completions.create)


@pytest.mark.asyncio
async def test_shim_forwards_api_base_and_drop_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shim must pass `api_base`, `api_key`, `timeout`, and `drop_params` to litellm.acompletion."""
    from fast_agent.llm.provider.litellm.llm_litellm import _LiteLLMClientShim

    captured: dict[str, Any] = {}

    async def _fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return MagicMock()

    import litellm

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)

    shim = _LiteLLMClientShim(
        api_key="proxy-key",
        base_url="http://localhost:4000",
        default_headers={"X-Trace": "abc"},
        drop_params=True,
        timeout=45,
    )
    await shim.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )

    assert captured["api_key"] == "proxy-key"
    assert captured["api_base"] == "http://localhost:4000"
    assert captured["extra_headers"] == {"X-Trace": "abc"}
    assert captured["timeout"] == 45
    assert captured["drop_params"] is True
    assert captured["stream"] is True
    assert captured["model"] == "anthropic/claude-sonnet-4-5"


def test_config_bridges_anthropic_api_key_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`anthropic.api_key` in fastagent.config.yaml should be exported as
    ANTHROPIC_API_KEY so LiteLLM's anthropic backing picks it up at call time.
    """
    from fast_agent.llm.provider.litellm.llm_litellm import (
        _bridge_fastagent_config_to_litellm_env,
    )

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _Section:
        def __init__(self, **fields: Any) -> None:
            for key, value in fields.items():
                setattr(self, key, value)

    class _FakeConfig:
        anthropic = _Section(api_key="cfg-anthropic-key", base_url=None)
        openai = _Section(api_key=None, base_url=None)

    _bridge_fastagent_config_to_litellm_env(_FakeConfig())

    assert os.environ.get("ANTHROPIC_API_KEY") == "cfg-anthropic-key"
    assert os.environ.get("OPENAI_API_KEY") is None


def test_config_bridge_does_not_overwrite_existing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """User-exported env vars must take precedence over the config file."""
    from fast_agent.llm.provider.litellm.llm_litellm import (
        _bridge_fastagent_config_to_litellm_env,
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-wins")

    class _Section:
        api_key = "cfg-loses"
        base_url = None

    class _FakeConfig:
        anthropic = _Section()

    _bridge_fastagent_config_to_litellm_env(_FakeConfig())

    assert os.environ.get("ANTHROPIC_API_KEY") == "env-wins"


def test_config_bridge_handles_none_config() -> None:
    from fast_agent.llm.provider.litellm.llm_litellm import (
        _bridge_fastagent_config_to_litellm_env,
    )

    # Should not raise when config is None
    _bridge_fastagent_config_to_litellm_env(None)


@pytest.mark.asyncio
async def test_shim_does_not_overwrite_explicit_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the caller passes `timeout` or `api_key`, the shim must not override it."""
    from fast_agent.llm.provider.litellm.llm_litellm import _LiteLLMClientShim

    captured: dict[str, Any] = {}

    async def _fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return MagicMock()

    import litellm

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)

    shim = _LiteLLMClientShim(
        api_key="shim-key",
        base_url="http://shim",
        default_headers=None,
        drop_params=False,
        timeout=10,
    )
    await shim.chat.completions.create(
        model="openai/gpt-4o",
        messages=[],
        api_key="caller-key",
        timeout=99,
    )

    assert captured["api_key"] == "caller-key"
    assert captured["timeout"] == 99
    assert captured["api_base"] == "http://shim"
    assert captured["drop_params"] is False
