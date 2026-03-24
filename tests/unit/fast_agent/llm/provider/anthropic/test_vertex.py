import types

import pytest

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.anthropic.vertex_config import GoogleAdcStatus
from fast_agent.llm.provider_key_manager import ProviderKeyManager


def _build_llm(config: Settings, *, via: str | None = None) -> AnthropicLLM:
    kwargs = {"context": Context(config=config), "model": "claude-sonnet-4-6"}
    if via is not None:
        kwargs["via"] = via
    return AnthropicLLM(**kwargs)


def test_vertex_cfg_accepts_model_object() -> None:
    anthropic = AnthropicSettings()
    setattr(
        anthropic,
        "vertex_ai",
        types.SimpleNamespace(
            enabled=True,
            project_id="proj",
            location="global",
            base_url="https://vertex.example",
        ),
    )
    config = Settings(anthropic=anthropic)

    llm = _build_llm(config)
    vertex_cfg = llm._vertex_cfg()

    assert vertex_cfg.enabled is True
    assert vertex_cfg.project_id == "proj"
    assert vertex_cfg.location == "global"
    assert vertex_cfg.base_url == "https://vertex.example"


def test_provider_key_manager_allows_vertex_route_without_api_key() -> None:
    config = Settings.model_validate(
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

    assert ProviderKeyManager.get_api_key("anthropic", config, route_hint="vertex") == ""
    with pytest.raises(ProviderKeyError):
        ProviderKeyManager.get_api_key("anthropic", config)
    with pytest.raises(ProviderKeyError):
        ProviderKeyManager.get_api_key("anthropic", config, route_hint="direct")


def test_initialize_anthropic_client_uses_vertex(monkeypatch) -> None:
    config = Settings.model_validate(
        {
            "anthropic": {
                "default_headers": {"X-Test": "vertex"},
                "vertex_ai": {
                    "project_id": "proj",
                    "location": "global",
                    "base_url": "https://vertex.example",
                },
            }
        }
    )
    llm = _build_llm(config, via="vertex")

    called: dict[str, object] = {}

    class FakeVertexClient:
        def __init__(self, **kwargs) -> None:
            called.update(kwargs)

    monkeypatch.setattr(
        "fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropicVertex",
        FakeVertexClient,
    )
    monkeypatch.setattr(
        "fast_agent.llm.provider.anthropic.llm_anthropic.detect_google_adc",
        lambda: GoogleAdcStatus(available=True, project_id="proj", credentials=object()),
    )

    client = llm._initialize_anthropic_client()

    assert isinstance(client, FakeVertexClient)
    assert called["project_id"] == "proj"
    assert called["region"] == "global"
    assert called["base_url"] == "https://vertex.example"
    assert called["default_headers"] == {"X-Test": "vertex"}
    assert "api_key" not in called


def test_initialize_anthropic_client_uses_direct_sdk(monkeypatch) -> None:
    config = Settings.model_validate(
        {
            "anthropic": {
                "api_key": "sk-ant",
                "base_url": "https://api.anthropic.example/v1",
                "default_headers": {"X-Test": "direct"},
            }
        }
    )
    llm = _build_llm(config)

    called: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            called.update(kwargs)

    monkeypatch.setattr(
        "fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic",
        FakeClient,
    )

    client = llm._initialize_anthropic_client()

    assert isinstance(client, FakeClient)
    assert called["api_key"] == "sk-ant"
    assert called["base_url"] == "https://api.anthropic.example"
    assert called["default_headers"] == {"X-Test": "direct"}


def test_vertex_client_requires_google_adc(monkeypatch) -> None:
    config = Settings.model_validate(
        {
            "anthropic": {
                "vertex_ai": {
                    "project_id": "proj",
                    "location": "global",
                }
            }
        }
    )
    llm = _build_llm(config, via="vertex")

    monkeypatch.setattr(
        "fast_agent.llm.provider.anthropic.llm_anthropic.detect_google_adc",
        lambda: GoogleAdcStatus(available=False, error=RuntimeError("missing")),
    )

    with pytest.raises(ProviderKeyError, match="Google ADC not found"):
        llm._initialize_anthropic_client()
