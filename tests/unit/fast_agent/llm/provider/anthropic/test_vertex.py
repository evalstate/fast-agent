import types

import pytest
from pydantic import BaseModel

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.provider.anthropic.beta_types import ToolParam
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.anthropic.llm_anthropic_vertex import AnthropicVertexLLM
from fast_agent.llm.provider.anthropic.vertex_config import (
    GoogleAdcStatus,
    anthropic_vertex_config,
)
from fast_agent.llm.provider_key_manager import ProviderKeyManager


class _StructuredResponse(BaseModel):
    answer: str


def _build_direct_llm(config: Settings) -> AnthropicLLM:
    return AnthropicLLM(context=Context(config=config), model="claude-sonnet-4-6")


def _build_vertex_llm(config: Settings) -> AnthropicVertexLLM:
    return AnthropicVertexLLM(context=Context(config=config), model="claude-sonnet-4-6")


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

    vertex_cfg = anthropic_vertex_config(config)

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

    assert ProviderKeyManager.get_api_key("anthropic-vertex", config) == ""
    with pytest.raises(ProviderKeyError):
        ProviderKeyManager.get_api_key("anthropic", config)


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
    llm = _build_vertex_llm(config)

    called: dict[str, object] = {}

    class FakeVertexClient:
        def __init__(self, **kwargs) -> None:
            called.update(kwargs)

    monkeypatch.setattr(
        "fast_agent.llm.provider.anthropic.llm_anthropic_vertex.AsyncAnthropicVertex",
        FakeVertexClient,
    )
    monkeypatch.setattr(
        "fast_agent.llm.provider.anthropic.llm_anthropic_vertex.detect_google_adc",
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
    llm = _build_direct_llm(config)

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
    llm = _build_vertex_llm(config)

    monkeypatch.setattr(
        "fast_agent.llm.provider.anthropic.llm_anthropic_vertex.detect_google_adc",
        lambda: GoogleAdcStatus(available=False, error=RuntimeError("missing")),
    )

    with pytest.raises(ProviderKeyError, match="Google ADC not found"):
        llm._initialize_anthropic_client()


def test_vertex_beta_support_is_selective() -> None:
    llm = AnthropicVertexLLM(
        context=Context(
            config=Settings.model_validate(
                {
                    "anthropic": {
                        "vertex_ai": {
                            "project_id": "proj",
                            "location": "global",
                        }
                    }
                }
            )
        ),
        model="claude-sonnet-4-5",
        long_context=True,
    )
    request_tools = [ToolParam(name="demo", description="", input_schema={})]

    beta_flags = llm._resolve_anthropic_beta_flags(
        model="claude-sonnet-4-5",
        structured_mode="json",
        thinking_enabled=False,
        request_tools=request_tools,
        web_tool_betas=("code-execution-web-tools-2026-02-09",),
    )

    assert "structured-outputs-2025-11-13" not in beta_flags
    assert "context-1m-2025-08-07" in beta_flags
    assert "fine-grained-tool-streaming-2025-05-14" in beta_flags
    assert "code-execution-web-tools-2026-02-09" in beta_flags


def test_vertex_supports_web_search_but_not_web_fetch() -> None:
    llm = _build_vertex_llm(
        Settings.model_validate(
            {
                "anthropic": {
                    "vertex_ai": {
                        "project_id": "proj",
                        "location": "global",
                    }
                }
            }
        )
    )

    assert llm.web_search_supported is True
    assert llm.web_fetch_supported is False


def test_vertex_auto_structured_output_mode_falls_back_to_tool_use() -> None:
    llm = _build_vertex_llm(
        Settings.model_validate(
            {
                "anthropic": {
                    "vertex_ai": {
                        "project_id": "proj",
                        "location": "global",
                    }
                }
            }
        )
    )

    structured_mode = llm._resolve_structured_output_mode(
        "claude-sonnet-4-6",
        _StructuredResponse,
    )

    assert structured_mode == "tool_use"
