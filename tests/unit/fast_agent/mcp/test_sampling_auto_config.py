from __future__ import annotations

import pytest

from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.mcp.sampling import _select_sampling_model, resolve_auto_sampling_enabled


def test_auto_sampling_defaults_to_enabled_without_context() -> None:
    assert resolve_auto_sampling_enabled(None) is True
    assert resolve_auto_sampling_enabled(Context(config=None)) is True


def test_auto_sampling_uses_settings_value() -> None:
    enabled = Settings.model_validate({"mcp": {"client": {"auto_sampling": True}}})
    disabled = Settings.model_validate({"mcp": {"client": {"auto_sampling": False}}})

    assert resolve_auto_sampling_enabled(Context(config=enabled)) is True
    assert resolve_auto_sampling_enabled(Context(config=disabled)) is False


def test_auto_sampling_requires_a_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FAST_AGENT_MODEL", raising=False)
    context = Context(config=Settings())

    with pytest.raises(ModelConfigError, match="No model configured for MCP sampling"):
        _select_sampling_model(
            server_config=None,
            agent_model=None,
            api_key=None,
            app_context=context,
        )


def test_empty_sampling_settings_do_not_select_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.config import MCPSamplingSettings, MCPServerSettings

    monkeypatch.delenv("FAST_AGENT_MODEL", raising=False)
    context = Context(config=Settings())
    server_config = MCPServerSettings(sampling=MCPSamplingSettings())

    with pytest.raises(ModelConfigError, match="No model configured for MCP sampling"):
        _select_sampling_model(
            server_config=server_config,
            agent_model=None,
            api_key=None,
            app_context=context,
        )
