from __future__ import annotations

from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.mcp.sampling import resolve_auto_sampling_enabled


def test_auto_sampling_defaults_to_enabled_without_context() -> None:
    assert resolve_auto_sampling_enabled(None) is True
    assert resolve_auto_sampling_enabled(Context(config=None)) is True


def test_auto_sampling_uses_settings_value() -> None:
    enabled = Settings.model_validate({"mcp": {"client": {"auto_sampling": True}}})
    disabled = Settings.model_validate({"mcp": {"client": {"auto_sampling": False}}})

    assert resolve_auto_sampling_enabled(Context(config=enabled)) is True
    assert resolve_auto_sampling_enabled(Context(config=disabled)) is False
