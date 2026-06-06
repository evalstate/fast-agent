from fast_agent.config import Settings
from fast_agent.core.direct_factory import _resolve_mcp_ui_mode


def test_resolve_mcp_ui_mode_defaults_to_auto_without_settings() -> None:
    assert _resolve_mcp_ui_mode(None) == "auto"


def test_resolve_mcp_ui_mode_uses_typed_settings_value() -> None:
    assert _resolve_mcp_ui_mode(Settings(mcp_ui_mode="disabled")) == "disabled"
    assert _resolve_mcp_ui_mode(Settings(mcp_ui_mode="enabled")) == "enabled"
