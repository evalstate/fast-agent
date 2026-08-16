from __future__ import annotations

import pytest
from mcp_types import Implementation
from pydantic import ValidationError

from fast_agent.config import MCPServerSettings, get_settings


def _write_config(tmp_path, text: str):
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def test_mcp_server_implementation_mapping_is_validated() -> None:
    settings = MCPServerSettings.model_validate(
        {
            "command": "server",
            "implementation": {"name": "spoof", "version": "9.9.9"},
        }
    )

    assert isinstance(settings.implementation, Implementation)
    assert settings.implementation.name == "spoof"
    assert settings.implementation.version == "9.9.9"


def test_get_settings_loads_nested_mcp_schema_and_applies_layered_defaults(tmp_path) -> None:
    config_path = _write_config(
        tmp_path,
        """
mcp:
  defaults:
    protocol_mode: modern
    reconnect_on_disconnect: true
    include_instructions: true
  client:
    auto_sampling: false
  diagnostics:
    enabled: false
    timeline:
      steps: 8
      step_seconds: 15
  servers:
    explicit:
      command: echo
      reconnect_on_disconnect: false
    inherited:
      command: echo
""",
    )
    (tmp_path / "fastagent.secrets.yaml").write_text(
        """
mcp:
  defaults:
    include_instructions: false
  servers:
    inherited:
      include_instructions: true
""",
        encoding="utf-8",
    )

    settings = get_settings(config_path, no_home=True)

    assert settings.mcp is not None
    assert settings.mcp.client.auto_sampling is False
    assert settings.mcp.diagnostics.enabled is False
    assert settings.mcp.diagnostics.timeline.steps == 8
    assert settings.mcp.diagnostics.timeline.step_seconds == 15

    explicit = settings.mcp.servers["explicit"]
    assert explicit.name == "explicit"
    assert explicit.protocol_mode == "modern"
    assert explicit.reconnect_on_disconnect is False
    assert explicit.include_instructions is False

    inherited = settings.mcp.servers["inherited"]
    assert inherited.name == "inherited"
    assert inherited.protocol_mode == "modern"
    assert inherited.reconnect_on_disconnect is True
    assert inherited.include_instructions is True


def test_get_settings_rejects_mcp_targets_with_actionable_migration(tmp_path) -> None:
    config_path = _write_config(
        tmp_path,
        """
mcp:
  targets:
    - https://example.com
""",
    )

    with pytest.raises(ValidationError) as exc_info:
        get_settings(config_path, no_home=True)

    message = str(exc_info.value)
    assert "`mcp.targets` is no longer supported" in message
    assert "`fast-agent config migrate-mcp`" in message


def test_get_settings_enforces_server_map_key_identity(tmp_path) -> None:
    matching_path = _write_config(
        tmp_path,
        """
mcp:
  servers:
    demo:
      name: demo
      command: echo
""",
    )

    with pytest.warns(DeprecationWarning, match="map key is the canonical server name"):
        settings = get_settings(matching_path, no_home=True)

    assert settings.mcp is not None
    assert settings.mcp.servers["demo"].name == "demo"

    mismatched_path = _write_config(
        tmp_path,
        """
mcp:
  servers:
    demo:
      name: other
      command: echo
""",
    )

    with pytest.raises(ValidationError, match="must match its map key"):
        get_settings(mismatched_path, no_home=True)


@pytest.mark.parametrize("field", ["transport", "url", "command", "args", "connector_id"])
def test_get_settings_rejects_target_source_siblings(tmp_path, field: str) -> None:
    values = {
        "transport": "http",
        "url": "https://other.example.com/mcp",
        "command": "echo",
        "args": "[]",
        "connector_id": "connector_dropbox",
    }
    config_path = _write_config(
        tmp_path,
        f"""
mcp:
  servers:
    demo:
      target: https://example.com
      {field}: {values[field]}
""",
    )

    with pytest.raises(ValidationError, match="cannot be combined with source fields"):
        get_settings(config_path, no_home=True)


def test_get_settings_migrates_legacy_mcp_fields_and_rejects_ambiguity(tmp_path) -> None:
    legacy_path = _write_config(
        tmp_path,
        """
auto_sampling: false
mcp_timeline:
  steps: 7
  step_seconds: 120
""",
    )

    with pytest.warns(DeprecationWarning) as warnings_info:
        settings = get_settings(legacy_path, no_home=True)

    assert len(warnings_info) == 2
    assert settings.mcp is not None
    assert settings.mcp.client.auto_sampling is False
    assert settings.mcp.diagnostics.timeline.steps == 7
    assert settings.mcp.diagnostics.timeline.step_seconds == 120
    assert "auto_sampling" not in type(settings).model_fields
    assert "mcp_timeline" not in type(settings).model_fields

    ambiguous_path = _write_config(
        tmp_path,
        """
auto_sampling: false
mcp:
  client:
    auto_sampling: true
""",
    )

    with pytest.raises(ValidationError, match="cannot both be set"):
        get_settings(ambiguous_path, no_home=True)


@pytest.mark.parametrize("removed_key", ["mcp_ui_mode", "mcp_ui_output_dir"])
def test_get_settings_rejects_removed_mcp_ui_settings(tmp_path, removed_key: str) -> None:
    config_path = _write_config(tmp_path, f"{removed_key}: legacy\n")

    with pytest.raises(
        ValidationError,
        match="were removed in fast-agent 0.10.*MCP Apps",
    ):
        get_settings(config_path, no_home=True)
