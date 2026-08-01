from __future__ import annotations

import pytest
from pydantic import ValidationError

from fast_agent.config import MCPServerSettings, MCPSettings, Settings, load_yaml_mapping


def test_default_mcp_settings_are_per_settings_instance() -> None:
    first = Settings()
    second = Settings()

    assert first.mcp is not None
    assert second.mcp is not None
    first.mcp.servers["demo"] = MCPServerSettings(name="demo", transport="stdio", command="echo")

    assert second.mcp.servers == {}


def test_mcp_source_and_effective_views_preserve_shorthand_and_redact_secrets() -> None:
    settings = Settings.model_validate(
        {
            "mcp": {
                "defaults": {"protocol_mode": "modern"},
                "servers": {
                    "docs": {
                        "target": "https://user:password@example.com/mcp",
                        "headers": {
                            "Authorization": "Bearer secret",
                            "Cookie": "session=secret",
                            "X-Api-Key": "secret",
                            "X-Tenant": "engineering",
                        },
                        "auth": {
                            "api_key": "secret",
                            "client_secret": "secret",
                        },
                    }
                },
            }
        }
    )

    assert settings.mcp is not None
    source = settings.mcp.source_server_view()
    effective = settings.mcp.effective_server_view()

    assert source["docs"]["target"] == "https://[REDACTED]@example.com/mcp"
    assert source["docs"]["headers"] == {
        "Authorization": "[REDACTED]",
        "Cookie": "[REDACTED]",
        "X-Api-Key": "[REDACTED]",
        "X-Tenant": "engineering",
    }
    assert source["docs"]["auth"]["api_key"] == "[REDACTED]"
    assert source["docs"]["auth"]["client_secret"] == "[REDACTED]"
    assert "transport" not in source["docs"]
    assert effective["docs"]["protocol_mode"] == "modern"
    assert effective["docs"]["headers"]["Authorization"] == "[REDACTED]"
    assert effective["docs"]["_provenance"]["url"] == "target"
    assert effective["docs"]["_provenance"]["protocol_mode"] == "mcp.defaults"


def test_wrapping_mcp_settings_preserves_source_declarations_and_provenance() -> None:
    original = MCPSettings.model_validate(
        {
            "servers": {
                "docs": {
                    "target": "https://example.com/mcp",
                }
            }
        }
    )

    wrapped = Settings(mcp=original)

    assert wrapped.mcp is not None
    assert wrapped.mcp.source_server_view() == {"docs": {"target": "https://example.com/mcp"}}
    assert wrapped.mcp.effective_server_view()["docs"]["_provenance"]["url"] == "target"


def test_config_mcp_target_shorthand_url_expansion() -> None:
    settings = Settings.model_validate(
        {
            "mcp": {
                "servers": {
                    "demo": {
                        "target": "https://demo.hf.space",
                    }
                }
            }
        }
    )

    assert settings.mcp is not None
    demo = settings.mcp.servers["demo"]
    assert demo.name == "demo"
    assert demo.transport == "http"
    assert demo.url == "https://demo.hf.space/mcp"
    assert demo.protocol_mode == "auto"


@pytest.mark.parametrize("protocol_mode", ["auto", "modern", "legacy"])
def test_mcp_server_settings_accepts_protocol_modes(protocol_mode: str) -> None:
    settings = MCPServerSettings.model_validate({"protocol_mode": protocol_mode})

    assert settings.protocol_mode == protocol_mode


def test_mcp_server_settings_rejects_invalid_protocol_mode() -> None:
    with pytest.raises(ValidationError, match="protocol_mode"):
        MCPServerSettings.model_validate({"protocol_mode": "discover"})


def test_mcp_server_settings_rejects_forced_modern_over_legacy_sse() -> None:
    with pytest.raises(ValidationError, match="not supported with legacy SSE"):
        MCPServerSettings.model_validate(
            {
                "transport": "sse",
                "url": "https://example.com/sse",
                "protocol_mode": "modern",
            }
        )


def test_config_mcp_target_shorthand_preserves_operational_siblings() -> None:
    settings = Settings.model_validate(
        {
            "mcp": {
                "defaults": {
                    "protocol_mode": "legacy",
                    "reconnect_on_disconnect": False,
                    "include_instructions": False,
                },
                "servers": {
                    "secure_api": {
                        "target": "https://api.example.com",
                        "load_on_start": False,
                        "reconnect_on_disconnect": True,
                        "headers": {"Authorization": "Bearer override"},
                    }
                },
            }
        }
    )

    assert settings.mcp is not None
    secure_api = settings.mcp.servers["secure_api"]
    assert secure_api.load_on_start is False
    assert secure_api.transport == "http"
    assert secure_api.url == "https://api.example.com/mcp"
    assert secure_api.headers == {"Authorization": "Bearer override"}
    assert secure_api.protocol_mode == "legacy"
    assert secure_api.reconnect_on_disconnect is True
    assert secure_api.include_instructions is False


def test_config_mcp_target_shorthand_keeps_legacy_canonical_shape() -> None:
    settings = Settings.model_validate(
        {
            "mcp": {
                "servers": {
                    "filesystem": {
                        "command": "npx",
                        "args": ["@modelcontextprotocol/server-filesystem"],
                    }
                }
            }
        }
    )

    assert settings.mcp is not None
    filesystem = settings.mcp.servers["filesystem"]
    assert filesystem.transport == "stdio"
    assert filesystem.command == "npx"
    assert filesystem.args == ["@modelcontextprotocol/server-filesystem"]


def test_mcp_server_settings_rejects_boolean_max_missed_pings() -> None:
    assert MCPServerSettings.model_validate({"max_missed_pings": "3"}).max_missed_pings == 3

    with pytest.raises(TypeError, match="max_missed_pings must be an integer"):
        MCPServerSettings.model_validate({"max_missed_pings": True})


def test_config_mcp_target_shorthand_rejects_embedded_cli_flags() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate(
            {
                "mcp": {
                    "servers": {
                        "bad": {
                            "target": "https://example.com --auth token",
                        }
                    }
                }
            }
        )

    message = str(exc_info.value)
    assert "mcp.servers.bad.target" in message
    assert "pure target string" in message
    assert "--auth" in message


def test_config_mcp_targets_is_rejected_with_migration_command() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate(
            {
                "mcp": {
                    "targets": ["https://example.com"],
                }
            }
        )

    message = str(exc_info.value)
    assert "`mcp.targets` is no longer supported" in message
    assert "`fast-agent config migrate-mcp`" in message


def test_provider_managed_target_normalizes_url_and_access_token() -> None:
    settings = Settings.model_validate(
        {
            "mcp": {
                "servers": {
                    "stripe": {
                        "target": "https://mcp.stripe.com",
                        "management": "provider",
                        "access_token": "Bearer token-123",
                    }
                }
            }
        }
    )

    assert settings.mcp is not None
    stripe = settings.mcp.servers["stripe"]
    assert stripe.management == "provider"
    assert stripe.url == "https://mcp.stripe.com/mcp"
    assert stripe.access_token == "token-123"
    assert stripe.headers is None


def test_provider_managed_direct_url_normalizes_url_and_access_token() -> None:
    settings = Settings.model_validate(
        {
            "mcp": {
                "servers": {
                    "demo": {
                        "url": "https://demo.hf.space",
                        "management": "provider",
                        "access_token": "Bearer token-123",
                    }
                }
            }
        }
    )

    assert settings.mcp is not None
    demo = settings.mcp.servers["demo"]
    assert demo.management == "provider"
    assert demo.url == "https://demo.hf.space/mcp"
    assert demo.access_token == "token-123"
    assert demo.headers is None


def test_client_managed_access_token_synthesizes_authorization_header() -> None:
    settings = Settings.model_validate(
        {
            "mcp": {
                "servers": {
                    "demo": {
                        "url": "https://demo.hf.space",
                        "access_token": "Bearer secret-token",
                    }
                }
            }
        }
    )

    assert settings.mcp is not None
    demo = settings.mcp.servers["demo"]
    assert demo.url == "https://demo.hf.space/mcp"
    assert demo.access_token == "secret-token"
    assert demo.headers == {"Authorization": "Bearer secret-token"}


def test_target_shorthand_with_access_token_keeps_synthesized_authorization_header() -> None:
    settings = Settings.model_validate(
        {
            "mcp": {
                "servers": {
                    "demo": {
                        "target": "https://demo.hf.space",
                        "access_token": "Bearer secret-token",
                    }
                }
            }
        }
    )

    assert settings.mcp is not None
    demo = settings.mcp.servers["demo"]
    assert demo.url == "https://demo.hf.space/mcp"
    assert demo.access_token == "secret-token"
    assert demo.headers == {"Authorization": "Bearer secret-token"}


def test_resolved_target_serialization_filters_padded_synthesized_authorization_header() -> None:
    resolved_settings = MCPServerSettings.model_construct(
        name="demo",
        transport="http",
        url="https://demo.hf.space/mcp",
        management="client",
        access_token="secret-token",
        headers={" Authorization ": "Bearer secret-token"},
    )

    payload = MCPSettings._serialize_resolved_target_settings(resolved_settings)

    assert payload["headers"] is None


def test_access_token_conflicts_with_explicit_authorization_header() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate(
            {
                "mcp": {
                    "servers": {
                        "demo": {
                            "url": "https://example.com",
                            "access_token": "token-123",
                            "headers": {"Authorization": "Bearer override"},
                        }
                    }
                }
            }
        )

    assert "access_token cannot be combined with headers.Authorization" in str(exc_info.value)


def test_provider_managed_rejects_prompt_and_resource_settings() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate(
            {
                "mcp": {
                    "servers": {
                        "demo": {
                            "management": "provider",
                            "url": "https://example.com",
                            "headers": {"X-Test": "1"},
                        }
                    }
                }
            }
        )

    assert "Provider-managed MCP servers have unsupported settings" in str(exc_info.value)


def test_provider_managed_connector_requires_access_token() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate(
            {
                "mcp": {
                    "servers": {
                        "dropbox": {
                            "management": "provider",
                            "connector_id": "connector_dropbox",
                        }
                    }
                }
            }
        )

    assert "Provider-managed connectors require access_token" in str(exc_info.value)


def test_provider_managed_connector_rejects_explicit_transport() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate(
            {
                "mcp": {
                    "servers": {
                        "dropbox": {
                            "management": "provider",
                            "connector_id": "connector_dropbox",
                            "access_token": "token-123",
                            "transport": "http",
                        }
                    }
                }
            }
        )

    assert "Provider-managed MCP servers have unsupported settings: transport" in str(
        exc_info.value
    )


def test_provider_managed_connector_rejects_url_combo() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate(
            {
                "mcp": {
                    "servers": {
                        "dropbox": {
                            "management": "provider",
                            "url": "https://example.com",
                            "connector_id": "connector_dropbox",
                            "access_token": "token-123",
                        }
                    }
                }
            }
        )

    assert "exactly one of url or connector_id" in str(exc_info.value)


def test_provider_managed_connector_rejects_unknown_connector_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate(
            {
                "mcp": {
                    "servers": {
                        "dropbox": {
                            "management": "provider",
                            "connector_id": "connector_not_real",
                            "access_token": "token-123",
                        }
                    }
                }
            }
        )

    assert "connector_id must be one of:" in str(exc_info.value)


def test_load_yaml_mapping_resolves_provider_access_token_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("STRIPE_TOKEN", "secret-from-env")
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mcp:",
                "  servers:",
                "    stripe:",
                "      management: provider",
                "      url: https://mcp.stripe.com",
                "      access_token: ${STRIPE_TOKEN}",
            ]
        ),
        encoding="utf-8",
    )

    payload = load_yaml_mapping(config_path)
    settings = Settings.model_validate(payload)

    assert settings.mcp is not None
    assert settings.mcp.servers["stripe"].access_token == "secret-from-env"
