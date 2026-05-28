from __future__ import annotations

import pytest
from pydantic import ValidationError

from fast_agent.config import Settings, load_yaml_mapping


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


def test_config_mcp_target_shorthand_preserves_load_on_start_and_overrides() -> None:
    settings = Settings.model_validate(
        {
            "mcp": {
                "servers": {
                    "secure_api": {
                        "target": "https://api.example.com",
                        "load_on_start": False,
                        "transport": "sse",
                        "url": "https://api.example.com/events/sse",
                        "headers": {"Authorization": "Bearer override"},
                    }
                }
            }
        }
    )

    assert settings.mcp is not None
    secure_api = settings.mcp.servers["secure_api"]
    assert secure_api.load_on_start is False
    assert secure_api.transport == "sse"
    assert secure_api.url == "https://api.example.com/events/sse"
    assert secure_api.headers == {"Authorization": "Bearer override"}


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


def test_config_mcp_targets_list_derives_server_aliases() -> None:
    settings = Settings.model_validate(
        {
            "mcp": {
                "targets": [
                    {"target": "https://demo.hf.space"},
                    {"target": "@modelcontextprotocol/server-filesystem /workspace"},
                ]
            }
        }
    )

    assert settings.mcp is not None
    assert "demo_hf_space" in settings.mcp.servers
    assert "server-filesystem" in settings.mcp.servers

    remote = settings.mcp.servers["demo_hf_space"]
    assert remote.transport == "http"
    assert remote.url == "https://demo.hf.space/mcp"

    filesystem = settings.mcp.servers["server-filesystem"]
    assert filesystem.transport == "stdio"
    assert filesystem.command == "npx"
    assert filesystem.args == ["@modelcontextprotocol/server-filesystem", "/workspace"]


def test_config_mcp_targets_list_allows_string_entries() -> None:
    settings = Settings.model_validate(
        {
            "mcp": {
                "targets": [
                    "@foo/bar",
                ]
            }
        }
    )

    assert settings.mcp is not None
    target = settings.mcp.servers["bar"]
    assert target.transport == "stdio"
    assert target.command == "npx"
    assert target.args == ["@foo/bar"]


def test_config_mcp_servers_override_targets_on_name_collision() -> None:
    settings = Settings.model_validate(
        {
            "mcp": {
                "targets": [
                    {
                        "target": "https://example.com",
                        "name": "demo",
                        "load_on_start": False,
                    }
                ],
                "servers": {
                    "demo": {
                        "command": "uvx",
                        "args": ["my-server"],
                    }
                },
            }
        }
    )

    assert settings.mcp is not None
    demo = settings.mcp.servers["demo"]
    assert demo.transport == "stdio"
    assert demo.command == "uvx"
    assert demo.args == ["my-server"]
    assert demo.load_on_start is True


def test_config_mcp_targets_rejects_duplicate_names_with_different_settings() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate(
            {
                "mcp": {
                    "targets": [
                        {"name": "dup", "target": "https://one.example.com"},
                        {"name": "dup", "target": "https://two.example.com"},
                    ]
                }
            }
        )

    message = str(exc_info.value)
    assert "duplicate server name 'dup'" in message
    assert "Set an explicit unique `name`" in message


def test_config_mcp_targets_rejects_embedded_cli_flags() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate(
            {
                "mcp": {
                    "targets": [
                        {"target": "https://example.com --auth token"},
                    ]
                }
            }
        )

    message = str(exc_info.value)
    assert "mcp.targets[0].target" in message
    assert "pure target string" in message
    assert "--auth" in message


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


def test_targets_list_shorthand_with_access_token_keeps_synthesized_authorization_header() -> None:
    settings = Settings.model_validate(
        {
            "mcp": {
                "targets": [
                    {
                        "name": "demo",
                        "target": "https://demo.hf.space",
                        "access_token": "Bearer secret-token",
                    }
                ]
            }
        }
    )

    assert settings.mcp is not None
    demo = settings.mcp.servers["demo"]
    assert demo.url == "https://demo.hf.space/mcp"
    assert demo.access_token == "secret-token"
    assert demo.headers == {"Authorization": "Bearer secret-token"}


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
