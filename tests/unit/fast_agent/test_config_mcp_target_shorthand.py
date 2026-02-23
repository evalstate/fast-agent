from __future__ import annotations

import pytest
from pydantic import ValidationError

from fast_agent.config import Settings


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
