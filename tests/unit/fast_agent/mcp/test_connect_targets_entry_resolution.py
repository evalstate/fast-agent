from __future__ import annotations

import pytest

from fast_agent.mcp.connect_targets import resolve_target_entry


def test_resolve_target_entry_url_target_to_http_config() -> None:
    resolved_name, settings = resolve_target_entry(
        target="https://demo.hf.space",
        default_name="demo_alias",
        overrides={},
        source_path="mcp.servers.demo_alias.target",
    )

    assert resolved_name == "demo_alias"
    assert settings.name == "demo_alias"
    assert settings.transport == "http"
    assert settings.url == "https://demo.hf.space/mcp"


def test_resolve_target_entry_package_target_to_stdio_config() -> None:
    resolved_name, settings = resolve_target_entry(
        target="@foo/bar",
        default_name=None,
        overrides={},
        source_path="mcp_connect[0].target",
    )

    assert resolved_name == "bar"
    assert settings.transport == "stdio"
    assert settings.command == "npx"
    assert settings.args == ["@foo/bar"]


def test_resolve_target_entry_explicit_overrides_win() -> None:
    resolved_name, settings = resolve_target_entry(
        target="https://example.com",
        default_name="example",
        overrides={
            "transport": "sse",
            "url": "https://example.com/events/sse",
            "headers": {"Authorization": "Bearer explicit"},
            "auth": {"oauth": False},
        },
        source_path="mcp.servers.example.target",
    )

    assert resolved_name == "example"
    assert settings.transport == "sse"
    assert settings.url == "https://example.com/events/sse"
    assert settings.headers == {"Authorization": "Bearer explicit"}
    assert settings.auth is not None
    assert settings.auth.oauth is False


def test_resolve_target_entry_rejects_url_targets_with_cli_flags() -> None:
    with pytest.raises(ValueError, match="pure target string"):
        resolve_target_entry(
            target="https://demo.hf.space --auth token",
            default_name="demo",
            overrides={},
            source_path="mcp_connect[0].target",
        )
