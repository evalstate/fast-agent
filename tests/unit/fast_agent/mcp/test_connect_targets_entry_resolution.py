from __future__ import annotations

import pytest

from fast_agent.mcp.connect_targets import resolve_target_entry


def test_resolve_target_entry_url_target_to_http_config() -> None:
    resolved_entry = resolve_target_entry(
        target="https://demo.hf.space",
        default_name="demo_alias",
        overrides={},
        source_path="mcp.servers.demo_alias.target",
    )

    assert resolved_entry.server_name == "demo_alias"
    assert resolved_entry.settings.name == "demo_alias"
    assert resolved_entry.settings.transport == "http"
    assert resolved_entry.settings.url == "https://demo.hf.space/mcp"


def test_resolve_target_entry_package_target_to_stdio_config() -> None:
    resolved_entry = resolve_target_entry(
        target="@foo/bar",
        default_name=None,
        overrides={},
        source_path="mcp_connect[0].target",
    )

    assert resolved_entry.server_name == "bar"
    assert resolved_entry.settings.transport == "stdio"
    assert resolved_entry.settings.command == "npx"
    assert resolved_entry.settings.args == ["@foo/bar"]


def test_resolve_target_entry_explicit_overrides_win() -> None:
    resolved_entry = resolve_target_entry(
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

    assert resolved_entry.server_name == "example"
    assert resolved_entry.settings.transport == "sse"
    assert resolved_entry.settings.url == "https://example.com/events/sse"
    assert resolved_entry.settings.headers == {"Authorization": "Bearer explicit"}
    assert resolved_entry.settings.auth is not None
    assert resolved_entry.settings.auth.oauth is False


def test_resolve_target_entry_provider_managed_keeps_normalized_url() -> None:
    resolved_entry = resolve_target_entry(
        target="https://demo.hf.space",
        default_name="demo",
        overrides={"management": " PROVIDER "},
        source_path="mcp.servers.demo.target",
    )

    assert resolved_entry.server_name == "demo"
    assert resolved_entry.settings.management == "provider"
    assert resolved_entry.settings.transport == "http"
    assert resolved_entry.settings.url == "https://demo.hf.space/mcp"


def test_resolve_target_entry_rejects_url_targets_with_cli_flags() -> None:
    with pytest.raises(ValueError, match="pure target string"):
        resolve_target_entry(
            target="https://demo.hf.space --auth token",
            default_name="demo",
            overrides={},
            source_path="mcp_connect[0].target",
        )
