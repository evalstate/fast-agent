"""Hostname validation at the public MCP URL parsing boundary."""

import pytest

from fast_agent.cli.commands.url_parser import (
    generate_server_configs,
    parse_server_url,
    parse_server_urls,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://:443/mcp",
        "http://:8080/api",
        "https://user@/mcp",
        "http://user:password@/sse",
    ],
)
def test_parse_server_url_rejects_authority_without_hostname(url: str) -> None:
    with pytest.raises(ValueError, match="URL must include a hostname"):
        parse_server_url(url)


@pytest.mark.parametrize("url", ["https://:443/mcp", "https://user@/mcp"])
def test_url_list_rejects_hostless_entry_before_generating_config(url: str) -> None:
    with pytest.raises(ValueError, match="URL must include a hostname"):
        generate_server_configs(
            parse_server_urls(f"https://example.com/mcp,{url}", hub_token_provider=lambda: None)
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.com:443/mcp",
        "http://[::1]:8080/mcp",
        "http://localhost:8080/mcp",
    ],
)
def test_hostname_validation_preserves_supported_authorities(url: str) -> None:
    name, transport, parsed = parse_server_url(url)
    assert name
    assert transport == "http"
    assert parsed == url
