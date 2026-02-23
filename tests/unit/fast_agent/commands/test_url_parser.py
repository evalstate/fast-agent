"""
Unit tests for the URL parser utility functions.
"""

from typing import Literal

import pytest

from fast_agent.cli.commands.url_parser import (
    generate_server_configs,
    generate_server_name,
    parse_server_url,
    parse_server_urls,
)


class TestUrlParser:
    """Tests for URL parsing utilities."""

    def test_parse_server_url_valid(self):
        """Test parsing valid URLs."""
        # HTTP URL ending with /mcp
        server_name, transport, url = parse_server_url("http://example.com/mcp")
        assert server_name == "example_com"
        assert transport == "http"
        assert url == "http://example.com/mcp"

        # HTTPS URL ending with /sse
        server_name, transport, url = parse_server_url("https://api.test.com/sse")
        assert server_name == "api_test_com"
        assert transport == "sse"
        assert url == "https://api.test.com/sse"

        # URL without /mcp or /sse gets /mcp by default
        server_name, transport, url = parse_server_url("http://localhost:8080/api")
        assert transport == "http"
        assert url == "http://localhost:8080/api/mcp"

        # Query-string endpoints are considered fully formed
        server_name, transport, url = parse_server_url("https://example.com/api?version=1")
        assert transport == "http"
        assert url == "https://example.com/api?version=1"

    def test_parse_server_url_invalid(self):
        """Test parsing invalid URLs."""
        # Empty URL
        with pytest.raises(ValueError, match="URL cannot be empty"):
            parse_server_url("")

        # Missing scheme
        with pytest.raises(ValueError, match="URL must have http or https scheme"):
            parse_server_url("example.com/mcp")

        # Invalid scheme
        with pytest.raises(ValueError, match="URL must have http or https scheme"):
            parse_server_url("ftp://example.com/mcp")

        # Missing hostname
        with pytest.raises(ValueError, match="URL must include a hostname"):
            parse_server_url("http:///mcp")

    def test_generate_server_name(self):
        """Test server name generation from URLs."""
        # Standard domain
        assert generate_server_name("http://example.com/mcp") == "example_com"

        # Domain with subdomain
        assert generate_server_name("https://api.example.com/mcp") == "api_example_com"

        # Localhost with port
        name = generate_server_name("http://localhost:8080/mcp")
        assert name.startswith("localhost_8080_")

        # IP address
        name = generate_server_name("http://192.168.1.1/api/mcp")
        assert name.startswith("192_168_1_1_")
        assert "api_mcp" in name or len(name.split("_")) > 4

        # Long domain name
        name = generate_server_name("http://very.long.domain.name:14432/api/someendpoint/mcp")
        assert "very_long_name" == name

    def test_parse_server_urls(self):
        """Test parsing multiple URLs."""
        urls = "http://example.com/mcp,https://api.test.com/sse,http://localhost:8080/api"
        result = parse_server_urls(urls)

        assert len(result) == 3

        # First URL
        assert result[0][0] == "example_com"
        assert result[0][1] == "http"
        assert result[0][2] == "http://example.com/mcp"
        assert result[0][3] is None  # No auth headers

        # Second URL
        assert result[1][0] == "api_test_com"
        assert result[1][1] == "sse"
        assert result[1][2] == "https://api.test.com/sse"
        assert result[1][3] is None  # No auth headers

        # Third URL
        assert result[2][1] == "http"
        assert result[2][2] == "http://localhost:8080/api/mcp"
        assert result[2][3] is None  # No auth headers

        # Empty input
        assert parse_server_urls("") == []

    def test_parse_server_urls_with_auth(self):
        """Test parsing URLs with authentication token."""
        urls = "http://example.com/mcp,https://api.test.com/sse"
        auth_token = "test_token_123"
        result = parse_server_urls(urls, auth_token)

        assert len(result) == 2

        # All URLs should have auth headers
        for server_name, transport, url, headers in result:
            assert headers is not None
            assert headers == {"Authorization": "Bearer test_token_123"}

    def test_parse_server_urls_with_auth_accepts_optional_bearer_prefix(self):
        """Test parsing URLs with auth values that include a Bearer prefix."""
        urls = "http://example.com/mcp,https://api.test.com/sse"
        auth_token = "Bearer test_token_123"
        result = parse_server_urls(urls, auth_token)

        assert len(result) == 2

        for server_name, transport, url, headers in result:
            assert headers is not None
            assert headers == {"Authorization": "Bearer test_token_123"}

    def test_generate_server_configs(self):
        """Test generating server configurations from parsed URLs."""
        parsed_urls: list[tuple[str, Literal["http", "sse"], str, dict[str, str] | None]] = [
            ("example_com", "http", "http://example.com/mcp", None),
            ("api_test_com", "sse", "https://api.test.com/sse", None),
        ]

        configs = generate_server_configs(parsed_urls)

        assert len(configs) == 2

        assert configs["example_com"]["transport"] == "http"
        assert configs["example_com"]["url"] == "http://example.com/mcp"
        assert "headers" not in configs["example_com"]

        assert configs["api_test_com"]["transport"] == "sse"
        assert configs["api_test_com"]["url"] == "https://api.test.com/sse"
        assert "headers" not in configs["api_test_com"]

    def test_generate_server_configs_with_auth(self):
        """Test generating server configurations with auth headers."""
        auth_headers = {"Authorization": "Bearer test_token_123"}
        parsed_urls: list[tuple[str, Literal["http", "sse"], str, dict[str, str] | None]] = [
            ("example_com", "http", "http://example.com/mcp", auth_headers),
            ("api_test_com", "sse", "https://api.test.com/sse", auth_headers),
        ]

        configs = generate_server_configs(parsed_urls)

        assert len(configs) == 2

        # Check both configs have headers
        for server_name, config in configs.items():
            assert "headers" in config
            assert config["headers"] == auth_headers

    def test_generate_server_configs_with_name_collisions(self):
        """Test handling of server name collisions."""
        # Create a list of parsed URLs with the same server name
        parsed_urls: list[tuple[str, Literal["http", "sse"], str, dict[str, str] | None]] = [
            (
                "evalstate",
                "sse",
                "https://evalstate-parler-tts-expresso.hf.space/gradio_api/mcp/sse",
                None,
            ),
            ("evalstate", "sse", "https://evalstate-shuttle.hf.space/gradio_api/mcp/sse", None),
            ("evalstate", "http", "https://evalstate-another.hf.space/gradio_api/mcp", None),
        ]

        configs = generate_server_configs(parsed_urls)

        # Should still have 3 configs despite name collisions
        assert len(configs) == 3

        # Should have unique keys by adding suffixes
        expected_keys = {"evalstate", "evalstate_1", "evalstate_2"}
        assert set(configs.keys()) == expected_keys

        # Check that URLs are preserved correctly
        urls = {config["url"] for config in configs.values()}
        assert len(urls) == 3
        assert "https://evalstate-parler-tts-expresso.hf.space/gradio_api/mcp/sse" in urls
        assert "https://evalstate-shuttle.hf.space/gradio_api/mcp/sse" in urls
        assert "https://evalstate-another.hf.space/gradio_api/mcp" in urls
