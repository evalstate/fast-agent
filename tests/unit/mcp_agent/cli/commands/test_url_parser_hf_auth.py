"""Unit tests for HuggingFace authentication integration in URL parser.

WARNING: This test suite modifies environment variables directly during testing.
Environment variables are volatile and may be temporarily modified during test execution.
"""

import os

from mcp_agent.cli.commands.url_parser import parse_server_urls


def _set_hf_token(value: str | None) -> str | None:
    """Set HF_TOKEN environment variable and return the original value."""
    original = os.getenv("HF_TOKEN")
    if value is None:
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
    else:
        os.environ["HF_TOKEN"] = value
    return original


def _restore_hf_token(original_value: str | None) -> None:
    """Restore HF_TOKEN environment variable to its original value."""
    if original_value is None:
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
    else:
        os.environ["HF_TOKEN"] = original_value


class TestParseServerUrlsHfAuth:
    """Test HuggingFace authentication integration in URL parsing."""

    def test_hf_url_with_token_adds_auth_header(self):
        """Test that HF URLs get HF_TOKEN auth headers when token is available."""
        original = _set_hf_token("hf_test_token")
        try:
            result = parse_server_urls("https://hf.co/models/gpt2")
            
            assert len(result) == 1
            server_name, transport_type, url, headers = result[0]
            
            assert server_name == "hf_co"
            assert transport_type == "http"
            assert url == "https://hf.co/models/gpt2/mcp"
            assert headers is not None
            assert headers["Authorization"] == "Bearer hf_test_token"
        finally:
            _restore_hf_token(original)

    def test_hf_url_without_token_no_auth_header(self):
        """Test that HF URLs don't get auth headers when no token is available."""
        original = _set_hf_token(None)
        try:
            result = parse_server_urls("https://hf.co/models/gpt2")
            
            assert len(result) == 1
            server_name, transport_type, url, headers = result[0]
            
            assert server_name == "hf_co"
            assert transport_type == "http"
            assert url == "https://hf.co/models/gpt2/mcp"
            assert headers is None
        finally:
            _restore_hf_token(original)

    def test_hf_url_with_existing_auth_token_preserves_auth(self):
        """Test that CLI auth token is preserved even when HF_TOKEN is available."""
        original = _set_hf_token("hf_test_token")
        try:
            result = parse_server_urls("https://hf.co/models/gpt2", auth_token="user_token")
            
            assert len(result) == 1
            server_name, transport_type, url, headers = result[0]
            
            assert server_name == "hf_co"
            assert transport_type == "http"
            assert url == "https://hf.co/models/gpt2/mcp"
            assert headers is not None
            assert headers["Authorization"] == "Bearer user_token"
        finally:
            _restore_hf_token(original)

    def test_non_hf_url_with_token_no_hf_auth(self):
        """Test that non-HF URLs don't get HF_TOKEN auth headers."""
        original = _set_hf_token("hf_test_token")
        try:
            result = parse_server_urls("https://example.com/api")
            
            assert len(result) == 1
            server_name, transport_type, url, headers = result[0]
            
            assert server_name == "example_com"
            assert transport_type == "http"
            assert url == "https://example.com/api/mcp"
            assert headers is None
        finally:
            _restore_hf_token(original)

    def test_non_hf_url_with_cli_auth_token(self):
        """Test that non-HF URLs get CLI auth tokens when provided."""
        original = _set_hf_token("hf_test_token")
        try:
            result = parse_server_urls("https://example.com/api", auth_token="user_token")
            
            assert len(result) == 1
            server_name, transport_type, url, headers = result[0]
            
            assert server_name == "example_com"
            assert transport_type == "http"
            assert url == "https://example.com/api/mcp"
            assert headers is not None
            assert headers["Authorization"] == "Bearer user_token"
        finally:
            _restore_hf_token(original)

    def test_multiple_urls_mixed_hf_and_non_hf(self):
        """Test parsing multiple URLs with mixed HF and non-HF domains."""
        original = _set_hf_token("hf_test_token")
        try:
            url_list = "https://hf.co/models/gpt2,https://example.com/api,https://huggingface.co/models/bert"
            result = parse_server_urls(url_list)
            
            assert len(result) == 3
            
            # First URL - HF with token
            server_name, transport_type, url, headers = result[0]
            assert server_name == "hf_co"
            assert headers["Authorization"] == "Bearer hf_test_token"
            
            # Second URL - non-HF, no token
            server_name, transport_type, url, headers = result[1]
            assert server_name == "example_com"
            assert headers is None
            
            # Third URL - HF with token
            server_name, transport_type, url, headers = result[2]
            assert server_name == "huggingface_co"
            assert headers["Authorization"] == "Bearer hf_test_token"
        finally:
            _restore_hf_token(original)

    def test_huggingface_co_variations(self):
        """Test different variations of HuggingFace URLs."""
        original = _set_hf_token("hf_test_token")
        try:
            test_urls = [
                "http://hf.co/models",
                "https://hf.co/datasets",
                "http://huggingface.co/models",
                "https://huggingface.co/datasets",
            ]
            
            for test_url in test_urls:
                result = parse_server_urls(test_url)
                assert len(result) == 1
                server_name, transport_type, url, headers = result[0]
                assert headers is not None
                assert headers["Authorization"] == "Bearer hf_test_token"
        finally:
            _restore_hf_token(original)

    def test_sse_transport_with_hf_auth(self):
        """Test that SSE transport URLs also get HF authentication."""
        original = _set_hf_token("hf_test_token")
        try:
            result = parse_server_urls("https://hf.co/models/gpt2/sse")
            
            assert len(result) == 1
            server_name, transport_type, url, headers = result[0]
            
            assert server_name == "hf_co"
            assert transport_type == "sse"
            assert url == "https://hf.co/models/gpt2/sse"
            assert headers is not None
            assert headers["Authorization"] == "Bearer hf_test_token"
        finally:
            _restore_hf_token(original)