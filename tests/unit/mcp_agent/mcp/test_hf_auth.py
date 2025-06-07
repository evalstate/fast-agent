"""Unit tests for HuggingFace authentication utilities.

WARNING: This test suite modifies environment variables directly during testing.
Environment variables are volatile and may be temporarily modified during test execution.
"""

import os

from mcp_agent.mcp.hf_auth import (
    add_hf_auth_header,
    get_hf_token_from_env,
    is_huggingface_url,
    should_add_hf_auth,
)


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


class TestIsHuggingfaceUrl:
    """Test URL detection for HuggingFace domains."""

    def test_hf_co_http(self):
        assert is_huggingface_url("http://hf.co/some/path") is True

    def test_hf_co_https(self):
        assert is_huggingface_url("https://hf.co/some/path") is True

    def test_huggingface_co_http(self):
        assert is_huggingface_url("http://huggingface.co/some/path") is True

    def test_huggingface_co_https(self):
        assert is_huggingface_url("https://huggingface.co/some/path") is True

    def test_subdomain_hf_co(self):
        assert is_huggingface_url("https://api.hf.co/some/path") is False

    def test_subdomain_huggingface_co(self):
        assert is_huggingface_url("https://api.huggingface.co/some/path") is False

    def test_other_domain(self):
        assert is_huggingface_url("https://example.com/some/path") is False

    def test_github_com(self):
        assert is_huggingface_url("https://github.com/some/repo") is False

    def test_invalid_url(self):
        assert is_huggingface_url("not-a-url") is False

    def test_empty_url(self):
        assert is_huggingface_url("") is False

    def test_url_with_port(self):
        assert is_huggingface_url("https://hf.co:8080/path") is True

    def test_url_with_path_and_query(self):
        assert is_huggingface_url("https://hf.co/models/gpt2?revision=main") is True


class TestGetHfTokenFromEnv:
    """Test HF_TOKEN environment variable retrieval."""

    def test_token_present(self):
        original = _set_hf_token("test_token_123")
        try:
            assert get_hf_token_from_env() == "test_token_123"
        finally:
            _restore_hf_token(original)

    def test_token_absent(self):
        original = _set_hf_token(None)
        try:
            assert get_hf_token_from_env() is None
        finally:
            _restore_hf_token(original)

    def test_token_empty_string(self):
        original = _set_hf_token("")
        try:
            assert get_hf_token_from_env() == ""
        finally:
            _restore_hf_token(original)


class TestShouldAddHfAuth:
    """Test the logic for determining when to add HF authentication."""

    def test_hf_url_no_existing_auth_with_token(self):
        original = _set_hf_token("test_token")
        try:
            assert should_add_hf_auth("https://hf.co/models", None) is True
        finally:
            _restore_hf_token(original)

    def test_hf_url_no_existing_auth_no_token(self):
        original = _set_hf_token(None)
        try:
            assert should_add_hf_auth("https://hf.co/models", None) is False
        finally:
            _restore_hf_token(original)

    def test_hf_url_existing_auth_with_token(self):
        original = _set_hf_token("test_token")
        try:
            headers = {"Authorization": "Bearer existing_token"}
            assert should_add_hf_auth("https://hf.co/models", headers) is False
        finally:
            _restore_hf_token(original)

    def test_hf_url_existing_other_headers_with_token(self):
        original = _set_hf_token("test_token")
        try:
            headers = {"Content-Type": "application/json"}
            assert should_add_hf_auth("https://hf.co/models", headers) is True
        finally:
            _restore_hf_token(original)

    def test_non_hf_url_with_token(self):
        original = _set_hf_token("test_token")
        try:
            assert should_add_hf_auth("https://example.com/api", None) is False
        finally:
            _restore_hf_token(original)

    def test_non_hf_url_no_token(self):
        original = _set_hf_token(None)
        try:
            assert should_add_hf_auth("https://example.com/api", None) is False
        finally:
            _restore_hf_token(original)


class TestAddHfAuthHeader:
    """Test adding HF authentication headers."""

    def test_adds_auth_header_when_appropriate(self):
        original = _set_hf_token("test_token_123")
        try:
            result = add_hf_auth_header("https://hf.co/models", None)
            expected = {"Authorization": "Bearer test_token_123"}
            assert result == expected
        finally:
            _restore_hf_token(original)

    def test_preserves_existing_headers(self):
        original = _set_hf_token("test_token_123")
        try:
            existing = {"Content-Type": "application/json", "User-Agent": "test"}
            result = add_hf_auth_header("https://hf.co/models", existing)
            expected = {
                "Content-Type": "application/json",
                "User-Agent": "test",
                "Authorization": "Bearer test_token_123",
            }
            assert result == expected
        finally:
            _restore_hf_token(original)

    def test_does_not_override_existing_auth(self):
        original = _set_hf_token("test_token_123")
        try:
            existing = {"Authorization": "Bearer existing_token"}
            result = add_hf_auth_header("https://hf.co/models", existing)
            assert result == existing
        finally:
            _restore_hf_token(original)

    def test_returns_original_for_non_hf_url(self):
        original = _set_hf_token("test_token_123")
        try:
            existing = {"Content-Type": "application/json"}
            result = add_hf_auth_header("https://example.com/api", existing)
            assert result == existing
        finally:
            _restore_hf_token(original)

    def test_returns_none_when_no_headers_and_no_auth_needed(self):
        original = _set_hf_token(None)
        try:
            result = add_hf_auth_header("https://example.com/api", None)
            assert result is None
        finally:
            _restore_hf_token(original)

    def test_returns_none_when_no_token_available(self):
        original = _set_hf_token(None)
        try:
            result = add_hf_auth_header("https://hf.co/models", None)
            assert result is None
        finally:
            _restore_hf_token(original)

    def test_case_sensitive_authorization_header(self):
        """Test that Authorization header check is case-sensitive as per HTTP spec."""
        original = _set_hf_token("test_token_123")
        try:
            # Lower case 'authorization' should not prevent HF auth
            existing = {"authorization": "Bearer existing_token"}
            result = add_hf_auth_header("https://hf.co/models", existing)
            expected = {
                "authorization": "Bearer existing_token",
                "Authorization": "Bearer test_token_123",
            }
            assert result == expected
        finally:
            _restore_hf_token(original)


class TestSecurityAndLeakagePrevention:
    """Test that HF_TOKEN is not leaked inappropriately."""

    def test_no_hf_token_for_github_urls(self):
        """Ensure HF_TOKEN is not added to GitHub or other non-HF URLs."""
        original = _set_hf_token("secret_token")
        try:
            result = add_hf_auth_header("https://github.com/user/repo", None)
            assert result is None
        finally:
            _restore_hf_token(original)

    def test_no_hf_token_for_arbitrary_domains(self):
        """Ensure HF_TOKEN is not added to arbitrary domains."""
        original = _set_hf_token("secret_token")
        try:
            test_urls = [
                "https://evil.com/hf.co/fake",
                "https://hf.co.evil.com/fake",
                "https://api.hf.co/models",  # subdomain
                "https://subdomain.huggingface.co/models",  # subdomain
                "https://localhost:8080/test",
                "http://127.0.0.1:3000/api",
                "https://openai.com/api",
            ]
            
            for url in test_urls:
                result = add_hf_auth_header(url, None)
                # Should either be None or not contain HF token
                if result:
                    assert "Authorization" not in result or "secret_token" not in result.get("Authorization", "")
        finally:
            _restore_hf_token(original)

    def test_respects_existing_authorization_completely(self):
        """Ensure existing Authorization headers are never modified."""
        original = _set_hf_token("hf_token")
        try:
            existing_headers = {
                "Authorization": "Bearer user_provided_token",
                "Content-Type": "application/json",
            }
            result = add_hf_auth_header("https://hf.co/models", existing_headers)
            
            # Should return exact same headers, no modification
            assert result == existing_headers
            assert result["Authorization"] == "Bearer user_provided_token"
        finally:
            _restore_hf_token(original)