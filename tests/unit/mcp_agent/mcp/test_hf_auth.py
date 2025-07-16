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

    def test_hf_space_valid(self):
        assert is_huggingface_url("https://space-name.hf.space") is True

    def test_hf_space_with_path(self):
        assert is_huggingface_url("https://evalstate-parler-tts-expresso.hf.space/api/v1") is True

    def test_hf_space_with_hyphens(self):
        assert is_huggingface_url("https://my-awesome-space.hf.space") is True

    def test_hf_space_with_numbers(self):
        assert is_huggingface_url("https://space123.hf.space") is True

    def test_hf_space_http(self):
        assert is_huggingface_url("http://test-space.hf.space") is True

    def test_hf_space_with_port(self):
        assert is_huggingface_url("https://space.hf.space:8080/path") is True


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
    
    def test_hf_space_existing_x_hf_auth_with_token(self):
        """Test that existing X-HF-Authorization prevents adding auth to .hf.space."""
        original = _set_hf_token("test_token")
        try:
            headers = {"X-HF-Authorization": "Bearer existing_token"}
            assert should_add_hf_auth("https://myspace.hf.space/api", headers) is False
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

    def test_adds_auth_header_for_hf_co(self):
        """Test that hf.co domains get the standard Authorization header."""
        original = _set_hf_token("test_token_123")
        try:
            result = add_hf_auth_header("https://hf.co/models", None)
            expected = {"Authorization": "Bearer test_token_123"}
            assert result == expected
        finally:
            _restore_hf_token(original)
    
    def test_adds_auth_header_for_huggingface_co(self):
        """Test that huggingface.co domains get the standard Authorization header."""
        original = _set_hf_token("test_token_123")
        try:
            result = add_hf_auth_header("https://huggingface.co/models", None)
            expected = {"Authorization": "Bearer test_token_123"}
            assert result == expected
        finally:
            _restore_hf_token(original)
    
    def test_adds_x_hf_auth_header_for_hf_space(self):
        """Test that .hf.space domains get the X-HF-Authorization header."""
        original = _set_hf_token("test_token_123")
        try:
            result = add_hf_auth_header("https://myspace.hf.space/api", None)
            expected = {"X-HF-Authorization": "Bearer test_token_123"}
            assert result == expected
        finally:
            _restore_hf_token(original)

    def test_preserves_existing_headers_for_hf_co(self):
        """Test that existing headers are preserved when adding auth to hf.co."""
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
    
    def test_preserves_existing_headers_for_hf_space(self):
        """Test that existing headers are preserved when adding auth to .hf.space."""
        original = _set_hf_token("test_token_123")
        try:
            existing = {"Content-Type": "application/json", "User-Agent": "test"}
            result = add_hf_auth_header("https://myspace.hf.space/api", existing)
            expected = {
                "Content-Type": "application/json",
                "User-Agent": "test",
                "X-HF-Authorization": "Bearer test_token_123",
            }
            assert result == expected
        finally:
            _restore_hf_token(original)

    def test_does_not_override_existing_auth_for_hf_co(self):
        """Test that existing Authorization header is not overridden for hf.co."""
        original = _set_hf_token("test_token_123")
        try:
            existing = {"Authorization": "Bearer existing_token"}
            result = add_hf_auth_header("https://hf.co/models", existing)
            assert result == existing
        finally:
            _restore_hf_token(original)
    
    def test_does_not_override_existing_x_hf_auth_for_hf_space(self):
        """Test that existing X-HF-Authorization header is not overridden for .hf.space."""
        original = _set_hf_token("test_token_123")
        try:
            existing = {"X-HF-Authorization": "Bearer existing_token"}
            result = add_hf_auth_header("https://myspace.hf.space/api", existing)
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


class TestHfSpaceAntiSpoofing:
    """Test comprehensive anti-spoofing measures for .hf.space domains."""

    def test_hf_space_spoofing_attempts_blocked(self):
        """Test that various spoofing attempts for .hf.space domains are blocked."""
        spoofing_urls = [
            "https://evil.hf.space.com",  # suffix spoofing
            "https://malicious.hf.space.evil.com",  # domain insertion
            "https://hf.space.malicious.com",  # prefix spoofing
            "https://notreally.hf.space.attacker.net",  # complex spoofing
            "https://hf.space",  # missing space name
            "https://.hf.space",  # empty space name
            "https://..hf.space",  # double dot
            "https://sub.space.hf.space",  # too many subdomains
            "https://api.space.hf.space",  # nested subdomains
            "https://hf.space.really",  # hf.space as subdomain
        ]
        
        for url in spoofing_urls:
            assert is_huggingface_url(url) is False, f"URL should be rejected: {url}"

    def test_hf_space_case_sensitivity(self):
        """Test that case variations are handled correctly."""
        # Note: urlparse normalizes hostnames to lowercase, so all these should work
        # The validation is case-insensitive for the domain part
        assert is_huggingface_url("https://SPACE.hf.space") is True
        assert is_huggingface_url("https://space.HF.SPACE") is True
        assert is_huggingface_url("https://space.Hf.Space") is True
        assert is_huggingface_url("https://My-Space.hf.space") is True

    def test_hf_space_empty_or_invalid_space_names(self):
        """Test that invalid space names are rejected."""
        invalid_names = [
            "https://.hf.space",  # empty space name
            "https://-.hf.space",  # just hyphen
            "https://..hf.space",  # double dot
            "https:// .hf.space",  # space character (will be URL encoded)
        ]
        
        for url in invalid_names:
            assert is_huggingface_url(url) is False, f"URL should be rejected: {url}"

    def test_hf_space_path_injection_attempts(self):
        """Test that path injection attempts don't bypass validation."""
        injection_urls = [
            "https://evil.com/space.hf.space",  # path-based spoofing
            "https://attacker.net?redirect=space.hf.space",  # query param spoofing
            "https://malicious.com#space.hf.space",  # fragment spoofing
        ]
        
        for url in injection_urls:
            assert is_huggingface_url(url) is False, f"URL should be rejected: {url}"


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

    def test_no_hf_token_for_hf_space_spoofing(self):
        """Ensure HF_TOKEN is not added to .hf.space spoofing attempts."""
        original = _set_hf_token("secret_token")
        try:
            spoofing_urls = [
                "https://evil.hf.space.com",
                "https://malicious.hf.space.evil.com", 
                "https://hf.space.malicious.com",
                "https://sub.space.hf.space",
                "https://hf.space",
                "https://.hf.space",
            ]
            
            for url in spoofing_urls:
                result = add_hf_auth_header(url, None)
                # Should either be None or not contain HF token
                if result:
                    assert "Authorization" not in result or "secret_token" not in result.get("Authorization", ""), f"Token leaked to: {url}"
        finally:
            _restore_hf_token(original)

    def test_hf_token_correctly_added_to_valid_hf_spaces(self):
        """Ensure HF_TOKEN is correctly added to valid .hf.space URLs."""
        original = _set_hf_token("test_token_123")
        try:
            valid_urls = [
                "https://space-name.hf.space",
                "https://my-awesome-space.hf.space/api",
                "http://test123.hf.space:8080/path",
                "https://evalstate-parler-tts-expresso.hf.space/v1/generate",
            ]
            
            for url in valid_urls:
                result = add_hf_auth_header(url, None)
                assert result is not None, f"Should add auth to: {url}"
                assert result["X-HF-Authorization"] == "Bearer test_token_123", f"Incorrect auth for: {url}"
                # Ensure Authorization header is NOT set for .hf.space domains
                assert "Authorization" not in result, f"Should not set Authorization header for: {url}"
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