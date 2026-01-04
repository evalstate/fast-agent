"""HuggingFace authentication utilities for MCP connections."""

import os
from collections.abc import Callable
from urllib.parse import urlparse

# Type alias for token provider functions
TokenProvider = Callable[[], str | None]


def _default_hub_token_provider() -> str | None:
    """Default token provider that uses huggingface_hub.get_token()."""
    try:
        from huggingface_hub import get_token  # type: ignore

        return get_token()
    except Exception:
        return None


def is_huggingface_url(url: str) -> bool:
    """
    Check if a URL is a HuggingFace URL that should receive HF_TOKEN authentication.

    Args:
        url: The URL to check

    Returns:
        True if the URL is a HuggingFace URL, False otherwise
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname is None:
            return False

        # Check for HuggingFace domains
        if hostname in {"hf.co", "huggingface.co"}:
            return True

        # Check for HuggingFace Spaces (*.hf.space)
        # Use endswith to match subdomains like space-name.hf.space
        # but ensure exact match to prevent spoofing like evil.hf.space.com
        if hostname.endswith(".hf.space") and hostname.count(".") >= 2:
            # Additional validation: ensure it's a valid HF Space domain
            # Format should be: {space-name}.hf.space
            parts = hostname.split(".")
            if len(parts) == 3 and parts[-2:] == ["hf", "space"]:
                space_name = parts[0]
                # Validate space name: not empty, not just hyphens/dots, no spaces
                return (
                    len(space_name) > 0
                    and space_name != "-"
                    and not space_name.startswith(".")
                    and not space_name.endswith(".")
                    and " " not in space_name
                )

        return False
    except Exception:
        return False


def get_hf_token_from_env(
    hub_token_provider: TokenProvider | None = None,
) -> str | None:
    """
    Get the HuggingFace token from the HF_TOKEN environment variable.

    Falls back to `huggingface_hub.get_token()` when available, so users who have
    authenticated via `hf auth login` don't need to manually export HF_TOKEN.

    Args:
        hub_token_provider: Optional callable that returns a token. Defaults to
            using huggingface_hub.get_token(). Pass a custom provider for testing.

    Returns:
        The HF_TOKEN value if set, None otherwise
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    provider = hub_token_provider if hub_token_provider is not None else _default_hub_token_provider
    return provider()


def should_add_hf_auth(
    url: str,
    existing_headers: dict[str, str] | None,
    hub_token_provider: TokenProvider | None = None,
) -> bool:
    """
    Determine if HuggingFace authentication should be added to the headers.

    Args:
        url: The URL to check
        existing_headers: Existing headers dictionary (may be None)
        hub_token_provider: Optional callable that returns a token. Defaults to
            using huggingface_hub.get_token(). Pass a custom provider for testing.

    Returns:
        True if HF auth should be added, False otherwise
    """
    # Only add HF auth if:
    # 1. URL is a HuggingFace URL
    # 2. No existing Authorization/X-HF-Authorization header is set
    # 3. HF_TOKEN environment variable is available

    if not is_huggingface_url(url):
        return False

    # Don't add auth if Authorization or X-HF-Authorization already present
    if existing_headers:
        if "Authorization" in existing_headers or "X-HF-Authorization" in existing_headers:
            return False

    return get_hf_token_from_env(hub_token_provider) is not None


def add_hf_auth_header(
    url: str,
    headers: dict[str, str] | None,
    hub_token_provider: TokenProvider | None = None,
) -> dict[str, str] | None:
    """
    Add HuggingFace authentication header if appropriate.

    Args:
        url: The URL to check
        headers: Existing headers dictionary (may be None)
        hub_token_provider: Optional callable that returns a token. Defaults to
            using huggingface_hub.get_token(). Pass a custom provider for testing.

    Returns:
        Updated headers dictionary with HF auth if appropriate, or original headers
    """
    if not should_add_hf_auth(url, headers, hub_token_provider):
        return headers

    hf_token = get_hf_token_from_env(hub_token_provider)
    if hf_token is None:
        return headers

    # Create new headers dict or copy existing one
    result_headers = dict(headers) if headers else {}

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname and hostname.endswith(".hf.space"):
            # For .hf.space domains, send BOTH headers:
            # - Authorization: for the app's OAuth (HF infra doesn't consume this)
            # - X-HF-Authorization: for HF infrastructure (inference credit tracking)
            result_headers["Authorization"] = f"Bearer {hf_token}"
            result_headers["X-HF-Authorization"] = f"Bearer {hf_token}"
        else:
            # For other HF domains, use standard Authorization header
            result_headers["Authorization"] = f"Bearer {hf_token}"
    except Exception:
        # Fallback to standard Authorization header
        result_headers["Authorization"] = f"Bearer {hf_token}"

    return result_headers
