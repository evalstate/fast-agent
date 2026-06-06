"""HuggingFace authentication utilities for MCP connections."""

import os
from collections.abc import Callable, Mapping
from urllib.parse import urlparse

from fast_agent.utils.huggingface_hub import get_huggingface_hub_token
from fast_agent.utils.text import strip_casefold

# Type alias for token provider functions
TokenProvider = Callable[[], str | None]
_HF_HOSTNAMES = frozenset({"hf.co", "huggingface.co"})
_HF_AUTH_HEADER_NAMES = frozenset({"authorization", "x-hf-authorization"})


def _default_hub_token_provider() -> str | None:
    """Default token provider that uses huggingface_hub.get_token()."""
    return get_huggingface_hub_token()


def _is_hf_space_hostname(hostname: str | None) -> bool:
    if hostname is None:
        return False

    parts = hostname.split(".")
    if len(parts) != 3 or parts[-2:] != ["hf", "space"]:
        return False

    space_name = parts[0]
    return (
        bool(space_name)
        and space_name != "-"
        and not space_name.startswith(".")
        and not space_name.endswith(".")
        and " " not in space_name
    )


def _has_hf_auth_header(headers: Mapping[str, str] | None) -> bool:
    if not headers:
        return False
    return any(strip_casefold(header_name) in _HF_AUTH_HEADER_NAMES for header_name in headers)


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

        if hostname in _HF_HOSTNAMES:
            return True

        return _is_hf_space_hostname(hostname)
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
    existing_headers: Mapping[str, str] | None,
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
    if not is_huggingface_url(url):
        return False

    if _has_hf_auth_header(existing_headers):
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
        if _is_hf_space_hostname(hostname):
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
