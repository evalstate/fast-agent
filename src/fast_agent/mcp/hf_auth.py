"""HuggingFace authentication utilities for hosted and remote connections."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from fast_agent.utils.huggingface_hub import get_huggingface_hub_token

# Type alias for token provider functions
TokenProvider = Callable[[], str | None]
HFAuthHeader = Literal["Authorization", "X-HF-Authorization"]


def _default_hub_token_provider() -> str | None:
    """Default token provider that uses huggingface_hub.get_token()."""
    return get_huggingface_hub_token()


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


def is_hf_space_url(url: str) -> bool:
    """Return True when ``url`` is a validated Hugging Face Space hostname."""
    if not is_huggingface_url(url):
        return False
    try:
        hostname = urlparse(url).hostname
    except Exception:
        return False
    return bool(hostname and hostname.endswith(".hf.space"))


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


def _has_auth_header(headers: dict[str, str] | None) -> bool:
    if not headers:
        return False
    return any(key.lower() in {"authorization", "x-hf-authorization"} for key in headers)


def _bearer(value: str) -> str:
    return f"Bearer {value}"


@dataclass(frozen=True)
class HuggingFaceAuthPolicy:
    """Policy for attaching Hugging Face bearer credentials to outbound requests.

    The policies below intentionally keep ambient Hugging Face credentials separate
    from explicit server authentication. Ambient HF tokens use X-HF-Authorization
    for Spaces so they can be consumed by Space apps without taking over app-level
    Authorization. Explicit auth, including --auth and OAuth challenges, uses the
    standard Authorization header because it is authenticating to that endpoint.
    """

    hf_space_header: HFAuthHeader

    def add_ambient_hf_token(
        self,
        url: str,
        headers: dict[str, str] | None,
        hub_token_provider: TokenProvider | None = None,
    ) -> dict[str, str] | None:
        if not is_huggingface_url(url) or _has_auth_header(headers):
            return headers

        hf_token = get_hf_token_from_env(hub_token_provider)
        if hf_token is None:
            return headers

        return self.add_bearer_token(url, headers, hf_token)

    def add_bearer_token(
        self,
        url: str,
        headers: dict[str, str] | None,
        token: str,
    ) -> dict[str, str]:
        result_headers = dict(headers) if headers else {}
        result_headers[self.header_for_url(url)] = _bearer(token)
        return result_headers

    def header_for_url(self, url: str) -> HFAuthHeader:
        return self.hf_space_header if is_hf_space_url(url) else "Authorization"


HF_CLI_AMBIENT_AUTH_POLICY = HuggingFaceAuthPolicy(
    hf_space_header="X-HF-Authorization",
)
HF_EXPLICIT_BEARER_AUTH_POLICY = HuggingFaceAuthPolicy(
    hf_space_header="Authorization",
)
HF_REQUEST_PASSTHROUGH_AUTH_POLICY = HuggingFaceAuthPolicy(
    hf_space_header="X-HF-Authorization",
)


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
    if _has_auth_header(existing_headers):
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
    return HF_CLI_AMBIENT_AUTH_POLICY.add_ambient_hf_token(
        url,
        headers,
        hub_token_provider,
    )


def add_explicit_bearer_auth_header(
    url: str,
    headers: dict[str, str] | None,
    token: str,
) -> dict[str, str]:
    """Add explicit bearer auth for a target endpoint.

    This is the policy behind ``--auth`` and OAuth-managed A2A/MCP endpoints.
    It uses Authorization even for ``*.hf.space`` because the credential is for
    the target server itself rather than an ambient HF token for a Space app.
    """
    return HF_EXPLICIT_BEARER_AUTH_POLICY.add_bearer_token(url, headers, token)


def add_forwarded_hf_auth_header(url: str, headers: dict[str, str] | None) -> dict[str, str] | None:
    """Add the request-scoped bearer token to Hugging Face URLs.

    This is intended for hosted agents that should call Hugging Face services as the
    inbound user rather than as the Space/server process. Existing auth headers are
    preserved.
    """
    if not is_huggingface_url(url):
        return headers

    if _has_auth_header(headers):
        return headers

    from fast_agent.mcp.auth.context import request_bearer_token

    token = request_bearer_token.get()
    if not token:
        return headers

    return HF_REQUEST_PASSTHROUGH_AUTH_POLICY.add_bearer_token(url, headers, token)
